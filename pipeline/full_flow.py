# full_flow.py
import os
import re
from typing import List, Tuple

import spacy
from tqdm import tqdm
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdf2image import convert_from_path
from pytesseract import pytesseract

from database_name_decider import get_document_heading  # ✅ your LLaMA title/domain generator

# =========================
# === CONFIGURATION =======
# =========================
pytesseract.tesseract_cmd = r"add your tesseract path here"
POPLER_PATH = r"add your poppler path here"
MONGO_URI = "add your mongo uri here"

# QG & candidate extraction config
MAX_QUESTIONS_PER_CHUNK = 5
MIN_ANSWER_LEN = 3
MAX_ANSWER_LEN = 20

# =========================
# === MONGO CLIENT ========
# =========================
mongo_client = MongoClient(MONGO_URI)

# =========================
# === LOAD MODELS (ONCE) ==
# =========================
print("🔄 Loading NLP & embedding models...")
nlp = spacy.load("en_core_web_sm")
qg_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
qg_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# === HELPERS =============
# =========================
def pdf_to_text(pdf_path: str, poppler_path: str) -> str:
    """Convert PDF to raw OCR text using Tesseract + Poppler."""
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
    all_text = ""
    for i, image in enumerate(pages):
        raw_text = pytesseract.image_to_string(image)
        all_text += f"\n\n--- Page {i + 1} ---\n{raw_text}"
    return all_text


def sanitize_db_name(name: str) -> str:
    """Make a safe MongoDB database name."""
    sanitized = re.sub(r"[^a-zA-Z0-9]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized.lower() if sanitized else "document_db"


def extract_answer_candidates(text: str) -> List[str]:
    """Find candidate answers using NER + noun phrases."""
    doc = nlp(text)
    answers = set()
    for ent in doc.ents:
        if MIN_ANSWER_LEN <= len(ent.text.split()) <= MAX_ANSWER_LEN:
            answers.add(ent.text.strip())
    for np in doc.noun_chunks:
        if MIN_ANSWER_LEN <= len(np.text.split()) <= MAX_ANSWER_LEN:
            answers.add(np.text.strip())
    return list(answers)


def generate_question(context: str, answer: str) -> str | None:
    """Generate a question given an answer inside the context."""
    if answer not in context:
        return None
    highlighted = context.replace(answer, f"<hl> {answer} <hl>", 1)
    input_text = f"generate question: {highlighted}"
    inputs = qg_tokenizer([input_text], return_tensors="pt", truncation=True)
    outputs = qg_model.generate(
        **inputs,
        max_new_tokens=64,
        num_beams=4,
        num_return_sequences=1,
        early_stopping=True,
    )
    return qg_tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# === MAIN PIPELINE =======
# =========================
def process_pdf(pdf_path: str, poppler_path: str = POPLER_PATH) -> Tuple[str, List[str], any]:
    """
    Full pipeline:
      1) OCR text from PDF
      2) Detect overall document heading (db name source)
      3) Split into chunks, detect per-chunk headings (domains)
      4) Generate Q&A
      5) Store in MongoDB with embeddings
    Returns: (safe_db_name, unique_headings, collection)
    """
    print("📄 Extracting text from PDF...")
    raw_text = pdf_to_text(pdf_path, poppler_path)

    # Overall document heading -> DB name
    doc_heading = get_document_heading(raw_text)
    print(f"📌 Document heading detected: {doc_heading}")

    safe_db_name = sanitize_db_name(doc_heading)
    print(f"💡 Using MongoDB database name: {safe_db_name}")

    db = mongo_client[safe_db_name]
    collection = db["structured_documents"]

    # Chunking (simple paragraph split; adjust as needed)
    chunks = [c.strip() for c in raw_text.split("\n\n") if len(c.strip()) > 80]
    print(f"🧩 Generating questions from {len(chunks)} chunks...")

    qa_pairs = []
    headings_all = []

    for idx, chunk in tqdm(list(enumerate(chunks, start=1)), total=len(chunks)):
        try:
            # Per-chunk heading (domain)
            chunk_heading = get_document_heading(chunk)
            headings_all.append(chunk_heading)

            candidates = extract_answer_candidates(chunk)
            used = 0
            for answer in candidates:
                if used >= MAX_QUESTIONS_PER_CHUNK:
                    break
                q = generate_question(chunk, answer)
                if not q:
                    continue

                qa_pairs.append((q, answer, chunk, chunk_heading))

                # Store immediately
                doc_dict = {
                    "heading": chunk_heading,        # domain
                    "question": q,
                    "answer": answer,
                    "context": chunk,
                    "question_embedding": embedder.encode(q).tolist(),
                    "answer_embedding": embedder.encode(answer).tolist(),
                    "context_embedding": embedder.encode(chunk).tolist(),
                }
                collection.insert_one(doc_dict)
                used += 1
        except Exception as e:
            print(f"⚠️ Error in chunk {idx}: {e}")

    # Store metadata
    db["metadata"].insert_one({
        "document_title": doc_heading,
        "database_name": safe_db_name,
        "total_chunks": len(chunks),
        "total_qa_pairs": len(qa_pairs),
    })

    # Unique headings (preserve order)
    seen = set()
    unique_headings = [h for h in headings_all if not (h in seen or seen.add(h))]

    print(f"✅ Stored {len(qa_pairs)} Q&A pairs in MongoDB under '{safe_db_name}'.")
    return safe_db_name, unique_headings, collection


if __name__ == "__main__":
    path = input("Enter PDF path: ").strip()
    process_pdf(path)
