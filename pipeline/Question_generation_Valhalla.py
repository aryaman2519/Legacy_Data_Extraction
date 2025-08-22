import os
import spacy
import pandas as pd
from tqdm import tqdm
from groq import Groq

# === CONFIG ===
input_path = "full_pdf_text.txt"
output_csv = "generated_questions_from_txt.csv"
max_questions_per_chunk = 5
min_answer_len = 3
max_answer_len = 20

# === LOAD MODELS ===
print("🔄 Loading models...")
nlp = spacy.load("en_core_web_sm")

# === INIT GROQ CLIENT ===
client = Groq(api_key=os.getenv("fill your groq api key here"))  # ⚡ put your Groq API key in env

# === UTILITIES ===
def extract_answer_candidates(text):
    doc = nlp(text)
    answers = set()

    # Named entities
    for ent in doc.ents:
        if min_answer_len <= len(ent.text.split()) <= max_answer_len:
            answers.add(ent.text.strip())

    # Noun phrases
    for np in doc.noun_chunks:
        if min_answer_len <= len(np.text.split()) <= max_answer_len:
            answers.add(np.text.strip())

    return list(answers)

def generate_question(context, answer):
    if answer not in context:
        return None

    highlighted = context.replace(answer, f"<hl> {answer} <hl>", 1)
    prompt = f"""
You are a question generation system.
Context: {highlighted}
Answer: {answer}
Generate a natural question whose answer is exactly "{answer}".
"""

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # ✅ Groq model name
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ API error: {e}")
        return None

# === READ AND CHUNK TEXT ===
with open(input_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

chunks = [chunk.strip() for chunk in raw_text.split("\n\n") if len(chunk.strip()) > 80]
print(f"📄 Processing {len(chunks)} chunks...")

# === GENERATE QUESTIONS ===
qa_pairs = []
for idx, chunk in tqdm(enumerate(chunks), total=len(chunks)):
    try:
        candidates = extract_answer_candidates(chunk)
        used = 0
        for answer in candidates:
            if used >= max_questions_per_chunk:
                break
            question = generate_question(chunk, answer)
            if question:
                qa_pairs.append((question, answer, chunk))
                used += 1
    except Exception as e:
        print(f"⚠️ Error in chunk {idx + 1}: {e}")

# === SAVE TO CSV ===
df = pd.DataFrame(qa_pairs, columns=["Question", "Answer", "Context"])
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved {len(df)} question-answer pairs to: {output_csv}")
