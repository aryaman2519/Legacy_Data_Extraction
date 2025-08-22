# database_name_decider.py
import re
from huggingface_hub import InferenceClient

# =========================
# === HF API CONFIGURATION ===
# =========================
HF_API_TOKEN = "fill your lama api key here"  # replace with your actual key
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Initialize Hugging Face client
hf_client = InferenceClient(HF_MODEL, token=HF_API_TOKEN)


def get_document_heading(raw_text, max_chars=50):
    """
    Generate a concise, meaningful title for a document using LLaMA 3 API.
    If API fails, prompt the user to manually enter a title.
    """

    prompt = (
        "You are an AI assistant. Generate a short and meaningful title "
        "for the following document (less than 10 words). "
        "Return ONLY the title without quotes or punctuation:\n\n"
        f"{raw_text[:2000]}"
    )

    try:
        print("📡 Requesting title from Hugging Face API...")
        response = hf_client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30
        )

        print("🔍 Raw API Response:", response)

        heading = response.choices[0].message["content"].strip()

        # sanitize for MongoDB DB name
        heading = "".join(c for c in heading if c.isalnum() or c in "_- ")
        heading = re.sub(r'_+', '_', heading).strip('_')

        if not heading:
            raise ValueError("Empty heading from LLaMA API")

        return heading[:max_chars]

    except Exception as e:
        print(f"⚠️ AI title generation failed: {e}")
        # Manual fallback
        manual = input("❓ Please enter a title for this document: ").strip()
        if manual:
            heading = "".join(c for c in manual if c.isalnum() or c in "_- ")
            return heading[:max_chars]
        return "document_db"
