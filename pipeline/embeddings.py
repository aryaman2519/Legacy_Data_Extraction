import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# === CONNECT TO MONGODB ===
client = MongoClient("add your mongo uri here")
db = client["pdf_summaries"]
collection = db["structured_documents"]

# === LOAD EMBEDDING MODEL ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === FUNCTION TO GENERICALLY EMBED CSV ROWS ===
def embed_csv_generic(csv_file):
    df = pd.read_csv(csv_file)
    
    print("Columns detected:", df.columns.tolist())

    for idx, row in df.iterrows():
        row_dict = row.to_dict()  # convert the row to a dict

        # For each textual field, create embeddings
        embeddings_dict = {}
        for col, val in row_dict.items():
            if isinstance(val, str) and val.strip():  # only embed non-empty strings
                embeddings_dict[col + "_embedding"] = embedder.encode(val).tolist()

        # Merge original data with embeddings
        mongo_doc = {**row_dict, **embeddings_dict, "document_id": f"ROW_{idx+1}"}

        # Insert into MongoDB
        collection.insert_one(mongo_doc)

    print("✅ CSV stored in MongoDB with embeddings for all textual fields.")

# === EXECUTE ===
csv_file = "generated_questions_from_txt.csv"
embed_csv_generic(csv_file)
