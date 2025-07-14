import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
CHROMA_DB_PATH = "chroma_store/"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# --- Load vector store ---
print("🔍 Loading Chroma vector store...")
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding)

# --- Fetch documents + metadata ---
store = db.get()
docs = store["documents"]
metas = store["metadatas"]

# --- Convert to DataFrame ---
df = pd.DataFrame(metas)

print("\n📊 Chunk Count:", len(df))
print("\n🔑 Columns:", df.columns.tolist())

# --- Pattern usage ---
pattern_counts = df["patterns"].apply(lambda x: bool(x.strip()) if isinstance(x, str) else False).value_counts()
print("\n✅ Pattern Metadata Usage:")
print(pattern_counts.rename(index={True: "Has Patterns", False: "No Patterns"}))

# --- Section distribution ---
print("\n📂 Section Breakdown:")
print(df["section"].value_counts())

# --- Abstraction level breakdown ---
if "abstraction_level" in df.columns:
    print("\n🧠 Abstraction Levels:")
    print(df["abstraction_level"].value_counts())

# --- Table content detection (for MDA only) ---
if "contains_table" in df.columns:
    print("\n📌 Table Content in MDA Chunks:")
    print(df[df["section"] == "item_7"]["contains_table"].value_counts())

# --- Optional: Save to CSV for manual inspection ---
df.to_csv("chunk_metadata_export.csv", index=False)
print("\n💾 Metadata exported to chunk_metadata_export.csv")