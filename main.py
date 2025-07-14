import os
from openai import OpenAI
import glob
import uuid
import sys
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import hdbscan
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import ast

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DATA_DIR = "data/"
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_store/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")
TOP_K = int(os.getenv("TOP_K", "5"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_documents():
    file_paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    docs = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            filename = os.path.basename(path).replace(".txt", "").lower()
            metadata = {
                "source": filename,
                "company": filename.split("-")[0],
                "doc_id": str(uuid.uuid4())
            }
            docs.append(Document(page_content=content, metadata=metadata))
    return docs


def chunk_risk_section(text):

    # bullet/numbered list splitting
    bullet_splits = re.split(r"(?m)^\s*(?:[-*•●]|\d+\.)\s+", text)
    bullet_splits = [chunk.strip() for chunk in bullet_splits if chunk.strip()]

    # If too small or poorly split, fallback to smarter recursive splitter
    if len(bullet_splits) < 3 or max(len(s) for s in bullet_splits) < 100:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_text(text)

    # Merge bullet chunks into ~800 char chunks
    merged_chunks = []
    buffer = ""
    for section in bullet_splits:
        if len(buffer) + len(section) < 800:
            buffer += " " + section
        else:
            if len(buffer.strip().split()) >= 20:
                merged_chunks.append(buffer.strip())
            buffer = section
    if len(buffer.strip().split()) >= 20:
        merged_chunks.append(buffer.strip())

    return merged_chunks


def chunk_mda_section(text):

    table_split = re.split(r"\n\s*-{5,}\s*\n", text)
    chunks = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for section in table_split:
        if len(section) < 500:
            continue

        digit_count = len(re.findall(r"\d", section))
        tab_count = section.count("\t")
        line_count = len(section.splitlines())
        char_count = len(section)

        contains_table = (
            digit_count > 50 and
            tab_count > 5 and
            digit_count / max(1, char_count) > 0.05 and
            line_count >= 3   
        )

        split_subsections = splitter.split_text(section)
        for chunk in split_subsections:
            chunks.append((chunk.strip(), contains_table))

    return chunks


def chunk_earnings_call(text):

    speaker_pattern = re.compile(
        r"^(?P<speaker>[A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(\s*[-–—]{1,2}\s*.+)?$"
    )

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    chunks = []
    current_chunk = []

    for i, line in enumerate(lines):
        if speaker_pattern.match(line):
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
        current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    if len(chunks) < 10:
        print("Still few turns detected, using fallback splitter...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_text(text)

    return chunks


def detect_document_themes(docs, min_cluster_size=5):
    print("🔎 Detecting themes across documents with HDBSCAN...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [doc.page_content for doc in docs]
    embeddings = model.encode(texts)

    if len(embeddings) < 2:
        raise ValueError("Not enough documents to cluster.")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)

    for i, doc in enumerate(docs):
        label = labels[i]
        doc.metadata["theme"] = f"theme_{label}" if label != -1 else "noise"

    return docs


def annotate_patterns_with_llm(text):
    if len(text.split()) < 50:
        return ""
    prompt = f"""
You are a financial analyst. Identify 1–3 abstract business or risk themes present in the following text.

Text:
\"\"\"
{text}
\"\"\"

Respond ONLY with a Python list of short lowercase strings. Example: ["regulatory risk", "supply chain disruption"]
"""
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        
        # Try parsing only if it looks like a list
        if content.startswith("[") and content.endswith("]"):
            tags = ast.literal_eval(content)
        else:
            tags = [x.strip().lower() for x in content.split(",") if x.strip()]

        if isinstance(tags, list):
            return ", ".join(tag for tag in tags if isinstance(tag, str))
        return ""
    except Exception as e:
        print("LLM tagging failed:", e)
        return ""

def split_by_theme(docs):
    print("Splitting documents by detected theme groups...")
    theme_groups = {}
    for doc in docs:
        theme = doc.metadata.get("theme", "unknown")
        theme_groups.setdefault(theme, []).append(doc)

    all_chunks = []

    for theme, docs_in_group in theme_groups.items():
        for doc in docs_in_group:
            content = doc.page_content
            source = doc.metadata.get("source", "").lower()
            # Prepare base_meta without theme, chunk, section
            base_meta = {k: v for k, v in doc.metadata.items() if k not in ("theme", "chunk", "section")}

            if "risk" in source:
                print(f"Using risk-aware chunking for {source}")
                chunks = chunk_risk_section(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            **base_meta,
                            "theme": theme,
                            "chunk": i,
                            "section": "item_1a",
                            "abstraction_level": "sentence"
                        }
                    ))
            elif "mda" in source:
                print(f"Using MDA-aware chunking for {source}")
                chunk_pairs = chunk_mda_section(content)
                for i, (chunk, contains_table) in enumerate(chunk_pairs):
                    group_id = f"{base_meta['source']}-mda-{i}"
                    paragraph_metadata = {
                        **base_meta,
                        "theme": theme,
                        "chunk": i,
                        "section": "item_7",
                        "abstraction_level": "paragraph",
                        "chunk_group_id": group_id,
                        "contains_table": contains_table,
                        "patterns": annotate_patterns_with_llm(chunk)
                    }
                    all_chunks.append(Document(page_content=chunk, metadata=paragraph_metadata))

                    if not contains_table:
                        sentences = chunk.split(". ")
                        for j, sent in enumerate(sentences):
                            if len(sent.strip()) < 30:
                                continue
                            sentence_metadata = {
                                **base_meta,
                                "theme": theme,
                                "chunk": f"{i}.{j}",
                                "section": "item_7",
                                "abstraction_level": "sentence",
                                "chunk_group_id": group_id,
                                "patterns": annotate_patterns_with_llm(sent)
                            }
                            all_chunks.append(Document(page_content=sent.strip(), metadata=sentence_metadata))
            elif "earnings" in source or "earning" in source:
                print(f"Using earnings-aware chunking for {source}")
                chunks = chunk_earnings_call(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append(Document(
                        page_content=chunk,
                        metadata={
                            **base_meta,
                            "theme": theme,
                            "chunk": i,
                            "section": "earnings_call",
                            "abstraction_level": "turn"
                        }
                    ))
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents([Document(page_content=content, metadata=base_meta)])
                for i, chunk_doc in enumerate(split_docs):
                    chunk_doc.metadata["theme"] = theme
                    chunk_doc.metadata["chunk"] = i
                    chunk_doc.metadata["abstraction_level"] = "paragraph"
                    all_chunks.append(chunk_doc)

    return all_chunks


def embed_and_upsert(chunks):
    print("Embedding and upserting into Chroma vectorstore...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
    db.persist()
    print(f"Indexed {len(chunks)} chunks.")


def main():
    print("Loading raw documents...")
    docs = load_documents()

    themed_docs = detect_document_themes(docs)
    chunks = split_by_theme(themed_docs)
    embed_and_upsert(chunks)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if mode == "pipeline":
        main()
    elif mode == "query":
        from query import query_mode
        query_mode()
    elif mode == "eval":
        from eval.run_eval import run_evaluation
        run_evaluation()
    else:
        print(f"Unknown mode: {mode}")