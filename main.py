import os
from openai import OpenAI
import glob
import sys
import re
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from lightrag.llm.hf import hf_embed
import asyncio
from lightrag import LightRAG
from transformers import AutoTokenizer, AutoModel
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc

load_dotenv()


DATA_DIR = "data/"
WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en")

def load_documents():
    file_paths = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    docs = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            filename = os.path.basename(path).replace(".txt", "").lower()
            metadata = {
                "source": filename
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
        
        split_subsections = splitter.split_text(section)
        for chunk in split_subsections:
            chunks.append((chunk.strip()))

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


def chunk_data(docs):
    print("Chunking documents")
    all_chunks = []
    for doc in docs:
        
        if "risk" in doc.metadata.get("source"):
            print(f"Using risk-aware chunking for {doc.metadata.get('source')}")
            chunks = chunk_risk_section(doc.page_content)
            for i, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata = {"source": doc.metadata.get("source")}
                ))
        elif "mda" in doc.metadata.get("source"):
            print(f"Using MDA-aware chunking for {doc.metadata.get('source')}")
            chunks = chunk_mda_section(doc.page_content)
            for i, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata = {"source": doc.metadata.get("source")}
                ))
        elif "earnings" in doc.metadata.get("source") or "earning" in doc.metadata.get("source"):
            print(f"Using earnings-aware chunking for {doc.metadata.get('source')}")
            chunks = chunk_earnings_call(doc.page_content)
            for i, chunk in enumerate(chunks):
                all_chunks.append(Document(
                    page_content=chunk,
                    metadata = {"source": doc.metadata.get("source")}
                ))
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents([Document(page_content=doc.page_content, metadata={"source": doc.metadata.get("source")})])
            for i, chunk_doc in enumerate(split_docs):
                all_chunks.append(Document(
                    page_content=chunk_doc,
                    metadata = {"source": doc.metadata.get("source")}
                ))


    return all_chunks




setup_logger("lightrag", level="ERROR")
embedding_func= EmbeddingFunc(
            embedding_dim=768,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    EMBEDDING_MODEL
                ),
                embed_model=AutoModel.from_pretrained(
                    EMBEDDING_MODEL
                ),
            ),
        )


async def embed_and_upsert(chunks):
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    import faiss
    faiss.omp_set_num_threads(8)
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func= embedding_func,
        llm_model_func=gpt_4o_mini_complete,
        vector_storage="FaissVectorDBStorage",
        chunk_token_size=800,
        chunk_overlap_token_size=100,
        embedding_batch_num=128,
        enable_llm_cache=True,
        max_parallel_insert=16
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    def _ensure_str(x):
        if isinstance(x, str):
            return x
        if isinstance(x, list):
            return " ".join(map(str, x))
        return str(x)


    texts = [_ensure_str(doc.page_content) for doc in chunks]


    await rag.ainsert(texts)

    await rag.finalize_storages()
    

def main():
    print("Loading raw documents...")
    docs = load_documents()
    chuncked_data = chunk_data(docs)

    asyncio.run(embed_and_upsert(chuncked_data))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    if mode == "pipeline":
        main()
    elif mode == "query":
        from query import main
        asyncio.run(main())
    elif mode == "eval":
        from eval.run_eval import run_evaluation
        asyncio.run(run_evaluation())
    else:
        print(f"Unknown mode: {mode}")