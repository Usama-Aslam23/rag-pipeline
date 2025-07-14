# Abstract Pattern-Aware RAG Chunking Pipeline

This project implements a RAG (Retrieval-Augmented Generation) chunking pipeline designed to identify and represent **abstract multi-document patterns**. It supports structured processing of 10-K filings (MDA, risks) and earnings call transcripts, enriching each chunk with metadata for more context-aware retrieval.

---

## 📁 Pipeline Overview

### 1. Document Ingestion
- All `.txt` files are extracted from a local `data/` directory.
- Documents include **MDA**, **risk disclosures**, and **earnings call transcripts**.

### 2. Pattern Identification & Chunking
- **Risks**: Chunked by bullet hierarchy and filtered for dense thematic content.
- **Calls**: Speaker turns are extracted and chunked while preserving conversational coherence.
- **MDA**: Abstract themes are inferred using clustering (HDBSCAN) or optionally tagged via LLM.

### 3. Metadata Enrichment
- Chunks are tagged with:
  - `company_name`
  - `doc_type`
  - `section`
  - Optional: `pattern` (abstract topic/theme)

### 4. Vector Store Upsertion
- Processed chunks are upserted into a local [LightRAG](https://github.com/langchain-ai/light-rag) instance for downstream retrieval.

---

## 🧪 Evaluation Framework

Evaluations are stored in `eval/eval_set.json` and test:
- ✅ Retrieval of the correct document chunks
- ✅ Answer generation fidelity (using GPT-3.5 as grader)

---

## 🧠 Reasoning & Assumptions

See [`REASONING.md`](./REASONING.md) for:
- Why reference answers are GPT-generated from retrieved chunks
- Justification for structural chunking approaches
- Thematic tagging for analysis coverage
- Query rewriting to align user intent with document language

---

## ⚙️ Environment Setup

1. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create a `.env` file** based on the provided template:
   ```bash
   cp .env.example .env
   ```

4. **Populate `.env`** with your OpenAI API key and model preferences:
   ```
   OPENAI_API_KEY=your-api-key
   GPT_MODEL=gpt-3.5-turbo
   TOP_K=8
   CHROMA_DB_PATH=chroma_store/
   ```

---

## Running the App

This pipeline has two main modes: document ingestion + chunking, and evaluation via query answering.

### 1. Run Chunking Pipeline (Ingest + Upsert)
Extracts, processes, and indexes documents into Chroma vector store for downstream RAG.

```bash
python main.py pipeline
```

### 2. Run Evaluations
Executes evaluation queries defined in `eval/eval_set.json`, logs results and saves a summary.

```bash
python main.py eval
```

### 3. Run Ad-hoc Query Script
To test a single query interactively using the retrieval pipeline:

```bash
python main.py query
```

Make sure your `.env` file includes your OpenAI key and Chroma settings.

---

## 📄 File Structure

```
├── main.py                # Entrypoint to run the pipeline
├── data/                  # Input directory containing .txt files
├── eval/
│   ├── eval_set.json      # Evaluation dataset
│   ├── run_eval.py        # Script to run evals
│   └── results/           # Outputs of eval runs
├── REASONING.md           # Design decisions and tradeoffs
├── README.md              # This file
├── query.py               # Interactive script to test queries via retrieval pipeline
└── .env.example           # Example environment config
```
