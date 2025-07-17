# REASONING.md

## 🔍 Assignment Overview

The task was to build a chunking pipeline for a Retrieval-Augmented Generation (RAG) system using LightRAG, focused on identifying and representing abstract multi-document patterns from text data.

---


## 🧩 Critical Design Reasoning

### 1. **Dataset Selection**

**Decision:** Use a mix of 10-K filings (MDA and Risk sections) and earnings call transcripts from various tech companies.

**Why:**
  
- They represent high-value, information-dense, and pattern-rich content.
- Abstract themes such as regulatory risk, AI investment, and macroeconomic outlook recur across different formats.
- These sources are widely used in real-world market and risk analysis, making them ideal for testing a pattern-aware RAG pipeline.
- Their inherent structural diversity (narrative text, bullets, dialogue, and semi-tabular content) provided a robust basis to stress-test chunking and retrieval strategies.


### 2. **Chunking Strategy by Document Type**

> **Note on LLM-based chunking:** An initial experiment asked GPT-4 to mark chunk boundaries directly on entire filings. In practice, 30-k-token inputs exceeded context limits, replies were often truncated, and the per-document cost was prohibitive. We therefore adopted rule-based splitters tailored to each document type.

#### a. **Risk Sections**

- **Primary rule:** split on bullet or numbered headings, preserving each risk factor as a self-contained chunk.  
- **Fallback:** if bullet structure is missing, apply a recursive character splitter capped at ~800 chunk size to maintain readability.

#### b. **Earnings-Call Transcripts**

- **Boundary:** speaker turns (Name, Operator).  
- **Post-processing:** merge consecutive turns from the same speaker and absorb ultra-short interjections so each chunk carries substantive content.

#### c. **Management Discussion & Analysis (MD&A)**

- **Tables:** detect tables based on numeric data, tabs and pipes 
- **Fallback:** apply a recursive character splitter capped at ~1000 chunk size to maintain readability.

---


### 3. **Embedding Model Selection**

**Choice:** `BAAI/bge-small-en` via HuggingFace. for pre-chunked data, openai_embed for raw data inserts in LightRag

**Why:** Pre-chunked data already takes more time because number of chunks much larger, using openai_embed would make injestion further slower and expensive.

---

### 4. **Vector Storage**
**Choice** Fiass vector storage

**Why** Direct compatibility with LightRag, lighter and faster. Since metadata is not required heavier storages like ChromaDB would be overkill

## Query Pipeline

Query pipeline is simple and straightforward. The prompt is slightly modified to exclude unnessary information from the model response that would make evaluations more convinient.

---

### 6. **Iterative Experiments with LightRAG**

During development I ran **two complete ingestion cycles** with LightRAG:

| Iteration | Ingestion approach | What I learned |
|-----------|-------------------|----------------|
| **Pre‑chunked** | Each document was split first using the custom, type‑aware splitters described above, then the resulting 100–350‑token chunks were passed to `LightRAG.ainsert()`. | • Ingestion was much slower because the number of documents were 100x more.
| **Raw (no upfront chunking)** | Entire documents were sent straight to LightRAG and segmented by its internal token splitter. OpenAI embedding was used for this iteration as speed was not a concern.

**Outcome:** pre‑chunked model provides much better results in grader evals, both models provide similar results on GPT prompt based evals.

### 7. **Evaluation Harness**

**How the bespoke test-set was built**

| Stage | What we did | Why it matters |
|-------|-------------|----------------|
| **1. Whole-file seeding with ChatGPT-4o** | Each complete source document (10-K risk section, MDA, or earnings-call transcript) was handed to ChatGPT-4o and asked to propose a handful of information-seeking Q-A pairs that could be answered solely from that file. Every GPT-generated Q-A pair was reviewed and, when needed, re-worded or corrected | Working at document scope keeps the questions faithful to real user scenarios (they often reference numbers or phrases scattered across many paragraphs). |
| **2. Hand-curated statistical add-ons** | We added ~2-3 numeric questions by hand (e.g., YoY margin deltas, subscriber counts) to make sure tables and figures are exercised. | Generative models tend to produce mostly qualitative questions; the manual layer stresses table retrieval and arithmetic reasoning. |
| **3. Hand-curated false positives** | Created questions that asked information about companies whose data was not present | Evaluates the model doesn't feel obligated to respond to any question |
| **4. Data Distrubution** | Question bank was scaterred across documents to expand coverage, some questions were intentionally curated in a way that the answers require information from multiple documents | Ensures coverage, abstract questions are important that evaluate the model for abstract multi document patterns which was a key requirement.

**Why Ragas-generated test-sets were sidelined**

* Initially explored Ragas for test set generation however it was discarded.
* The test generator produced overly broad encyclopedia-style questions that asked generic questions often without including company name in context.
* Our blended ChatGPT + manual approach yielded company-specific, retrieval-friendly questions that better reflect analyst workflows.

**Scoring rubric (macro-averaged)**

| Dimension     | Rule for full credit |
|---------------|----------------------|
| **Correctness** | Does the answer reasonably address the main point?  |
| **Relevance**    | Are details drawn mainly from the provided context and on topic? |
| **Abstractness**   | Does the answer synthesize across sources without long quotes? |

The final set contains 20 Q-A pairs (15 ChatGPT-seeded + 3 statistical and 2 case to check for false positives), providing a balanced, human-vetted benchmark for the LightRAG pipeline.

---

### 🔧 Automated Grading Engines

During pilot testing we tried **two separate LLM-based graders**:

| Grader | Setup | Observed behaviour |
|--------|-------|--------------------|
| **GPT-4 “Grader” system prompt** | Default with custom criteria | Extremely strict: a single rounding variance on a numeric answer or a missing adjective in the rationale would mark the whole response as **Fail** for *Correctness* **and** *Relevance*. | 
| **GPT-4o-mini with custom prompt** | Prompt explicitly weights criteria (*Correctness 50 %  ➜ Relevance 30 % ➜ Abstractness 20 %*) and allows non-binary grading | Much closer to human judgment: penalises big factual mistakes yet tolerates minor phrasing differences. |


### 📊 Evaluation Results & Interpretation

| Query mode&nbsp;↘︎ / Ingestion&nbsp;→ | **Pre-chunked** | **Raw** |
|--------------------------------------|-----------------|---------|
| **Naïve dense search** | Grader 70 % (14/20)<br>GPT 75 % (15/20) | Grader 50 % (10/20)<br>GPT 80 % (16/20) |
| **Hybrid (dense + BM25)** | Grader 50 % (10/20)<br>GPT 90 % (18/20) | Grader 35 % (7/20)<br>GPT 80 % (16/20) |

<small>*Grader = strict GPT-4 rubric, GPT = custom-weighted GPT-4o-mini grader, total questions = 20.*</small>


#### Breakdown by question type

Model was largely accurate for statistcial and false positive questions across all configurations.
