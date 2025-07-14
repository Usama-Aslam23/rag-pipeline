# REASONING.md

## 🔍 Assignment Overview

The task was to build a chunking pipeline for a Retrieval-Augmented Generation (RAG) system focused on identifying and representing abstract multi-document patterns from text data.

---

## 🧠 Key Assumptions

- Real-world documents like 10-K filings and earnings calls are structurally different; chunking should be adapted accordingly.
- Abstract business patterns (e.g., "AI investments", "supply chain risk") exist across documents and capturing them improves downstream retrieval.
- LLMs can help label such patterns but are expensive — so usage must be scoped.
- Ground-truth reference answers are needed for meaningful evaluation, not just keyword overlap.
- The dataset (10-K filings including MDA and Risk sections, along with earnings call transcripts) was self-curated to reflect realistic financial disclosures where abstract multi-document patterns could naturally emerge. This choice was made to align with the assignment's goal while introducing structural and semantic variety across documents.


---

## 🧩 Critical Design Reasoning

### 0. **Dataset Selection**

**Decision:** Use a mix of 10-K filings (MDA and Risk sections) and earnings call transcripts from various tech companies.

**Why:**
  
- They represent high-value, information-dense, and pattern-rich content.
- Abstract themes such as regulatory risk, AI investment, and macroeconomic outlook recur across different formats.
- These sources are widely used in real-world market and risk analysis, making them ideal for testing a pattern-aware RAG pipeline.
- Their inherent structural diversity (narrative text, bullets, dialogue, and semi-tabular content) provided a robust basis to stress-test chunking and retrieval strategies.

### 1. **File Ingestion & Metadata**

**Decision:** Filenames are used to extract metadata (company name, document type).

**Why:** The documents didn’t have standardized headers; parsing file names was the most reliable source of document-level metadata.

---

### 2. **Chunking Strategy by Document Type**

#### a. **Risk Section Chunking**

**Why bullet-based?** Risks tend to be disclosed as bullet points or numbered lists. A naive sentence splitter fragments them unnaturally.

**Fallback:** If structure isn't clear, we fall back to recursive splitting to maintain readability.

#### b. **Earnings Call Chunking**

**Why speaker turns?** These documents are dialogues. Chunking by speaker helps isolate who said what — crucial for business insight and attribution.

**Challenges:** Speaker formatting was inconsistent (colons, dashes, etc.), so regex had to handle a range of patterns robustly.

#### c. **MDA (Management Discussion & Analysis)**

**Why not fixed rules?** MDA text is semi-structured — paragraphs, embedded tables, etc. Using rules like "split every 4 sentences" loses nuance.

**LLM use:** GPT-3.5 was used to tag abstract patterns, but only in MDA to limit cost. Other sources didn’t benefit meaningfully.

#### d. **Theme Detection with HDBSCAN**

**Why not KMeans?** KMeans requires specifying the number of clusters in advance and assumes all clusters are similarly sized and shaped. This is unrealistic for abstract themes, which vary in length, density, and complexity.

**Why HDBSCAN?** HDBSCAN discovers natural groupings in the data without needing to predefine how many themes exist. It also marks "noise" or low-content chunks as unclustered, which improves theme coherence and avoids polluting semantic groupings with boilerplate text. This was especially valuable in documents like the MD&A and risk sections where not all paragraphs contain meaningful content.

---

### 3. **Pattern Representation**

**Why add patterns to metadata?** Including LLM-tagged abstract patterns (e.g., "regulatory scrutiny", "capital allocation") helps group and retrieve semantically rich content.

**Tradeoff:** Risks and earnings call documents were intentionally left untagged with abstract patterns. Risk disclosures already exhibit thematic separation through structured bullet formatting. Earnings calls, being conversational, are chunked by speaker turns — preserving natural discourse units. Abstract tagging in these contexts would add minimal value while increasing annotation cost and potential noise.


---

### 4. **Embedding Model Selection**

**Choice:** `BAAI/bge-small-en` via HuggingFace.

**Why:** It balances performance with speed, and has been shown to work well for passage-level embedding tasks in industry RAG use cases.

---

### 5. Query Pipeline

**Query Rewriting:** The pipeline includes a query rewriting module that uses an LLM to reformulate the original user query into a more structured, keyword-rich prompt. This boosts retrieval effectiveness, particularly in financial documents where user queries may be vague or underspecified.

**Metadata Filtering:** Additionally, queries can be filtered using metadata fields such as `company_name` and `doc_type`. This ensures that irrelevant chunks (e.g., earnings call content for a risk-related query) are excluded from consideration.

**Why this matters:** Abstract multi-document patterns often require retrieving semantically aligned passages from heterogeneous document types. Without rewriting or metadata filters, the retriever might rank unaligned chunks highly, especially when some documents contain richer embeddings or longer texts. The query pipeline addresses this gap through both pre-retrieval refinement and post-retrieval narrowing.

---

### 6. **Evaluation Harness**

**Key insight:** A good RAG system should retrieve the **right chunks** and synthesize the **right answer** — both must be evaluated.

- Reference answers are generated by prompting GPT-3.5 to answer **only from retrieved chunks.** This approach simulates the ideal generation condition in a RAG system — where retrieval is perfect — allowing us to isolate and test the model's ability to synthesize information. It also ensures the reference answers are grounded, as they do not include external knowledge or hallucinations, and match the retrieval constraints imposed on the actual system.
- Grading checks both the generated answer and whether the expected documents were retrieved, with a focus on correctness and coverage of source content.

**Binary vs Graded:** Initially used binary scores, but expanded to include correctness, coverage, and abstraction using GPT feedback.

---

## 🧪 Evaluation Strategy Reasoning

Each query in the eval set was manually crafted to reflect realistic user intents, then paired with GPT-generated reference answers synthesized from relevant chunks. While `expected_documents` are included in the dataset for future use, current evaluation logic does not yet validate retrieved document IDs.

We adopted a four-part grading rubric to move beyond binary success/failure, aiming to evaluate:

- **Correctness**: Is the core information accurate and aligned with the query?
- **Relevance**: Does the answer stick to what the question asked without digressing?
- **Coverage**: Are key points from the source documents adequately addressed?
- **Abstraction**: Does the answer demonstrate summarization and synthesis rather than shallow copying?

These criteria ensure the RAG pipeline is assessed on both factual accuracy and its ability to reason over retrieved content — essential for abstract multi-document tasks.

