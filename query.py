import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_store/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo")
TOP_K = int(os.getenv("TOP_K", "8"))

def query_pipeline(query: str, company: str = None) -> dict:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOpenAI(temperature=0, model=GPT_MODEL)

    section_filter = infer_metadata_filter(query)
    company_filter = {"company": {"$in": [company]}} if company else infer_company_filter(query, vectordb)
    filters = []
    if section_filter.get("section"):
        filters.append({"section": {"$eq": section_filter["section"]}})
    if company_filter.get("company", {}).get("$in"):
        filters.append({"company": {"$in": company_filter["company"]["$in"]}})

    if filters:
        retriever.search_kwargs["filter"] = filters[0] if len(filters) == 1 else {"$and": filters}

    rewriter_llm = ChatOpenAI(temperature=0, model=GPT_MODEL)
    query_rewrite_prompt = ChatPromptTemplate.from_template(
        """You are assisting a question-answering system that retrieves information from 10-K filings, MD&A sections, risk disclosures, and earnings call transcripts.

Your task is to rewrite the following question to improve its clarity and alignment with how these documents are chunked and stored. Make it concise, keyword-rich, and semantically optimized for retrieval — especially for locating high-level business concerns and strategic themes.

Do not change the original intent or introduce new meaning.

Original question:
"{query}"

Rewritten version:"""
    )
    query_rewriter_chain = query_rewrite_prompt | rewriter_llm | StrOutputParser()
    rewritten_query = query_rewriter_chain.invoke({"query": query})
    print(rewritten_query)
    docs = retriever.invoke(rewritten_query)

    pattern_llm = ChatOpenAI(temperature=0, model=GPT_MODEL)
    pattern_prompt = ChatPromptTemplate.from_template("List 3 abstract business concerns implied in: '{query}'")
    pattern_chain = pattern_prompt | pattern_llm | StrOutputParser()
    query_themes = pattern_chain.invoke({"query": rewritten_query}).lower().split("\n")
    query_themes = [t.strip("-• ").strip() for t in query_themes if t.strip()]

    def rerank(doc):
        patterns = doc.metadata.get("patterns", "")
        return sum(1 for theme in query_themes if theme and theme.lower() in patterns.lower())

    docs = sorted(docs, key=rerank, reverse=True)
    trimmed_docs = truncate_docs(docs)

    # Build custom prompt and chain for document QA
    prompt = PromptTemplate.from_template(
        """Use the following context to answer the user's question.
Only use the context provided.

Context:
{context}

Question: {question}
Answer:"""
    )
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    answer = document_chain.invoke({
        "context": trimmed_docs,
        "question": query
    })
    output_text = answer["output_text"] if isinstance(answer, dict) else answer
    return {
        "output_text": output_text,
        "input_documents": docs,
        "filters": retriever.search_kwargs.get("filter", {})
    }

def infer_metadata_filter(query: str):
    q = query.lower()
    if any(word in q for word in [
        "risk", "threat", "concern", "uncertainty", "exposure",
        "regulatory", "legal risk", "compliance"
    ]):
        return {"section": "item_1a"}
    if any(word in q for word in [
        "mda", "management", "outlook", "strategy", "forecast", "md&a",
        "operating results", "performance", "growth"
    ]):
        return {"section": "item_7"}
    if any(word in q for word in [
        "earnings", "call", "conference", "analyst", "q&a", "transcript",
        "prepared remarks"
    ]):
        return {"section": "earnings_call"}
    return {}

def infer_company_filter(query: str, vectordb):
    # Dynamically fetch all distinct company values from the index
    collection = vectordb._collection
    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    companies = list({meta.get("company") for meta in all_metadata if meta.get("company")})
    return {
        "company": {"$in": [c for c in companies if c and c in query.lower()]}
    }

def truncate_docs(docs, max_total_chars=12000):
    total = 0
    selected = []
    for doc in docs:
        if total + len(doc.page_content) > max_total_chars:
            break
        selected.append(doc)
        total += len(doc.page_content)
    return selected

def query_mode():
    print("🤖 Starting RAG query mode...")
    while True:
        query = input("\n💬 Ask a question (or type 'exit'): ")
        if query.lower() in ("exit", "quit"):
            print("👋 Exiting.")
            break
        result = query_pipeline(query)
        print("\n🔍 Applied Metadata Filter:")
        print(result["filters"])
        print("\n📄 Top Retrieved Chunks:")
        for i, doc in enumerate(result["input_documents"]):
            print(f"\n--- Chunk {i+1} ---")
            print("Source:", doc.metadata.get("source"))
            print("Section:", doc.metadata.get("section"))
            print("Preview:", doc.page_content[:500])
        print(f"\n🧠 Answer: {result['output_text']}")