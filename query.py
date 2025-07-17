from dotenv import load_dotenv
import asyncio
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.llm.hf import hf_embed
from transformers import AutoTokenizer, AutoModel
from lightrag.utils import EmbeddingFunc


load_dotenv()

WORKING_DIR = os.getenv("WORKING_DIR", "./rag_storage")

embedding_func= EmbeddingFunc(
            embedding_dim=768,
            max_token_size=5000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "BAAI/bge-base-en"
                ),
                embed_model=AutoModel.from_pretrained(
                    "BAAI/bge-base-en"
                ),
            ),
        )
async def initialize_rag():
    
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        
        llm_model_func=gpt_4o_mini_complete,
        vector_storage="FaissVectorDBStorage",
        embedding_batch_num=128,
    )
    await rag.initialize_storages()

    return rag


prompt = """Follow these additional instructions. In case of conflict with previous instructions, the following instructions take precedence:
(1) EXCLUDE references
(2) Don't use MD formatting, provide plain string output.
"""

async def main():
    rag = await initialize_rag()
    print("🤖 Starting RAG query mode...")
    while True:
        query = input("\n💬 Ask a question (or type 'exit'): ")
        if query.lower() in ("exit", "quit"):
            print("👋 Exiting.")
            await rag.finalize_storages()
            break
        print(await rag.aquery(
                query, param=QueryParam(mode="hybrid", user_prompt=prompt ))
            )

if __name__ == "__main__":
    
    asyncio.run(main())
    print("\nDone!")
