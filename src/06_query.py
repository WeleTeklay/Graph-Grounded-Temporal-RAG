#src/06_query.py
import chromadb
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
import ollama
import os
from dotenv import load_dotenv
import pandas as pd
from functools import lru_cache
os.environ["CHROMA_TELEMETRY_IMPL"] = "none"

load_dotenv()

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
MODEL_NAME = "llama3.2:3b"

class QueryEngine:
    def __init__(self):
        # Vector DB
        self.chroma_client = chromadb.PersistentClient(path="data/vectors")
        self.collection = self.chroma_client.get_collection("legal_clauses")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        # Graph DB - DISABLED for now
        self.neo4j_driver = None
        
    def get_newest_version(self, chunk_id):
        # Bypass Neo4j: fetch chunk text directly from ChromaDB
        results = self.collection.get(ids=[chunk_id])
        if results and results['documents']:
            text = results['documents'][0]
            meta = results['metadatas'][0] if results['metadatas'] else {}
            effective_date = meta.get('effective_date', 'unknown')
            return chunk_id, text, effective_date
        return chunk_id, None, None
    
    def answer(self, question, target_date="2026-04-08"):
        # 1. Vector Search
        query_embedding = self.embedder.encode(question).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=10)
        
        # 2. Filter to active versions (Graph Grounding)
        seen_topics = set()
        context_pieces = []
        historical_pieces = []
        
        for chunk_id in results['ids'][0]:
            newest_id, newest_text, newest_date = self.get_newest_version(chunk_id)
            
            # Simple deduplication by first few words (topic)
            topic_key = newest_text[:50] if newest_text else ""
            if topic_key in seen_topics:
                continue
            seen_topics.add(topic_key)
            
            # Compare dates
            if newest_date and newest_date <= target_date:
                context_pieces.append(f"[CURRENT] Clause {newest_id} (Effective {newest_date}):\n{newest_text[:500]}\n")
            else:
                historical_pieces.append(f"[SUPERSEDED] Clause {newest_id} (Effective {newest_date}):\n{newest_text[:500]}\n")
        
        prompt = f"""[INST] You are a helpful assistant answering questions based strictly on the provided legal document context.

        CONTEXT:
        {''.join(context_pieces[:3])}

        Question: {question}
        Answer: [/INST]"""
            
        # 4. Get LLM response
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        return response['response']
    
    class QueryEngine:
        @lru_cache(maxsize=1000)
        def get_embedding(self, text: str):
            return self.embedder.encode(text).tolist()

if __name__ == "__main__":
    engine = QueryEngine()
    
    print("Legal Q&A (type 'exit' to quit)")
    while True:
        q = input("\nQuestion: ")
        if q.lower() == 'exit':
            break
        ans = engine.answer(q)
        print(f"\nAnswer: {ans}")