#build_index.py
"""
Manual Vector Index Builder - Bypasses HuggingFace download issues
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

# Force offline mode to use cached model
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

print("=" * 60)
print("Manual Vector Index Builder")
print("=" * 60)

# 1. Load model (should use cached version)
print("\n[1/4] Loading embedding model...")
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded (dimension: {model.get_embedding_dimension()})")
except Exception as e:
    print(f"Failed: {e}")
    print("   Trying fallback model...")
    model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# 2. Load chunks
print("\n[2/4] Loading chunks...")
chunks_path = Path("data/chunks/clauses.json")
with open(chunks_path, 'r', encoding='utf-8') as f:
    chunks = json.load(f)
print(f"Loaded {len(chunks)} chunks")

# 3. Create ChromaDB collection
print("\n[3/4] Creating vector index...")
client = chromadb.PersistentClient(path="data/vectors")

# Delete existing if any
try:
    client.delete_collection("legal_clauses")
    print("Deleted existing collection")
except:
    pass

collection = client.create_collection(
    name="legal_clauses",
    metadata={"description": "Legal document clauses"}
)

# 4. Add chunks in batches
batch_size = 100
total = len(chunks)

print(f"\n[4/4] Indexing {total} chunks in batches of {batch_size}...")

for i in range(0, total, batch_size):
    batch = chunks[i:i+batch_size]
    
    ids = [c['chunk_id'] for c in batch]
    texts = [c['text'] for c in batch]
    
    # Generate embeddings
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    
    metadatas = [{
        'doc_id': c['doc_id'],
        'effective_date': c['effective_date'],
        'chunk_id': c['chunk_id'],
        'char_count': len(c['text']),
        'word_count': len(c['text'].split())
    } for c in batch]
    
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    percent = (i + len(batch)) / total * 100
    print(f"   Progress: {i + len(batch)}/{total} ({percent:.1f}%)")

print(f"\n Index complete! {collection.count()} vectors in ChromaDB")