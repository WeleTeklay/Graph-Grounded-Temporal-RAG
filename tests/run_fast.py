#tests/run_fast.py
"""
Fast query interface using OptimizedQueryEngine
"""
from src.optimized_retriever import OptimizedQueryEngine
import time

engine = OptimizedQueryEngine()

print("\n" + "=" * 60)
print("LexTemporal AI - Fast Query Engine")
print("=" * 60)
print(f"Graph enabled: {engine.graph_enabled}")
print(f"Vector count: {engine.collection.count()}")
print("=" * 60 + "\n")

while True:
    q = input("Question (or 'exit'): ").strip()
    if q.lower() == 'exit':
        break
    if not q:
        continue
    
    start = time.time()
    result = engine.answer(q)
    elapsed = time.time() - start
    
    print(f"\nAnswer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} | Time: {elapsed:.2f}s")
    
    if result.get('graph_used'):
        print(f"Graph: Used")
    print("-" * 60 + "\n")