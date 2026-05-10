#check_documents.py
# save as check_documents.py
import json
from pathlib import Path

# Load chunks
with open('data/chunks/clauses.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print("=" * 60)
print("DOCUMENT CONTENT ANALYSIS")
print("=" * 60)

# Search for key terms
keywords = {
    "parties": ["party", "parties", "employer", "employee", "borrower", "lender"],
    "governing law": ["governing", "law", "jurisdiction", "govern", "applicable law"],
    "section 1.1": ["section 1.1", "1.1", "§1.1"],
    "penalty": ["penalty", "fee", "charge", "late fee", "penalty clause"]
}

for topic, terms in keywords.items():
    print(f"\n Searching for '{topic}':")
    found = False
    for chunk in chunks[:100]:  # Check first 100 chunks
        text = chunk.get('text', '').lower()
        for term in terms:
            if term in text:
                preview = text[:200].replace('\n', ' ')
                print(f"Found in {chunk.get('doc_id')}: {preview[:100]}...")
                found = True
                break
        if found:
            break
    if not found:
        print(f"NOT FOUND in documents")