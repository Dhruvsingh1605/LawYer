import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chatbot.query_handler import parse_query
from chatbot.retriever import retrieve_legal_documents
import json

def main():
    queries = [
        {"topic": "environmental law", "court": "supreme", "year": 2024},
        {"topic": "antitrust",        "court": "federal",    "year": 2023},
    ]

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    for q in queries:
        docs = retrieve_legal_documents(q, year=q.get("year"))
        for doc in docs:
            source = doc["source"]
            case_id = doc["id"]
            filename = f"{source}_{case_id}.json"
            filepath = os.path.join(raw_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
