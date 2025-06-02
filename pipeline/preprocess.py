import sys
import os
import json
from bs4 import BeautifulSoup
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def clean_text_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw = data.get("text") or data.get("plain_text") or data.get("decision_text") or ""
    clean = BeautifulSoup(raw, "html.parser").get_text(separator="\n")
    clean = "\n".join(line.strip() for line in clean.splitlines() if line.strip())

    return {
        "source":     data.get("source", ""),
        "case_id":    data.get("id", ""),
        "case_name":  data.get("case_name", data.get("name", "")),
        "date":       data.get("date_filed", data.get("decision_date", "")),
        "clean_text": clean
    }

def main():
    raw_dir   = Path(PROJECT_ROOT) / "data" / "raw"
    clean_dir = Path(PROJECT_ROOT) / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in raw_dir.glob("*.json"):
        cleaned = clean_text_from_json(raw_file)
        out_name = raw_file.stem + ".clean.json"
        out_path = clean_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
