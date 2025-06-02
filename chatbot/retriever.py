import os
import requests
from bs4 import BeautifulSoup
import json
import time
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import faiss
import numpy as np

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

logger = logging.getLogger(__name__)

def fetch_cases_from_courtlistener(query, year=None, max_results=5):
    base_url = "https://www.courtlistener.com/api/rest/v3/opinions/"
    params = {
        "search": query["topic"],
        "court": "scotus" if query["court"] == "supreme" else "",
        "date_filed__gte": f"{year}-01-01" if year else "",
        "page_size": max_results
    }
    headers = {"User-Agent": "LegalBot/1.0"}
    resp = requests.get(base_url, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json().get("results", [])

def fetch_cases_from_cap(query, year=None, max_results=5):
    cap_api_key = os.getenv("CAP_API_KEY")
    base_url = "https://api.case.law/v1/cases/"
    params = {
        "search": query["topic"],
        "jurisdiction": "us",
        "page_size": max_results
    }
    if year:
        params["decision_date_min"] = f"{year}-01-01"
    headers = {"Authorization": f"Token {cap_api_key}"}
    resp = requests.get(base_url, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json().get("results", [])

def fetch_cases_from_govinfo(query, year=None, max_results=5):
    gov_key = os.getenv("GOVINFO_API_KEY")
    base_url = "https://api.govinfo.gov/collections/USCOURTS/"
    params = {
        "api_key": gov_key,
        "offset": 0,
        "pageSize": max_results
    }
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    results = resp.json().get("packages", [])
    filtered = []
    for pkg in results:
        date = pkg.get("dateIssued", "")
        if year and not date.startswith(str(year)):
            continue
        if query["topic"].lower() in pkg.get("title", "").lower():
            filtered.append(pkg)
    return filtered

def fetch_scarcity_scotus(query, year=None):
    # Example scraper for SCOTUS opinions (HTML index)
    scotus_index = "https://www.supremecourt.gov/opinions/slipopinion/22"
    resp = requests.get(scotus_index)
    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.select("table.tablegrid a")
    cases = []
    for a in links:
        href = a.get("href")
        title = a.text.strip()
        if query["topic"].lower() in title.lower():
            cases.append({"case_name": title, "pdf_url": f"https://www.supremecourt.gov{href}"})
    return cases[:5]

def retrieve_legal_documents(query, year=None, max_results=5):
    all_docs = []
    try:
        cl = fetch_cases_from_courtlistener(query, year, max_results)
        for doc in cl:
            text = doc.get("plain_text") or ""
            all_docs.append({"source": "CourtListener", "id": doc.get("id"), "case_name": doc.get("case_name", ""), "text": text})
    except Exception as e:
        logger.error(f"Error fetching from CourtListener: {e}")

    try:
        cap = fetch_cases_from_cap(query, year, max_results)
        for doc in cap:
            text = doc.get("decision_text") or ""
            all_docs.append({"source": "CAP", "id": doc.get("id"), "case_name": doc.get("name", ""), "text": text})
    except Exception as e:
        logger.error(f"Error fetching from CAP: {e}")

    try:
        gi = fetch_cases_from_govinfo(query, year, max_results)
        for pkg in gi:
            summary = pkg.get("summary", "")
            all_docs.append({"source": "GovInfo", "id": pkg.get("package_id"), "case_name": pkg.get("title", ""), "text": summary})
    except Exception as e:
        logger.error(f"Error fetching from GovInfo: {e}")

    return all_docs



def embed_texts(texts):
    model = genai.EmbeddingModel("gemini-embedding-exp")
    resp = model.get_embeddings(texts)
    embeddings = [e.embedding for e in resp]  
    return embeddings


def build_faiss_index(docs, index_path="data/faiss_index.index"):
    texts = [d["text"] for d in docs]
    embeddings = embed_texts(texts)
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    idx_to_meta = []
    for i, vec in enumerate(embeddings):
        index.add(np.array([vec], dtype="float32"))
        idx_to_meta.append(docs[i])
    faiss.write_index(index, index_path)
    with open(index_path + ".meta", "w") as f:
        json.dump(idx_to_meta, f)
    return index, idx_to_meta


def search_faiss(query_text, index_path="data/faiss_index.index", top_k=5):
    index = faiss.read_index(index_path)
    with open(index_path + ".meta", "r") as f:
        idx_to_meta = json.load(f)
    q_embed = embed_texts([query_text])[0]
    D, I = index.search(np.array([q_embed], dtype="float32"), top_k)
    results = [idx_to_meta[i] for i in I[0]]
    return results