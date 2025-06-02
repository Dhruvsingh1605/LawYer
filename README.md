# LawYer: A Retrieval-Augmented Legal Chatbot


## Table of Contents

1. [Project Overview](#project-overview)  
2. [High-Level Architecture](#high-level-architecture)  
3. [Directory Structure](#directory-structure)  
4. [Installation & Environment Setup](#installation--environment-setup)  
5. [Usage (CLI & Streamlit)](#usage-cli--streamlit)  
6. [Core Components & Deep Dive](#core-components--deep-dive)  
   - [1. Query Processing](#1-query-processing)  
   - [2. Data Retrieval (Retriever)](#2-data-retrieval-retriever)  
   - [3. Data Pipeline (DVC Stages)](#3-data-pipeline-dvc-stages)  
   - [4. Data Cleaning (Preprocess)](#4-data-cleaning-preprocess)  
   - [5. Embedding & FAISS Indexing](#5-embedding--faiss-indexing)  
   - [6. LLM Integration & Prompt Template](#6-llm-integration--prompt-template)  
   - [7. Logging & Auditing](#7-logging--auditing)  
   - [8. MLflow Experiment Tracking](#8-mlflow-experiment-tracking)  
7. [Environment Variables & API Keys](#environment-variables--api-keys)  
8. [Running the DVC Pipeline](#running-the-dvc-pipeline)  
9. [Running LawYer](#running-lawyer)  
   - [CLI Mode](#cli-mode)  
   - [Streamlit Mode](#streamlit-mode)  
10. [Interview-Prep & FAQs](#interview-prep--faqs)  
11. [Future Enhancements](#future-enhancements)  
12. [License & Acknowledgements](#license--acknowledgements)  

---

## Project Overview

**LawYer** is a Python-based chatbot designed to answer legal questions by dynamically retrieving both **live** and **historical** court rulings, statutes, and government-published legal documents. At its core, LawYer integrates:

- **Retrieval Modules** (CourtListener, Caselaw Access Project, GovInfo, custom scrapers)  
- **LLM-Based Generation** (Google’s Gemini 1.5-Flash)  
- **Data Pipelines** (DVC) to version raw and cleaned legal texts  
- **Experiment Tracking** (MLflow) for reproducibility  
- **Embeddings & FAISS** for semantic search over large corpora  
- **Structured Logging** for auditing every query and response  
- **UI Modes**: 
  - **CLI** (command-line interface)  
  - **Streamlit** (web-based)  

A distinguishing feature is its **“first-irrelevant vs. second-irrelevant”** logic:  
- On an initial non-legal query, LawYer prompts the user to ask a legal question.  
- If the user again asks something unrelated, LawYer falls back to Gemini 1.5 to answer generically.

This README explains every aspect—from installation to the deep architecture—so you can walk into an interview and speak confidently about design decisions, code mechanics, and trade-offs.

---

## High-Level Architecture

         ┌──────────────────────────────────────────────────┐
         │                    User                          │
         │  (CLI or Streamlit UI — “Ask your Question”)     │
         └───────────────┬──────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │        Query Processor        │
         │  (parse_query → extract topic,│
         │   court, year from text)      │
         └───────────────┬───────────────┘
                         │
         ┌───────────────▼──────────────────┐
         │        Data Retriever             │
         │  ┌─────────────────────────────┐  │
         │  │ CourtListener (API)         │  │
         │  │ Caselaw Access Project (API)│  │──┐
         │  │ GovInfo (API)               │  │  │
         │  │ Custom Scraper (Juriscraper)│  │  │
         │  └─────────────────────────────┘  │  │
         └───────────────┬──────────────────┘  │
                         │                     │
         ┌───────────────▼──────────────────┐  │
         │   (Optional) Data Storage &       │  │
         │  Local FAISS Index (embeddings)   │  │
         │       via Gemini embeddings       │  │
         └───────────────┬──────────────────┘  │
                         │                     │
         ┌───────────────▼──────────────────┐  │
         │        LLM Handler                 │  │
         │  (build prompt + Gemini 1.5 call)  │  │
         └───────────────┬──────────────────┘  │
                         │                     │
         ┌───────────────▼──────────────────┐  │
         │      Response & Logging            │  │
         │ (print to CLI or Streamlit UI;     │  │
         │  write to chatbot.log; log to MLflow)│ │
         └─────────────────────────────────────┘  │
                                               │
         ┌─────────────────────────────────────┐  │
         │           DVC Pipeline              │◀─┘
         │ (fetch_cases.py → data/raw/;        │
         │  preprocess.py → data/clean/)        │
         └─────────────────────────────────────┘


- **Query Processor**: A lightweight parser that extracts keywords, the jurisdiction (e.g. “Supreme Court”), and time frame (e.g. “2024”) from the user’s text.  
- **Data Retriever**: A multi-source module that calls APIs / scrapes websites to obtain up-to-date or historical case texts.  
- **Embeddings & FAISS (Optional)**: We embed “cleaned” case texts using Gemini 1.5’s embedding model to build a local FAISS index—this accelerates similarity search for semantically related documents.  
- **LLM Handler**: Constructs a prompt with retrieved case excerpts and invokes Gemini 1.5-Flash to generate a concise, context-aware answer.  
- **Logging & MLflow**: Each run is recorded in a logfile (`logs/chatbot.log`) and tracked as an MLflow experiment (parameters, metrics, artifacts).  
- **DVC Pipeline**: Defines reproducible stages for fetching raw data (`fetch_cases.py`) and preprocessing (`preprocess.py`). Each stage’s inputs and outputs are versioned.

---

## Directory Structure

```text
legal_bot/
├── .env                          # API keys and configuration variables
├── data/
│   ├── raw/                      # Raw JSON/PDF blobs from external sources (DVC tracks this)
│   └── clean/                    # Cleaned JSON texts (DVC tracks this)
│   └── faiss_index.index         # (Optional) FAISS index file
│   └── faiss_index.index.meta    # (Optional) Metadata for FAISS index
│
├── logs/
│   └── chatbot.log               # Structured log file (timestamped queries/responses)
│
├── pipeline/
│   ├── fetch_cases.py            # DVC “fetch” stage: pull raw data
│   └── preprocess.py             # DVC “preprocess” stage: clean raw data
│
├── chatbot/
│   ├── __init__.py
│   ├── query_handler.py          # parse_query(…) → {court, topic, year}
│   ├── retriever.py              # retrieve_legal_documents(…) & individual fetchers
│   ├── llm_handler.py            # generate_response_with_gemini(…) + prompt logic
│   └── logger.py                 # setup_logger(…) → Python’s logging configuration
│
├── mlflow_logger.py              # log_to_mlflow(…) → record params/metrics/artifacts
├── dvc.yaml                      # DVC pipeline configuration
├── dvc.lock                      # Auto-generated by DVC; pinning versions
├── requirements.txt              # Python dependencies
├── main.py                       # CLI mode entry point
├── app.py                        # Streamlit mode entry point
└── README.md                     # ← (You’re reading this file)
