# main.py

import sys
from chatbot.query_handler import parse_query
from chatbot.retriever import retrieve_legal_documents
from chatbot.llm_handler import generate_response_with_gemini
from chatbot.logger import setup_logger
from mlflow_logger import log_to_mlflow

logger = setup_logger()

def chatbot_main():
    last_irrelevant = False  

    print("⚖️ LawYer (CLI mode). Ask your legal question (type 'exit' to quit).")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "exit":
            print("Exiting application.")
            sys.exit()

        query = parse_query(user_input)
        logger.info(f"Parsed Query: {query}")

        docs = retrieve_legal_documents(query, year=query.get("year"))
        logger.info(f"Fetched {len(docs)} document(s) from all sources")

        if not docs:
            response = generate_response_with_gemini(query, docs, second_chance=last_irrelevant)
            last_irrelevant = True
        else:
            response = generate_response_with_gemini(query, docs, second_chance=False)
            last_irrelevant = False  
        logger.info(f"Generated Response: {response}")

        print("\n" + response + "\n")

        log_to_mlflow(query, docs, response)


if __name__ == "__main__":
    chatbot_main()
