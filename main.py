
import sys
import streamlit as st

from chatbot.query_handler import parse_query
from chatbot.retriever import retrieve_legal_documents
from chatbot.llm_handler import generate_response_with_gemini
from chatbot.logger import setup_logger
from mlflow_logger import log_to_mlflow

logger = setup_logger()

st.set_page_config(page_title="LegalBot", page_icon="⚖️")

st.title("⚖️ LawYerBot: Your Legal Assistant")
st.markdown(
    """
    Ask any legal question (e.g. “Recent Supreme Court rulings on environmental law”).  
    Type **exit** to close the app.
    """
)

user_input = st.text_input("Your query:")

if user_input:
    if user_input.strip().lower() == "exit":
        st.write("Exiting application.")
        sys.exit()

    query = parse_query(user_input)
    logger.info(f"Parsed Query: {query}")

    # 2. Fetch from all configured sources (CourtListener, CAP, GovInfo, etc.)
    docs = retrieve_legal_documents(query, year=query.get("year"))
    logger.info(f"Fetched {len(docs)} document(s) from all sources")


    response = generate_response_with_gemini(query, docs)
    logger.info(f"Generated Response: {response}")

    st.subheader("LegalBot’s Answer:")
    st.write(response)


    log_to_mlflow(query, docs, response)
