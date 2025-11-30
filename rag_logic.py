# rag_logic.py
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# -------- PDF LOADING (NO TEMP FILES) --------
def load_pdf(pdf_bytes):
    return PyPDFLoader.from_bytes(pdf_bytes).load()


# -------- VECTORSTORE --------
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vectordb


# -------- GEMINI LLM --------
def build_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        api_key=st.secrets["GEMINI_API_KEY"]
    )


# -------- ANSWER QUERY --------
def answer_query(vectordb, llm, question):
    retrieved_docs = vectordb.similarity_search(question, k=3)

    if not retrieved_docs:
        return "Not found in PDF."

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    You are a strict PDF-based assistant.
    You MUST use ONLY the text in the context.
    If the answer is not contained in the PDF, reply exactly:
    "Not found in PDF."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    response = llm.invoke(prompt)
    return response.content
