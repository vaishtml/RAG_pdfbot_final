import streamlit as st
import os
import rag_logic

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 RAG PDF Chatbot")


# ---------------- SESSION STATE ----------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# ---------------- LOAD GEMINI KEY ----------------
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]


# ---------------- SIDEBAR: PDF UPLOAD + RESET ----------------
st.sidebar.header("📤 Upload PDF")
pdf = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# Reset chat button
if st.sidebar.button("🔁 Reset Chat"):
    st.session_state.messages = []
    st.rerun()


# Process PDF only once
if pdf and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF..."):
        docs = rag_logic.load_pdf(pdf.read())
        st.session_state.vectordb = rag_logic.build_vectorstore(docs)
        st.session_state.llm = rag_logic.build_llm()
        st.session_state.messages = []
        st.session_state.pdf_processed = True

    st.sidebar.success("PDF processed successfully!")


# ---------------- CHAT HISTORY DISPLAY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])


# ---------------- CHAT INPUT ----------------
if st.session_state.vectordb is None:
    st.chat_input("Upload a PDF to start chatting...", disabled=True)

else:
    user_input = st.chat_input("Ask a question about the PDF...")

    if user_input:
        # Add USER message
        st.session_state.messages.append({"role": "user", "text": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        
        
        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_logic.answer_query(
                st.session_state.vectordb,
                st.session_state.llm,
                user_input
        )

# Store assistant message FIRST
        st.session_state.messages.append({"role": "assistant", "text": answer})

# Clean UI + prevent duplicates
        st.rerun()


