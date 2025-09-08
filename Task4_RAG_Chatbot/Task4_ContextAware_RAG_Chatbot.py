# Task4_ContextAware_RAG_Chatbot.py

import os
import glob
from typing import List

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI  # optional


# -------------------------
# Helper functions
# -------------------------

def load_corpus_from_txts(folder_path: str) -> List[Document]:
    docs = []
    for path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        metadata = {"source": os.path.basename(path)}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

def chunk_documents(documents: List[Document], chunk_size: int = 500, overlap: int = 50):
    new_docs = []
    for d in documents:
        text = d.page_content
        start = 0
        while start < len(text):
            chunk = text[start:start+chunk_size]
            new_meta = dict(d.metadata)
            new_meta["chunk_start"] = start
            new_docs.append(Document(page_content=chunk, metadata=new_meta))
            start += chunk_size - overlap
    return new_docs

MODEL_NAME = "all-MiniLM-L6-v2"

def build_vectorstore(documents: List[Document]):
    hf_embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": "cpu"})
    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]
    vectorstore = FAISS.from_texts(texts, hf_embeddings, metadatas=metadatas)
    return vectorstore

def build_conversational_chain(vectorstore, llm_choice: str = "mock", openai_api_key: str = None):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if llm_choice == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API key required")
        os.environ["OPENAI_API_KEY"] = openai_api_key
        llm = OpenAI(temperature=0)
    else:
        from langchain.llms.fake import FakeListLLM
        fake = FakeListLLM(responses=["[Mock reply] Replace with real LLM for actual answers."])
        llm = fake
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return qa

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="Context-Aware RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Context-Aware RAG Chatbot")
st.markdown("This chatbot can answer your questions from your **custom documents** and remembers context.")

# -------------------------
# Session state
# -------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# -------------------------
# Upload folder and build vectorstore
# -------------------------

folder_path = st.text_input("Enter folder path containing your .txt files:", "data")

if st.button("Build Vectorstore"):
    with st.spinner("Loading documents and building vectorstore..."):
        docs = load_corpus_from_txts(folder_path)
        chunks = chunk_documents(docs)
        st.session_state.vectorstore = build_vectorstore(chunks)
        st.session_state.qa_chain = build_conversational_chain(st.session_state.vectorstore, llm_choice="mock")
    st.success("Vectorstore built! You can now ask questions.")

# -------------------------
# Chat interface
# -------------------------

user_input = st.text_input("Ask me something:")

if user_input and st.session_state.qa_chain is not None:
    response = st.session_state.qa_chain.run(input=user_input)
    st.markdown(f"**Bot:** {response}")

st.markdown("---")
st.markdown("**Note:** Currently using a mock LLM. To use OpenAI GPT, set `llm_choice='openai'` and provide your API key.")
