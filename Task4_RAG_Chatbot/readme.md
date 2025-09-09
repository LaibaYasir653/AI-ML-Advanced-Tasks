# 🤖 Context-Aware Chatbot with LangChain & RAG

## 📌 Objective
The goal of this project is to build a **context-aware conversational chatbot** using **LangChain** and **Retrieval-Augmented Generation (RAG)**.  
The chatbot can:
- Remember previous interactions (context memory)  
- Retrieve knowledge from a **custom document corpus**  
- Answer queries with relevant context  
- Run on an **interactive Streamlit interface**

---

## 🧰 Tech Stack
- **Python 3.13**
- [LangChain](https://www.langchain.com/) — conversational AI framework
- **Sentence-Transformers** — document embeddings
- **FAISS** — vector similarity search
- **Streamlit** — frontend deployment
- **Pandas, NumPy** — data preprocessing
- **Matplotlib / Scikit-learn** (optional for evaluation/visualizations)

---

## 📂 Project Structure
📦 context-aware-chatbot
├── data/ # Folder for .txt documents
├── app.py # Streamlit frontend
├── Task4_ContextAware_RAG_Chatbot.py # Backend pipeline
├── requirements.txt # Dependencies
├── README.md # Documentation
└── faiss_store/ # Persisted vector database (generated)

#  📑 Methodology / Approach
## 🔹 1. Dataset Loading

Documents are loaded from /data/ folder (.txt files).

Each file is converted into LangChain Documents with metadata.

## 🔹 2. Preprocessing & Chunking

Documents are split into chunks of 500 tokens with overlap for better retrieval.

## 🔹 3. Embedding & Vector Store

Each chunk is embedded using all-MiniLM-L6-v2.

Stored in a FAISS vectorstore for similarity search.

## 🔹 4. Conversational Chain

Uses ConversationalRetrievalChain with:

Retriever from FAISS

ConversationBufferMemory (context persistence)

LLM (OpenAI or mock)

## 🔹 5. Streamlit Deployment

Frontend built with Streamlit.

Features:

File upload

Vectorization progress

Live chat window

Memory + retrieval settings

Example code snippets

## Run Streamlit app
streamlit run app.py

# ✅ Key Results / Observations

Chatbot successfully integrates RAG pipeline with memory.

Retrieval is fast and relevant due to FAISS vector indexing.

Conversation history improves response continuity.

Deployment with Streamlit provides a user-friendly interface.

# 🚀 Skills Gained

Conversational AI development with LangChain

Document embedding & vector search using FAISS

Retrieval-Augmented Generation (RAG)

Building interactive AI apps with Streamlit

# 📌 Future Improvements

Add evaluation metrics dashboard

Support for multiple document formats (PDF, DOCX, HTML)

Integration with OpenAI GPT-4 or local LLMs

Long-term memory persistence (e.g., Pinecone, ChromaDB)

# 🏁 Final Summary

This project demonstrates how to build a context-aware chatbot that combines:

LangChain pipelines

Sentence-Transformer embeddings

FAISS vectorstore

Conversation memory

Streamlit deployment

👉 A practical implementation of Retrieval-Augmented Generation (RAG) for real-world conversational AI.

```bash
