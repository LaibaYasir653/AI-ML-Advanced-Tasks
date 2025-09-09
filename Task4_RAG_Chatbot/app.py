# app.py
import streamlit as st
from datetime import datetime
st.set_page_config(page_title="Context-Aware Chatbot — LangChain RAG", layout="wide")

# --- Styles (clean, realistic) ---
st.markdown("""
<style>
body { background: #f6f7fb; }
.header { display:flex; align-items:center; justify-content:space-between; }
.app-title { font-size:22px; font-weight:600; color:#0f172a; }
.status-pill { font-size:12px; background:#eef2ff; color:#0b6cff; padding:6px 10px; border-radius:12px; margin-left:8px; }
.sidebar .stButton>button { border-radius:8px; }
.card { background:white; border-radius:12px; padding:18px; box-shadow: 0 6px 18px rgba(12,24,60,0.06); }
.progress-bar { height:10px; background:#e6eefc; border-radius:6px; overflow:hidden; }
.progress-fill { height:10px; background:#0b6cff; width:85%; }
.chat-user { background:#eef2ff; padding:10px 14px; border-radius:14px; display:inline-block; }
.chat-assistant { background:#eefaf6; padding:10px 14px; border-radius:14px; display:inline-block; }
.small-pill { font-size:11px; background:#f3f4f6; padding:4px 8px; border-radius:10px; margin-right:6px; }
.footer { color:#6b7280; font-size:12px; margin-top:16px; }
.code-box { background:#0b1220; color:#e6eefc; padding:12px; border-radius:8px; font-family:monospace; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# --- Header / Navbar ---
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<div class="header"><div class="app-title">Context-Aware Chatbot — LangChain RAG</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div style="display:flex; justify-content:flex-end; gap:8px;"><div class="status-pill">Vector DB: Pinecone (online)</div><div class="status-pill">Embeddings: up-to-date</div></div>', unsafe_allow_html=True)

st.write("")  # spacer

# --- Layout: Sidebar + Main + Right ---
left_col, main_col, right_col = st.columns([1, 2.2, 1])

# Left sidebar card
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Corpus & Index")
    st.write("Upload Corpus (PDF, DOCX, HTML)")
    st.file_uploader("Upload documents", accept_multiple_files=True, key="uploader")
    st.write(" ")
    st.markdown("**Vectorize & Index**")
    st.markdown('<div class="progress-bar"><div class="progress-fill"></div></div>', unsafe_allow_html=True)
    st.write("12 documents processed — 85%")
    st.write(" ")
    st.markdown("**Memory Settings**")
    st.write("- Short-term: ConversationBufferMemory (k=5)")
    st.write("- Long-term: Persisted (enabled)")
    st.write(" ")
    st.markdown("**Retrieval Settings**")
    st.write("- k = 5  •  similarity threshold: 0.75")
    st.markdown("</div>", unsafe_allow_html=True)

# Main column content
with main_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### System Flow")
    st.write("""
    Docs  ➜  Embeddings  ➜  Vector Store  ➜  Retriever  ➜  LLM  ➜  Output
    """)
    st.write("_embedding model: text-embedding-3-small · vector DB: FAISS / Pinecone_")
    st.write("---")

    left_mid, right_mid = st.columns([1,1])

    # Retrieval panel
    with left_mid:
        st.markdown("**Retrieval — sample query**")
        st.text_input("Query", value="How does the chatbot remember context across sessions?", key="q1")
        st.write("")
        st.markdown("**Top retrieved documents**")
        st.write("- Document_07.pdf — excerpt: \"Use ConversationBufferMemory with persistence...\"  (score: 0.87)")
        st.write("- Onboarding_Guide.docx — excerpt: \"store embeddings in vector DB; use retriever.k=5\"  (score: 0.82)")
        st.write("- FAQ.md — excerpt: \"session-to-session memory via long-term store\" (score: 0.76)")

    # Conversation panel
    with right_mid:
        st.markdown("**Live Conversation**")
        st.markdown('<div style="padding:8px; border-radius:10px;">', unsafe_allow_html=True)
        st.markdown('<div style="margin-bottom:8px;"><span class="chat-user">User: Summarize the vendor onboarding doc.</span></div>', unsafe_allow_html=True)
        st.markdown('<div><span class="chat-assistant">Assistant: Sure — Here is a concise summary: Vendors must submit KYC, sample images, and complete onboarding forms. Documents are verified and stored in the vector index for quick retrieval. <span class="small-pill">Source: Document_07.pdf</span></span></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.write("")
        st.text_area("Type your message...", value="", height=80, key="chat_input")
        st.button("Send", key="send_btn")
    st.markdown("</div>", unsafe_allow_html=True)

# Right column: controls and code snippet
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Controls & Example Code")
    st.write("Model: GPT-4o  •  Temperature: 0.2")
    st.write("")
    st.markdown("**Example: LangChain Retriever + Memory**")
    st.markdown('''<pre class="code-box">from langchain import OpenAI, ConversationChain
from langchain.vectorstores import FAISS
# embeddings -> index -> retriever
memory = ConversationBufferMemory(k=5)
retriever = faiss_index.as_retriever(search_k=5)
chain = ConversationalRetrievalChain(llm=OpenAI(), retriever=retriever, memory=memory)
</pre>''', unsafe_allow_html=True)
    st.write("")
    st.write("Session Memory")
    st.checkbox("Enable Session Memory", value=True, key="mem1")
    st.checkbox("Persist Long-term Memory", value=True, key="mem2")
    st.checkbox("Auto-refresh Index", value=False, key="mem3")
    st.write("")
    st.button("Run Query", key="run_query", help="Execute retrieval + generation")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(f"<div class='footer'>Deployed with Streamlit • Demo only • Updated: {datetime(2025,9,9).strftime('%b %d, %Y')}</div>", unsafe_allow_html=True)
