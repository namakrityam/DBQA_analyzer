import streamlit as st
import hashlib
from dotenv import load_dotenv

from utils.qa_chain import BASE_PROMPT, get_qa_chain
from utils.loader import load_document
from utils.splitter import split_text
from utils.vectorstore import create_vectorstore
from utils.llm_provider import get_llm

# =====================================================
# ENV + PAGE CONFIG (ONLY ONCE)
# =====================================================
load_dotenv()

st.set_page_config(
    page_title="Document Analyzer – GenAI",
    layout="wide"
)

# =====================================================
# ENHANCED STYLES WITH FIXED BOTTOM INPUT
# =====================================================
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}

.stApp {
    background-color: #020617;
    color: #e5e7eb;
    padding-bottom: 140px; /* space for fixed input */
    margin-left: 15%;
    margin-right: 15%;
}

div[data-testid="stChatInput"] {
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
    padding-left: 17px;
    padding-right: 17px;
}        

/* keep content centered instead of margins */
.block-container {
    max-width: 900px;
    margin: auto;
}

/* fixed input – properly centered */
.fixed-input-container {
    position: fixed;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    max-width: 900px;
    background: #020617;
    padding: 20px;
    border-top: 2px solid #4b5563;
    z-index: 9999;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE (CLEAN & SINGLE SOURCE)
# =====================================================
if "processing" not in st.session_state:
    st.session_state.processing = False

if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {}

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# =====================================================
# LLM (ALWAYS AVAILABLE)
# =====================================================
llm = get_llm()

# =====================================================
# HEADER
# =====================================================
st.title("📄 Document Analyzer")
st.caption("Chat with your document or ask general questions")

# =====================================================
# SIDEBAR — DOCUMENT UPLOAD (ONE PIPELINE ONLY)
# =====================================================
with st.sidebar:
    st.header("Upload Content here")

    uploaded_file = st.file_uploader(
        "Share your document here and let the AI analyze it",
        type=["pdf", "txt", "docx", "jpg", "jpeg", "png"]
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        current_hash = hashlib.md5(file_bytes).hexdigest()

        if st.session_state.file_hash != current_hash:
            with st.spinner("Processing document..."):
                docs = load_document(uploaded_file)
                chunks = split_text(docs)
                vectordb = create_vectorstore(chunks)

                st.session_state.vectorstore = vectordb
                st.session_state.qa_chain = get_qa_chain(vectordb)
                st.session_state.doc_processed = True
                st.session_state.file_hash = current_hash

                st.session_state.doc_stats = {
                    "pages": len({d.metadata.get("page", 1) for d in docs}),
                    "chunks": len(chunks),
                    "words": sum(len(d.page_content.split()) for d in docs)
                }

            st.success("Document loaded successfully")

        else:
            st.info("Document already loaded")

    else:
        st.info("No document uploaded. Chat works in normal mode.")

# =====================================================
# DOCUMENT STATS 
# =====================================================
if st.session_state.doc_processed:
    st.markdown("### 📊 Document Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pages", st.session_state.doc_stats["pages"])
    c2.metric("Chunks", st.session_state.doc_stats["chunks"])
    c3.metric("Words", f"~{st.session_state.doc_stats['words']}")

st.divider()

# =====================================================
# CHAT UI - SCROLLABLE MESSAGES
# =====================================================
st.markdown("## 💬 Chat")

# Create scrollable container for chat messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Render full chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Show thinking indicator if processing
if st.session_state.processing:
    with st.chat_message("assistant"):
        st.markdown('<div class="thinking-indicator"> Thinking...</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# FIXED BOTTOM CHAT INPUT
# =====================================================
user_prompt = st.chat_input("Ask anything…")
submitted = bool(user_prompt)

# =====================================================
# HANDLE NEW MESSAGE
# =====================================================
if submitted and user_prompt:
    st.session_state.processing = True
    st.session_state.current_question = user_prompt
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })
    st.rerun()

# =====================================================
# GENERATE ANSWER
# =====================================================
if st.session_state.processing and st.session_state.current_question:
    try:
        if st.session_state.vectorstore and st.session_state.qa_chain:
            result = st.session_state.qa_chain(st.session_state.current_question)
            answer = result["answer"]
        else:
            messages = BASE_PROMPT.format_messages(
                context="No document uploaded.",
                question=st.session_state.current_question
            )
            answer = llm.invoke(messages).content

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        st.session_state.processing = False
        st.session_state.current_question = ""
        st.rerun()
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Sorry, an error occurred: {str(e)}"
        })
        st.session_state.processing = False
        st.session_state.current_question = ""
        st.rerun()

# =====================================================
# RESET BUTTON 
# =====================================================
if st.session_state.doc_processed:
    st.divider()
    if st.button("🔄 Reset Document"):
        st.session_state.clear()
        st.rerun()