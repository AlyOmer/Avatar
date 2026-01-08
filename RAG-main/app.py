"""
Streamlit Web UI for RAG Pipeline
Sophisticated chat interface for querying your documents
"""

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from config import config

# Suppress ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Page config
st.set_page_config(
    page_title="📚 RAG Document Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a cleaner, modern look
st.markdown(
    """
    <style>
        /* Layout */
        .main {padding: 1rem 2rem;}
        /* Chat bubbles */
        .user-bubble {
            background: linear-gradient(135deg, #e8f0fe 0%, #d2e3fc 100%);
            border: 1px solid #c2d6ff;
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        .assistant-bubble {
            background: linear-gradient(135deg, #f5f7fa 0%, #e9edf5 100%);
            border: 1px solid #dfe3ec;
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 8px;
        }
        /* Expander tweak */
        details > summary {
            font-weight: 600;
        }
        /* Upload box */
        .upload-box {
            border: 2px dashed #ccc;
            padding: 16px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Streaming callback
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")


@st.cache_resource
def load_embeddings():
    """Load embeddings model."""
    return OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )


def load_vectorstore(embeddings, path: str):
    """Load vector store at given path."""
    return Chroma(persist_directory=path, embedding_function=embeddings)


def create_llm(streaming: bool = False, callbacks=None):
    """Create LLM with optional streaming."""
    return OllamaLLM(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_ctx=config.CONTEXT_WINDOW,
        callbacks=callbacks if streaming else None,
    )


def create_qa_chain(vectorstore, llm):
    """Create QA chain."""
    prompt = PromptTemplate(template=config.QA_PROMPT, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVAL_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def process_pdf(uploaded_file, persist_path: str):
    """Process uploaded PDF and create embeddings."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)

        embeddings = load_embeddings()

        # Replace existing vector store at target path
        if os.path.exists(persist_path):
            try:
                shutil.rmtree(persist_path)
            except PermissionError:
                # If locked, raise to try a new path
                raise PermissionError(
                    f"Vector store path in use: {persist_path}. Please close other running instances or try again."
                )

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_path,
        )

        return len(documents), len(chunks)

    finally:
        os.unlink(tmp_path)


def init_session():
    """Initialize session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = os.path.exists(config.VECTOR_STORE_PATH)
    if "stats" not in st.session_state:
        st.session_state.stats = {"questions": 0, "sources": 0}
    if "vectorstore_path" not in st.session_state:
        st.session_state.vectorstore_path = config.VECTOR_STORE_PATH


def render_header():
    st.title("💬 RAG Document Chat")
    st.caption("Ask questions about your PDFs with local LLMs and on-device embeddings.")

    ready = "✅ Ready" if st.session_state.vectorstore_ready else "⚠️ Awaiting document"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", ready)
    col2.metric("Model", config.OLLAMA_MODEL)
    col3.metric("Embeddings", config.EMBEDDING_MODEL)
    col4.metric("Questions", st.session_state.stats.get("questions", 0))


def render_sidebar():
    with st.sidebar:
        st.title("📚 RAG Document Chat")
        st.markdown("---")

        # Upload
        st.markdown("### 📤 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], help="Upload a PDF to chat with")
        if uploaded_file:
            if st.button("🔄 Process PDF", use_container_width=True):
                with st.spinner("Processing PDF... This may take a minute."):
                    try:
                        # Use a unique path to avoid locking issues on Windows
                        base_path = Path(config.VECTOR_STORE_PATH)
                        target_path = base_path
                        if base_path.exists():
                            target_path = base_path.parent / f"{base_path.name}_{uuid.uuid4().hex[:8]}"
                        st.session_state.vectorstore_path = str(target_path)

                        pages, chunks = process_pdf(uploaded_file, persist_path=str(target_path))
                        st.session_state.vectorstore_ready = True
                        st.session_state.messages = []
                        st.session_state.stats = {"questions": 0, "sources": 0}
                        st.success(f"✅ Processed {pages} pages into {chunks} chunks!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_sources = st.checkbox("Show sources", value=True)
        use_streaming = st.checkbox("Stream response", value=True)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.stats = {"questions": 0, "sources": 0}
            st.experimental_rerun()

        st.markdown("---")
        st.markdown("### 📊 Info")
        st.caption(f"Model: {config.OLLAMA_MODEL}")
        st.caption(f"Embeddings: {config.EMBEDDING_MODEL}")
        st.caption(f"Vector Store: {Path(st.session_state.vectorstore_path).name}")

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown(
            """
            - Upload a PDF first
            - Ask focused, specific questions
            - Try "Summarize section X"
            - Use streaming for faster feedback
            """
        )

        return show_sources, use_streaming


def render_empty_state():
    st.info("👆 Upload a PDF in the sidebar to get started!")
    st.markdown(
        """
        ### How it works
        1. **Upload** a PDF document
        2. **Process** to create embeddings
        3. **Ask** questions about your document
        4. **Get** AI-powered answers with sources
        """
    )


def render_history(show_sources: bool):
    for msg in st.session_state.messages:
        bubble_class = "assistant-bubble" if msg["role"] == "assistant" else "user-bubble"
        with st.chat_message(msg["role"]):
            st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)
            if msg["role"] == "assistant" and show_sources and msg.get("sources"):
                with st.expander("📚 View Sources"):
                    for s in msg["sources"]:
                        st.caption(f"**Page {s['page']}:** {s['text'][:250]}...")


def main():
    init_session()
    show_sources, use_streaming = render_sidebar()
    render_header()

    # Guard: need vector store
    if not st.session_state.vectorstore_ready:
        render_empty_state()
        return

    # Load components
    try:
        embeddings = load_embeddings()
        vectorstore = load_vectorstore(embeddings, st.session_state.vectorstore_path)
        st.success("✅ Ready to chat!")
    except Exception as e:
        st.error(f"❌ Error loading: {e}")
        st.info("💡 Make sure Ollama is running: `ollama serve`")
        return

    # Chat history
    render_history(show_sources)

    # Chat input
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

        with st.chat_message("assistant"):
            try:
                if use_streaming:
                    response_container = st.empty()
                    stream_handler = StreamHandler(response_container)
                    llm = create_llm(streaming=True, callbacks=[stream_handler])
                else:
                    llm = create_llm(streaming=False)

                chain = create_qa_chain(vectorstore, llm)

                with st.spinner("Searching document..."):
                    result = chain.invoke({"query": prompt})

                answer = result["result"]
                sources = []
                if result.get("source_documents"):
                    for doc in result["source_documents"]:
                        page = doc.metadata.get("page", 0)
                        sources.append(
                            {
                                "page": page + 1 if isinstance(page, int) else page,
                                "text": doc.page_content,
                            }
                        )

                # Update stats
                st.session_state.stats["questions"] += 1
                st.session_state.stats["sources"] += len(sources)

                if use_streaming:
                    response_container.markdown(answer)
                else:
                    st.markdown(f"<div class='assistant-bubble'>{answer}</div>", unsafe_allow_html=True)

                if show_sources and sources:
                    with st.expander("📚 View Sources"):
                        for s in sources:
                            st.caption(f"**Page {s['page']}:** {s['text'][:250]}...")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("💡 Make sure Ollama is running")


if __name__ == "__main__":
    main()
