"""
Script to create embeddings from PDF document and store in ChromaDB vector store.
Run this script first to index your documents before using the RAG pipeline.
"""

import os
import shutil
import sys

# Suppress ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from config import config


def load_pdf(pdf_path: str):
    """Load PDF document and return pages."""
    print(f"📄 Loading PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"✅ Loaded {len(pages)} pages from PDF")
    return pages


def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    print(f"✂️ Splitting documents into chunks...")
    print(f"   Chunk size: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks


def create_embeddings():
    """Create embedding model using Ollama."""
    print(f"🔧 Initializing embedding model: {config.EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    print("✅ Embedding model ready")
    return embeddings


def create_vector_store(chunks, embeddings):
    """Create and persist ChromaDB vector store."""
    print(f"💾 Creating vector store at: {config.VECTOR_STORE_PATH}")
    print(f"   This may take a few minutes...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.VECTOR_STORE_PATH
    )
    
    print(f"✅ Vector store created with {len(chunks)} documents")
    return vectorstore


def delete_existing_store():
    """Delete existing vector store if it exists."""
    if os.path.exists(config.VECTOR_STORE_PATH):
        print(f"🗑️ Deleting existing vector store at: {config.VECTOR_STORE_PATH}")
        shutil.rmtree(config.VECTOR_STORE_PATH)
        print("✅ Old vector store deleted")


def main(pdf_path: str = None):
    """Main function to create embeddings."""
    print("\n" + "="*60)
    print("📚 Document Embedding Creator")
    print("="*60 + "\n")
    
    # Use provided path or default from config
    pdf_file = pdf_path or config.PDF_PATH
    
    # Print config
    config.print_config()
    
    try:
        # Step 1: Delete existing vector store (fresh start)
        delete_existing_store()
        
        # Step 2: Load PDF
        documents = load_pdf(pdf_file)
        
        # Step 3: Split into chunks
        chunks = split_documents(documents)
        
        # Step 4: Initialize embeddings model
        embeddings = create_embeddings()
        
        # Step 5: Create vector store with embeddings
        print("\n🔄 Creating embeddings (this may take a while)...")
        vectorstore = create_vector_store(chunks, embeddings)
        
        # Summary
        print("\n" + "="*60)
        print("✅ Embedding Creation Complete!")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   • PDF: {pdf_file}")
        print(f"   • Pages loaded: {len(documents)}")
        print(f"   • Chunks created: {len(chunks)}")
        print(f"   • Vector store: {config.VECTOR_STORE_PATH}")
        print(f"   • Embedding model: {config.EMBEDDING_MODEL}")
        print(f"\n💡 You can now run:")
        print(f"   • 'python main.py' for CLI")
        print(f"   • 'streamlit run app.py' for Web UI")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure the PDF file exists.")
        sys.exit(1)
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure Ollama is running: 'ollama serve'")
        print(f"   2. Pull the embedding model: 'ollama pull {config.EMBEDDING_MODEL}'")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    # Allow passing PDF path as argument
    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(pdf_arg)
