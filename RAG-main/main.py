"""
Enhanced RAG (Retrieval-Augmented Generation) Pipeline
With Conversation Memory and Improved Prompts
"""

import os

# Suppress ChromaDB telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from config import config


def create_embeddings():
    """Create embedding model using Ollama."""
    print(f"🔧 Loading embedding model: {config.EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.OLLAMA_BASE_URL,
    )
    print("✅ Embedding model loaded")
    return embeddings


def load_vector_store(embeddings):
    """Load existing vector store."""
    print(f"📂 Loading vector store from: {config.VECTOR_STORE_PATH}")
    vectorstore = Chroma(
        persist_directory=config.VECTOR_STORE_PATH,
        embedding_function=embeddings
    )
    print("✅ Vector store loaded")
    return vectorstore


def create_llm():
    """Initialize Ollama LLM with optimized settings."""
    print(f"🤖 Initializing LLM: {config.OLLAMA_MODEL}")
    llm = OllamaLLM(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=config.TEMPERATURE,
        num_ctx=config.CONTEXT_WINDOW,
    )
    print("✅ LLM initialized")
    return llm


def create_memory():
    """Create conversation memory to remember context."""
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5  # Remember last 5 exchanges
    )
    return memory


def create_conversational_chain(vectorstore, llm, memory):
    """Create conversational RAG chain with memory."""
    
    condense_prompt = PromptTemplate(
        template=config.CONDENSE_PROMPT,
        input_variables=["chat_history", "question"]
    )
    
    qa_prompt = PromptTemplate(
        template=config.QA_PROMPT,
        input_variables=["context", "question"]
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RETRIEVAL_K}
        ),
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False
    )
    
    return chain


def setup_rag_pipeline():
    """Set up the enhanced RAG pipeline."""
    print("\n" + "="*60)
    print("🚀 Setting up Enhanced RAG Pipeline")
    print("="*60 + "\n")
    
    # Print config
    config.print_config()
    
    # Check if vector store exists
    if not os.path.exists(config.VECTOR_STORE_PATH) or not os.listdir(config.VECTOR_STORE_PATH):
        print("\n❌ Vector store not found!")
        print(f"   Expected location: {config.VECTOR_STORE_PATH}")
        print("\n💡 Run 'python create_embeddings.py' first.")
        raise FileNotFoundError("Vector store not found.")
    
    embeddings = create_embeddings()
    vectorstore = load_vector_store(embeddings)
    llm = create_llm()
    memory = create_memory()
    
    print("\n🔗 Creating conversational chain with memory...")
    chain = create_conversational_chain(vectorstore, llm, memory)
    print("✅ Chain ready with conversation memory!")
    
    print("\n" + "="*60)
    print("✅ Enhanced RAG Pipeline Ready!")
    print("="*60 + "\n")
    
    return chain, vectorstore, memory


def query_rag(chain, question: str, show_sources: bool = True):
    """Query the RAG pipeline with conversation memory."""
    print(f"\n❓ Question: {question}")
    print("-" * 50)
    
    result = chain.invoke({"question": question})
    
    print(f"\n📝 Answer:\n{result['answer']}")
    
    if show_sources and result.get('source_documents'):
        print("\n" + "-" * 50)
        print("📚 Sources:")
        for i, doc in enumerate(result['source_documents'], 1):
            page = doc.metadata.get('page', 'N/A')
            page_display = page + 1 if isinstance(page, int) else page
            print(f"\n  [{i}] Page {page_display}:")
            content = doc.page_content[:300].replace('\n', ' ')
            print(f"      {content}...")
    
    return result


def clear_memory(memory):
    """Clear conversation memory."""
    memory.clear()
    print("🧹 Conversation memory cleared!")


def show_help():
    """Show available commands."""
    print("\n" + "="*50)
    print("📖 Available Commands:")
    print("="*50)
    print("  • Type your question to ask about the document")
    print("  • 'clear' - Clear conversation memory")
    print("  • 'history' - Show conversation history")
    print("  • 'config' - Show current configuration")
    print("  • 'help' - Show this help message")
    print("  • 'quit' or 'exit' - Exit the program")
    print("="*50)


def show_history(memory):
    """Show conversation history."""
    messages = memory.chat_memory.messages
    if not messages:
        print("\n📜 No conversation history yet.")
        return
    
    print("\n" + "="*50)
    print("📜 Conversation History:")
    print("="*50)
    for i, msg in enumerate(messages):
        role = "You" if msg.type == "human" else "Assistant"
        content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
        print(f"\n{role}: {content}")


def interactive_mode(chain, memory):
    """Run interactive Q&A mode with memory."""
    print("\n" + "="*60)
    print("💬 Interactive RAG Chat (with Memory)")
    print("Type 'help' for commands, 'quit' to exit")
    print("="*60)
    
    while True:
        print()
        question = input("🎤 You: ").strip()
        
        if not question:
            continue
        
        cmd = question.lower()
        
        if cmd in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        elif cmd == 'clear':
            clear_memory(memory)
            continue
        elif cmd == 'history':
            show_history(memory)
            continue
        elif cmd == 'help':
            show_help()
            continue
        elif cmd == 'config':
            config.print_config()
            continue
        
        try:
            query_rag(chain, question)
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    try:
        chain, vectorstore, memory = setup_rag_pipeline()
        
        # Welcome message
        print("🎯 Ask me anything about your document!")
        print("💡 I remember our conversation, so feel free to ask follow-up questions.")
        
        # Start interactive mode
        interactive_mode(chain, memory)
        
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Make sure Ollama is running: 'ollama serve'")
        print(f"2. Verify model exists: 'ollama pull {config.OLLAMA_MODEL}'")
        print("3. Run 'python create_embeddings.py' first")
        raise


if __name__ == "__main__":
    main()
