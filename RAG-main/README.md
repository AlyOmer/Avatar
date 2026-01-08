# 📚 RAG Document Chat

A **Retrieval-Augmented Generation (RAG)** pipeline that allows you to chat with your PDF documents using local LLMs via Ollama.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple.svg)

## ✨ Features

- 🔒 **100% Local** - All processing happens on your machine, no data sent to cloud
- 📄 **PDF Upload** - Upload any PDF through the web interface
- 💬 **Interactive Chat** - Ask questions about your documents
- 📚 **Source Citations** - See exactly where answers come from
- 🔄 **Streaming Responses** - Watch answers generate in real-time
- ⚙️ **Configurable** - Easy settings via config file or environment variables

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   PDF       │────▶│  Chunking   │────▶│  Embeddings │
│   Upload    │     │  (1000 char)│     │  (Nomic)    │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Answer    │◀────│   LLM       │◀────│  ChromaDB   │
│   + Sources │     │  (Llama3.1) │     │  Vector DB  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** - [Download here](https://ollama.com/download)
- **8GB+ RAM** recommended

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-document-chat.git
cd rag-document-chat
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama models

```bash
# Start Ollama
ollama serve

# Pull required models (in another terminal)
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

### 5. Run the application

**Option A: Web UI (Recommended)**
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser

**Option B: Command Line**
```bash
# First, create embeddings
python create_embeddings.py

# Then run chat
python main.py
```

## 📁 Project Structure

```
RAG/
├── app.py                 # Streamlit web interface
├── main.py                # CLI interface with memory
├── create_embeddings.py   # Create vector embeddings
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.example)
├── .env.example           # Example environment file
├── README.md              # This file
├── Dockerfile             # Docker container config
├── docker-compose.yml     # Docker compose config
└── chroma_db/             # Vector store (auto-created)
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or set these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM model for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Model for embeddings |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `VECTOR_STORE_PATH` | `./chroma_db` | Where to store vectors |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `2` | Number of docs to retrieve |
| `TEMPERATURE` | `0.1` | LLM temperature (0-1) |

### Using Different Models

```bash
# For faster responses (smaller model)
ollama pull llama3.2:1b

# For better quality (larger model)
ollama pull llama3.1:70b

# Update .env or config.py
OLLAMA_MODEL=llama3.2:1b
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

### Manual Docker Build

```bash
docker build -t rag-chat .
docker run -p 8501:8501 rag-chat
```

## 🔧 Troubleshooting

### "Failed to connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve

# Check if models are installed
ollama list
```

### "Vector store not found"
```bash
# Create embeddings first
python create_embeddings.py

# Or upload a PDF in the web UI
```

### Slow responses
- Use a smaller model: `llama3.2:1b`
- Reduce `RETRIEVAL_K` to 1
- Reduce `CONTEXT_WINDOW` to 1024

## 📊 Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Ollama (Llama 3.1) |
| **Embeddings** | Nomic Embed Text |
| **Vector Store** | ChromaDB |
| **Framework** | LangChain |
| **Web UI** | Streamlit |
| **Language** | Python 3.10+ |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [Ollama](https://ollama.com/) - Local LLM runtime
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Streamlit](https://streamlit.io/) - Web UI framework

---

Made with ❤️ by Omair
