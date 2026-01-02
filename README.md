# Multimodal RAG Chatbot

An advanced **Multimodal Retrieval-Augmented Generation (RAG)** system that processes PDFs to understand both text and images, allowing users to ask natural language questions and receive answers supported by visual context.

## 🚀 Key Features

* **Multimodal Ingestion**: Parses PDFs using `Docling` to extract text hierarchy and images (with captions).
* **Hybrid Retrieval**: uses **Milvus** to perform vector search on:
  * **Text**: `BAAI/bge-large-en-v1.5` embeddings.
  * **Images**: `google/siglip-so400m-patch14-384` embeddings + `BLIP` captioning.
* **Intelligent Linking**: Automatically links text paragraphs to relevant images ("Figure 1") and vice-versa.
* **Interactive UI**: A **Streamlit** frontend for document upload, management, and chat.
* **Model Agnostic**: Supports **OpenAI**, **Groq**, and **Ollama** for answer generation.

## 🛠️ System Architecture

### 1. Backend (`/backend`)

* **Framework**: FastAPI
* **Database**: Milvus (Vector Store)
* **Services**:
  * `parser.py`: Extracts content from PDFs.
  * `multimodal.py`: Generates embeddings and maps image-text relationships.
  * `rag.py`: Orchestrates the retrieval and answer generation.

### 2. Frontend (`/frontend`)

* **Framework**: Streamlit
* **Features**:
  * **Sidebar**: Document Upload & List.
  * **Chat**: Main interface supporting text+image responses.
  * **Config**: Real-time LLM provider selection.

## 📦 Setup & Usage

### Prerequisites

* Docker & Docker Compose
* Python 3.10+

### 1. Start Infrastructure (Milvus)

```bash
docker-compose up -d
```

### 2. Start Backend

```bash
cd backend
python -m venv .venv
# source .venv/bin/activate
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload
```

*API will run at `http://localhost:8000`*

### 3. Start Frontend

```bash
cd frontend
pip install -r requirements.txt

# Run UI
streamlit run app.py
```

*UI will open at `http://localhost:8501`*

## 📝 API Reference

### `POST /documents/upload`

Upload a PDF for processing. Triggers the background ETL pipeline.

### `GET /documents/list`

List all currently processed documents.

### `POST /chat/query`

Ask a question.

```json
{
  "query": "Show me process and structure  ",
  "provider": "openai"
}
```

## ⚠️ Current Status & Optimization

**Completed Phases:**

* [x] Document Parsing & Chunking
* [x] Multimodal Embedding Pipeline (Text + Image)
* [x] Hybrid RAG Logic
* [x] Streamlit Frontend Implementation

**Optimization Note:**
> **Answer Generation Optimization**: While the retrieval pipeline successfully finds relevant text and images, the **LLM Answer Generation** step requires further optimization. Currently, the prompt context integration could be improved to better synthesize information from both text snippets and image captions for a more cohesive final answer.

## 🔮 Future Roadmap

* [ ] **Deployment**: Dockerize the application stack.

* [ ] **Caching**: Redis layer for frequent queries.
* [ ] **Evaluation**: RAGAS framework integration for accuracy metrics.
