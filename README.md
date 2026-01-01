
## 5. RAG & LLM Integration (Phase 5)

We have integrated a **Large Language Model (LLM)** service to enable natural language querying of the processed documents.

### Architecture

1. **Unified LLM Service**: Abstracts providers (OpenAI, Groq, Ollama), allowing easy switching.
2. **Hybrid Retrieval**: Queries both Milvus text and image collections simultaneously.
3. **Context Merging**: Inteprets linked_chunk_id metadata to combine text snippets with their relevant images.

### The RAG Flow

1. **User Query**: What is the structure of the tooth?`n2. **Embedding**: Query converted to BGE (Text) and SigLIP (Image) vectors.
2. **Search**: Top 5 Texts + Top 5 Images retrieved from Milvus.
3. **Answer Generation**: The LLM receives a prompt with the user query and the retrieved context (Text + Image Captions).

## 6. API Reference

### POST /chat/query`n- **Purpose**: Ask questions about the document

- **Payload**:
  `json
  {
    "query": "Your question here",
    "provider": "openai" // or "groq", "ollama"
  }
  `

## 7. Future Roadmap (Phase 6)

- [ ] **Frontend Application**: Build a React/Streamlit UI for easier interaction.
- [ ] **Evaluation**: Implement automated accuracy testing (RAGAS or similar).
- [ ] **Deployment**: Dockerize the complete stack for production.
- [ ] **Caching**: Implement Redis to cache frequent queries.
