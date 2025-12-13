# RAG Document Processing Pipeline (Phase 1 Complete)

This project implements a robust, RAG-ready backend for processing complex PDF documents. It utilizes a **hierarchical parsing and chunking strategy** to ensure AI models retrieval high-quality context, including text and related images.

## 🚀 Key Features Implemented

### 1. High-Fidelity PDF Parsing (`Docling`)
We integrated the **Docling** library to move beyond simple text extraction.
- **Why?** Standard parsers lose structure. Docling allows us to distinguish between headers, paragraphs, lists, tables, and images.
- **Implementation**: The `DocumentConverter` parses the PDF into a document tree. We verified this by handling both tree-based and iterator-based traversal to ensure robust content extraction across different PDF versions.

### 2. Hierarchical Structure Extraction
Instead of a "flat" text file, we build a **Section-aware Tree**.
- **Logic**: We detect headers to determine nesting levels (e.g., Chapter 1 -> Section 1.1).
- **Benefit**: This allows us to know exactly which section a piece of text belongs to, preserving semantic context.

### 3. Hierarchical & Context-Aware Chunking
We implemented a custom strategy to solve the "context loss" problem in RAG.
- **Grouping**: First, we aggregate text **by section**. Text from "Section 1" never bleeds into "Section 2".
- **Splitting**: We use a `RecursiveCharacterTextSplitter` (Target: 500 chars, Overlap: 100 chars) *within* each section.
- **Result**: Small, precise chunks that remain self-contained within their parent topic.

### 4. Image-to-Text Metadata Linking
A critical requirement was linking images to their relevant text.
- **Extraction**: Images are extracted and saved as PNG files.
- **Linking**: When a chunk is created, we identify which "source blocks" (paragraphs) generated it. If those blocks contained image references (e.g., `[IMAGE: ...]`), the Image ID is automatically attached to the text chunk.
- **Outcome**: When the LLM retrieves a text chunk, it also gets the exact images referenced in that text.

## 5. Verification & Validation
We validated the pipeline through a rigorous testing process:
1.  **Automated Script (`verify_pipeline.py`)**: A script uploads test PDFs and polls for the result, verifying the presence of sections, blocks, and image files.
2.  **Manual JSON Inspection**: We audited the generated `metadata.json` to confirm:
    *   Hierarchy correctness (Sections properly nested).
    *   Chunk integrity (Text matches PDF content).
    *   Image References (Chunks contain correct `image_ids`).

## 📂 Output Structure
Processed documents are stored in `backend/data/processed/{doc_id}/`:
- `metadata.json`: The complete structured data for RAG.
- `images/`: Extracted image files.

## 🛠️ How to Run
1.  **Start Backend**: `uvicorn app.main:app --reload`

## 📦 Module Breakdown

### `app/services/`
*   **`parser.py`**: The core engine.
    *   Initializes `docling.DocumentConverter`.
    *   Iterates through document items using `doc.iterate_items()` for robustness.
    *   Builds the `SectionNode` tree structure based on headers.
    *   Extracts images and associates them with their paragraph blocks.
*   **`chunker.py`**: Intelligent text splitting.
    *   `create_chunks(sections, blocks)`: Takes the structured data and splits text *per section*.
    *   Calculates overlaps (block spans) to map generated chunks back to their source `ParagraphBlock`s and linked `ImageAssets`.
*   **`storage.py`**: File system abstraction.
    *   Manages `data/raw` (original PDF uploads) and `data/processed` (JSON + Images).
    *   Ensures directories exist and handles safe file naming.

### `app/routers/`
*   **`documents.py`**: API Endpoints.
    *   `POST /upload`: Validates PDF input, saves the file, and spawns the background processing task.
    *   Uses FastAPI `BackgroundTasks` to ensure the API responds immediately while Docling runs asynchronously.

### `app/models/`
*   **`schema.py`**: Data contracts (Pydantic).
    *   `SectionNode`: Recursive model for the document tree.
    *   `FineChunk`: The final unit for RAG, containing `content`, `section_id`, and `image_ids`.
    *   `ImageAsset`: Metadata for extracted images.

### `app/`
*   **`main.py`**: Application entry point. Configures the FastAPI app and includes routers.