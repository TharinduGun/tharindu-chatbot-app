# Testing Strategy & Quality Assurance

This document outlines the testing framework implemented for the Document Processing Pipeline. The focus is on verifying the core logic of parsing, chunking, and storage through isolated unit tests.

## 1. Implemented Unit Tests

We have established a suite of unit tests located in `backend/tests/` to ensure reliability and robustness.

### A. Parser Service (`test_parser.py`)

**Goal**: Verify the orchestration of the document processing pipeline.

- **`test_process_document_success`**: Mocks `Docling`, `Chunker`, and `Storage` to verify that a document flows correctly from parsing to chunking to saving. Checks that metadata is structured correctly and the status is updated to "completed".
- **`test_process_document_failure`**: Simulates a crash in the conversion process to ensure the exception is caught and the document status is updated to "failed".

### B. Chunker Service (`test_chunker.py`)

**Goal**: Validate the logic for splitting text and preserving context.

- **`test_create_chunks_basic`**: Ensures that text is correctly split based on the `RecursiveCharacterTextSplitter` rules and that chunks are assigned the correct `section_id`.
- **`test_create_chunks_image_linking`**: critical verification that `image_ids` found in paragraph blocks are correctly propagated to the resulting fine-grained text chunks.

### C. Storage Service (`test_storage.py`)

**Goal**: Ensure file system interactions and registry management are safe and correct.

- **`test_calculate_content_hash`**: Verifies SHA-256 hash generation for duplicate detection.
- **`test_save_upload_file`**: Checks that uploaded files are saved to the correct `raw` directory.
- **`test_registry_operations`**: Validates the JSON registry logic, including adding new entries, updating status (e.g., specific processing states), and retrieving cached document metadata.

## 2. How to Run Tests

The project uses `pytest` for testing.

1. Navigate to the `backend` directory:

   ```bash
   cd backend
   ```

2. Activate your virtual environment (if not already active).
3. Run the tests:

   ```bash
   pytest
   ```

## 3. Expected & Future Tests

To further harden the system, the following tests are planned:

### Integration Tests

- **End-to-End Pipeline**: Run the full pipeline with a real PDF sample (not mocked) to verify compatibility between `Docling` output and our internal data structures.
- **Concurrency**: Verify behavior when multiple documents are processed simultaneously.

### API Tests

- **`TestClient` Verification**: Use FastAPI's `TestClient` to test endpoints like `POST /upload` and `GET /status`.
- **Error Handling**: Verify 400/500 responses for invalid inputs or server errors.

### Edge Case Verification

- **Empty/Malformed PDFs**: Ensure the parser handles zero-byte files or corrupted PDFs gracefully.
- **Deeply Nested Headers**: Validate hierarchical chunking on documents with complex TOC structures (e.g., Level 5 headers).
- **Large Images**: Verify extraction performance on high-resolution images.
