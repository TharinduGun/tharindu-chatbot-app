# Multimodal Module Testing (Phase 4)

This branch focuses on the unit testing and validation of the Multimodal RAG Pipeline (`backend/app/services/multimodal.py`).

## Testing Strategy: Mocking Heavy Models

The multimodal pipeline relies on massive state-of-the-art models:

1. **SigLIP** (Image & Text Embeddings)
2. **BGE** (Text Embeddings)
3. **BLIP** (Image Captioning)

Loading these models requires gigabytes of VRAM and significant time. To ensure our **Unit Tests** are fast, reliable, and runnable in any environment (including CI without GPUs), we employ **extensive mocking**.

We assume the underlying libraries (`transformers`, `torch`) work as expected. Our tests focus on validating the **business logic**, **data flow**, and **orchestration** of these models.

## Test Suite: `backend/tests/test_multimodal.py`

We have implemented a comprehensive test suite using `pytest`.

### Key Test Cases

1. **`test_validate_caption`**
   - **Goal**: Ensure the caption quality filter works.
   - **Logic**: Rejects captions like "Figure 1" or empty strings. Accepts descriptive text.

2. **`test_initialization`**
   - **Goal**: Verify that the `MultimodalPipeline` class initializes correctly (loading mocked processors/models).

3. **`test_get_bge_embedding` & `test_get_siglip_...`**
   - **Goal**: Validate that input text/images are correctly passed to the models and that the output vectors have the correct dimensions (e.g., 768 for BGE, 1024/1152 for SigLIP).

4. **`test_run_flow` (End-to-End Logic)**
   - **Goal**: Verify the full processing loop for a single document.
   - **Scope**:
     - Mocks the file system (`PROCESSED_DIR`) and storage.
     - Simulates image and chunk data.
     - **Verifies Linking Logic**: We inject specific "mock vectors" to force a high similarity score between a specific image and a text chunk. We then assert that the system correctly creates the bi-directional link (`linked_chunk_id` in image, `linked_image_ids` in chunk) and saves the result.

## How to Run Tests

Ensure you are in the project root.

```powershell
$env:PYTHONPATH="backend"; python -m pytest backend/tests/test_multimodal.py
```

## Validation Results

- **Status**: ✅ All Tests Passed
- **Coverage**: The tests successfully verify that the pipeline handles missing files, generates captions when needed, and correctly executes the hybrid matching algorithm (Image + Caption similarity).
