from transformers import SiglipProcessor, SiglipModel
import torch
import time

print("Start Script")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

start = time.time()
print("Loading Processor...")
try:
    processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384", local_files_only=False)
    print("Processor Loaded.")
except Exception as e:
    print(f"Processor Failed: {e}")

print("Loading Model...")
try:
    model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-384", local_files_only=False)
    print("Model Loaded (CPU).")
    
    print(f"Moving to {device}...")
    model.to(device)
    print(f"Moved to {device}.")
except Exception as e:
    print(f"Model Failed: {e}")

print(f"Done in {time.time() - start:.2f}s")
