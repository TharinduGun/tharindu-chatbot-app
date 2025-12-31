import json
import os
from PIL import Image
from pathlib import Path

# Simulate running from backend directory
os.chdir("backend")
print(f"CWD: {os.getcwd()}")

# Find latest metadata
processed_dir = Path("data/processed")
if not processed_dir.exists():
    print("Processed dir not found!")
    exit(1)

latest_dir =  max(processed_dir.glob("*"), key=os.path.getmtime)
print(f"Latest Dir: {latest_dir}")

meta_path = latest_dir / "metadata.json"
with open(meta_path, "r") as f:
    data = json.load(f)

for img in data.get("images", []):
    path = img["file_path"]
    print(f"Checking: {path}")
    if os.path.exists(path):
        print("  [OK] File exists.")
        try:
            image = Image.open(path).convert("RGB")
            print("  [OK] Image.open success.")
        except Exception as e:
            print(f"  [FAIL] Image.open error: {e}")
    else:
        print("  [FAIL] File NOT found.")
