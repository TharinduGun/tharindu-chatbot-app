from pathlib import Path

env_path = Path("backend/.env")

try:
    # Try reading as UTF-16 (PowerShell default)
    content = env_path.read_text(encoding="utf-16")
    print("Read as UTF-16.")
except UnicodeError:
    try:
        # Try UTF-8 (just in case)
        content = env_path.read_text(encoding="utf-8")
        print("Read as UTF-8.")
    except Exception as e:
        print(f"Failed to read file: {e}")
        exit(1)

# Write back as UTF-8
env_path.write_text(content, encoding="utf-8")
print("Saved as UTF-8.")
