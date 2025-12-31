import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.vector_store import MilvusService

def reset():
    print("Connecting to Milvus...")
    service = MilvusService()
    try:
        service.connect()
        print("Dropping collections...")
        service.drop_collections()
        print("Collections dropped. They will be recreated with new schema on next run/connect.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    reset()
