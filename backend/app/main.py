from fastapi import FastAPI
from app.routers import documents

app = FastAPI(title="Tharindu Chatbot Document Processing API")

app.include_router(documents.router, prefix="/documents", tags=["documents"])

@app.get("/")
def read_root():
    return {"message": "Document Processing Pipeline API is running."}
