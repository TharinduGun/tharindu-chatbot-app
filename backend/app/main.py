from fastapi import FastAPI
from dotenv import load_dotenv
import os

load_dotenv()

from app.routers import documents, chat

app = FastAPI(title="Tharindu Chatbot Document Processing API")

app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])

@app.get("/")
def read_root():
    return {"message": "Document Processing Pipeline API is running."}
