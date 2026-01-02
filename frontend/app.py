import streamlit as st
import httpx
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Tharindu Multimodal RAG",
    page_icon="🤖",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("Document Knowledge Base")
    
    # NEW: Model Selection
    st.subheader("Configuration")
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ["openai", "groq", "ollama"],
        index=0,
        help="Choose which AI model generates the answer."
    )
    st.divider()

    # Upload Section
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        if st.button("Process Upload"):
            with st.spinner("Uploading and Processing..."):
                try:
                    # Sync upload using httpx
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = httpx.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=60.0)
                    
                    if response.status_code == 200:
                        st.success(f"Uploaded: {uploaded_file.name}")
                        st.rerun() # Refresh to show in list
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    
    # List Documents Section
    st.subheader("Available Documents")
    
    # Function to fetch docs
    def fetch_documents():
        try:
            response = httpx.get(f"{API_BASE_URL}/documents/list")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception:
            return []
        except Exception:
            return []
            
    def resolve_image_path(path_str):
        """
        Resolves the local path of the image.
        Handles relative paths from backend, absolute paths, and running from different directories.
        """
        # 1. Normalize path separators
        path_str = str(path_str).replace("/", os.sep).replace("\\", os.sep)
        
        # 2. Check if it's already a valid absolute path
        if os.path.isabs(path_str) and os.path.exists(path_str):
            return path_str
            
        # 3. Candidates to check
        candidates = [
            path_str,                                      # Relative to CWD
            os.path.join("backend", path_str),             # From Root: backend/data/...
            os.path.join("..", "backend", path_str),       # From Frontend: ../backend/data/...
            os.path.join("..", path_str)                   # From Frontend: ../data/... (if path includes backend)
        ]
        
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        # Debug info (will show in Streamlit error if fails)
        # return f"MISSING: {path_str} (CWD: {os.getcwd()})" 
        return path_str

    docs = fetch_documents()
    if docs:
        for doc in docs:
            st.text(f"📄 {doc.get('name', 'Unknown')}")
    else:
        st.info("No documents found.")

# --- Main Chat Interface ---
st.title("🤖 Multimodal Chatbot")
st.caption("Ask questions about your uploaded documents. I can see both text and images!")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are images associated with this message
        if "images" in msg and msg["images"]:
            for img in msg["images"]:
                # The backend returns 'image_path'. 
                # Ideally, we should serve these via a static mount or read strictly.
                # For local dev, checking if path exists or if backend serves it.
                # Backend does NOT serve static files yet in main.py? 
                # We might need to add static mount in backend or read bytes here.
                # IMPLEMENTATION NOTE: Streamlit can display local path if same machine.
                # But safer to assume URL. 
                # Let's assume we display caption for now or try local path.
                
                # Check backend implementation: It returns 'image_path' which is absolute local path.
                # Streamlit can render local paths.
                resolved_path = resolve_image_path(img["image_path"])
                st.image(resolved_path, caption=img.get("caption", "Reference Image"))


# Chat Input
if prompt := st.chat_input("Ask a question..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Bot Response
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking ({llm_provider})..."):
            try:
                payload = {"query": prompt, "provider": llm_provider}
                response = httpx.post(f"{API_BASE_URL}/chat/query", json=payload, timeout=60.0)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    images = data.get("images", [])
                    
                    st.markdown(answer)
                    
                    if images:
                        for img in images:
                            resolved_path = resolve_image_path(img["image_path"])
                            st.image(resolved_path, caption=img.get("caption", "Reference Image"))
                    
                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "images": images
                    })
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
