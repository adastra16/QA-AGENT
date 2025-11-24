import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from retriever import Retriever
import shutil
from pathlib import Path

# Fix memory usage of Chroma
chromadb.api.client.SharedSystemClient.clear_system_cache()

client = chromadb.Client(
    Settings(
        anonymized_telemetry=False,
        is_persistent=False   # prevent Render OOM
    )
)

# Create collection
collection = client.create_collection(name="qa_agent_docs")

app = FastAPI(title="QA Agent")

# Allow CORS (important for Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploaded_docs")
UPLOAD_DIR.mkdir(exist_ok=True)

GENERATED_TESTCASES = {}


class TestcaseRequest(BaseModel):
    prompt: str


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Store uploaded files."""
    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"file_id": file_id, "filename": file.filename}


@app.post("/generate_testcases/")
async def generate_testcases(prompt: str = Form(...)):
    """Generate test cases using Retriever + prompt."""
    retriever = Retriever()

    results = retriever.retrieve(prompt, top_k=5)

    uid = str(uuid.uuid4())
    GENERATED_TESTCASES[uid] = {
        "id": uid,
        "prompt": prompt,
        "results": results
    }

    return GENERATED_TESTCASES[uid]


@app.get("/get_testcases/{test_id}")
async def get_testcases(test_id: str):
    return GENERATED_TESTCASES.get(test_id, {})


@app.get("/")
def root():
    return {"message": "Backend running successfully"}
