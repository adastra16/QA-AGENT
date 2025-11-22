# backend/main.py
import os
import json
import uuid
import logging
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel

# ChromaDB
from chromadb import PersistentClient

# Embedding model
from sentence_transformers import SentenceTransformer

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# HTML parsing
from bs4 import BeautifulSoup

from fastapi.middleware.cors import CORSMiddleware

# ================================
# App + CORS
# ================================
app = FastAPI(title="QA-Agent Backend")

# NOTE: For development you can use ["*"].
# When you deploy, replace "*" with the exact Streamlit app origin:
# e.g. "https://your-streamlit-subdomain.streamlit.app"
CORS_ORIGINS = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    # "https://app-agent-d6uqpfm5wsp37zyyfkks9p.streamlit.app"  <-- add your deployed UI origin here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use CORS_ORIGINS here in production for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# Logging
# ================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa-agent")

# ================================
# Embedding / Chroma config
# ================================
CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# ================================
# Chroma client & collection
# ================================
client = PersistentClient(path=CHROMA_DIR)

collection_name = "qa_agent_docs"
try:
    collection = client.get_collection(collection_name)
    logger.info(f"Loaded collection '{collection_name}'")
except Exception:
    collection = client.create_collection(collection_name)
    logger.info(f"Created collection '{collection_name}'")

# ================================
# Embedding model
# ================================
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# ================================
# Persistent Testcase Storage
# ================================
TESTCASE_FILE = "generated_testcases.json"

if os.path.exists(TESTCASE_FILE):
    try:
        with open(TESTCASE_FILE, "r", encoding="utf-8") as f:
            GENERATED_TESTCASES = json.load(f)
    except Exception:
        GENERATED_TESTCASES = {}
else:
    GENERATED_TESTCASES = {}

def save_testcases():
    try:
        with open(TESTCASE_FILE, "w", encoding="utf-8") as f:
            json.dump(GENERATED_TESTCASES, f, indent=2)
    except Exception as e:
        logger.exception(f"Failed to save testcases: {e}")

# ================================
# Helpers
# ================================
def extract_text_from_file(filename: str, bytes_data: bytes) -> str:
    """
    Heuristic extractor for html, md, txt, json, pdf.
    """
    name = filename.lower()
    text = ""

    try:
        if name.endswith(".html") or name.endswith(".htm"):
            soup = BeautifulSoup(bytes_data.decode("utf-8", errors="ignore"), "html.parser")
            text = soup.get_text(separator="\n")

            # collect element attribute summaries (helpful for selector grounding)
            els = []
            for el in soup.find_all(True):
                attrs = {k: v for k, v in el.attrs.items() if k in ("id", "name", "class", "type", "placeholder")}
                if attrs:
                    els.append(f"<{el.name}> attrs={attrs} text={el.get_text().strip()[:60]}")
            if els:
                text += "\n\nHTML_ELEMENTS:\n" + "\n".join(els)

        elif name.endswith((".md", ".txt", ".json")):
            try:
                text = bytes_data.decode("utf-8", errors="ignore")
            except Exception:
                text = str(bytes_data)

        elif name.endswith(".pdf"):
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=bytes_data, filetype="pdf")
                pages = [p.get_text() for p in doc]
                text = "\n".join(pages)
            except Exception as e:
                logger.warning(f"PDF extraction failed for {filename}: {e}")
                text = ""

        else:
            try:
                text = bytes_data.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
    except Exception as e:
        logger.exception(f"Error extracting text from {filename}: {e}")
        text = ""

    return text

# ================================
# Endpoints
# ================================
@app.post("/upload_files/")
async def upload_files(files: List[UploadFile] = File(...)):
    saved = []
    upload_dir = "uploaded_assets"
    os.makedirs(upload_dir, exist_ok=True)

    for f in files:
        path = os.path.join(upload_dir, f.filename)
        content = await f.read()
        with open(path, "wb") as fh:
            fh.write(content)
        saved.append({"filename": f.filename, "path": path})
        logger.info(f"Saved uploaded file to {path}")

    return {"status": "ok", "saved": saved}


@app.post("/build_kb/")
async def build_kb(
    file_paths: List[str] = Body(...),
    chunk_size: int = Body(1000),
    chunk_overlap: int = Body(200)
):
    """
    Build knowledge base from provided file paths.
    Accepts a JSON body with:
      - file_paths: list of local paths (strings)
      - chunk_size, chunk_overlap: integers
    """
    logger.info(f"[build_kb] received file_paths = {file_paths}")

    docs = []
    metadatas = []
    ids = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for raw_path in file_paths:
        if not isinstance(raw_path, str):
            continue
        # normalize path separators
        path = raw_path.replace("\\\\", "\\").replace("/", os.sep).replace("\\", os.sep)

        if not os.path.exists(path):
            logger.warning(f"[build_kb] path not found: {path}")
            continue

        with open(path, "rb") as fh:
            b = fh.read()

        text = extract_text_from_file(path, b)
        if not text:
            logger.warning(f"[build_kb] no text extracted from: {path}")
            continue

        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            docs.append(chunk)
            metadatas.append({"source": os.path.basename(path), "path": path, "chunk_index": i})
            ids.append(doc_id)

    if len(docs) == 0:
        return {"status": "no_docs_found", "received_paths": file_paths}

    embeddings = embed_model.encode(docs, show_progress_bar=False, convert_to_numpy=True)

    # Add to Chroma collection
    try:
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
    except Exception as e:
        # Some chroma builds accept numpy arrays directly; try fallback
        try:
            collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=embeddings)
        except Exception:
            logger.exception(f"Failed adding to chroma collection: {e}")
            return {"status": "error", "error": "chroma_add_failed", "details": str(e)}

    logger.info(f"[build_kb] ingested {len(docs)} chunks from {len(set([m['source'] for m in metadatas]))} files")

    return {"status": "kb_built", "num_chunks": len(docs), "ingested_files": list({m['source'] for m in metadatas})}


# ---------------------------
# Testcase generation (simple rule-based)
# ---------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/generate_testcases/")
async def generate_testcases(req: QueryRequest):
    # Retrieve top_k docs from Chroma
    try:
        results = collection.query(query_texts=[req.query], n_results=req.top_k)
    except Exception as e:
        logger.exception(f"Chroma query failed: {e}")
        return {"status": "error", "error": "chroma_query_failed", "details": str(e)}

    retrieved_docs = []
    try:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_docs.append({"text": doc, "meta": meta})
    except Exception:
        logger.warning("Unexpected results shape from collection.query()")
        retrieved_docs = []

    # Simple rule-based testcase generator (no LLM)
    testcases = []
    for idx, item in enumerate(retrieved_docs, start=1):
        tc = {
            "Test_ID": f"TC-{idx:03}",
            "Feature": "Checkout Flow",
            "Test_Scenario": f"Validate: {req.query}",
            "Expected_Result": "Checkout should function correctly.",
            "Grounded_In": [item["meta"].get("source", "unknown")]
        }
        key = str(uuid.uuid4())
        GENERATED_TESTCASES[key] = {"id": key, "payload": tc}
        testcases.append({"id": key, "payload": tc})

    save_testcases()
    return {"status": "ok", "generated": testcases, "retrieved": len(retrieved_docs)}


@app.get("/list_testcases/")
async def list_testcases():
    return {"count": len(GENERATED_TESTCASES), "items": list(GENERATED_TESTCASES.values())}


# ---------------------------
# Selenium script generation
# ---------------------------
class SeleniumRequest(BaseModel):
    testcase_id: str


@app.post("/generate_selenium_script/")
async def generate_selenium_script(req: SeleniumRequest):
    tc = GENERATED_TESTCASES.get(req.testcase_id)
    if not tc:
        return {"error": "testcase_not_found"}

    testcase = tc["payload"]

    # Simple static selenium script (no GPT) - user should replace selectors with real ones
    script = f"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get("http://example.com/checkout")

print("Running Testcase: {testcase['Test_ID']}")

# NOTE: This is a placeholder script because no real DOM was inspected.
# Replace selectors and steps with the real page selectors and assertions.

try:
    print("Test Scenario: {testcase['Test_Scenario']}")
    print("Expected: {testcase['Expected_Result']}")

    # Example wait
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    print("✓ Page loaded successfully")

except Exception as e:
    print("✗ Test failed:", e)

driver.quit()
""".strip()

    # Return both keys to support different UI versions
    return {"status": "ok", "script": script, "selenium_script": script}
