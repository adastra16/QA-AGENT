# backend/main.py
import os
import json
import uuid
import logging
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel

# ChromaDB persistent client
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter



# HTML parsing
from bs4 import BeautifulSoup

from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa-agent")

CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# Chroma client
client = PersistentClient(path=CHROMA_DIR)
collection_name = "qa_agent_docs"
try:
    collection = client.get_collection(name=collection_name)
    logger.info(f"Loaded existing chroma collection '{collection_name}'")
except Exception:
    collection = client.create_collection(name=collection_name)
    logger.info(f"Created new chroma collection '{collection_name}'")

# FastAPI app + CORS
app = FastAPI(title="QA-Agent Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# persistent testcase store
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

def extract_text_from_file(filename: str, bytes_data: bytes) -> str:
    name = filename.lower()
    text = ""
    try:
        if name.endswith(".html") or name.endswith(".htm"):
            soup = BeautifulSoup(bytes_data.decode("utf-8", errors="ignore"), "html.parser")
            text = soup.get_text(separator="\n")
            els = []
            for el in soup.find_all(True):
                attrs = {k: v for k, v in el.attrs.items() if k in ("id", "name", "class", "type", "placeholder")}
                if attrs:
                    els.append(f"<{el.name}> attrs={attrs} text={el.get_text().strip()[:60]}")
            if els:
                text += "\n\nHTML_ELEMENTS:\n" + "\n".join(els)
        elif name.endswith((".md", ".txt", ".json")):
            text = bytes_data.decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(stream=bytes_data, filetype="pdf")
                pages = [p.get_text() for p in doc]
                text = "\n".join(pages)
            except Exception as e:
                logger.warning(f"PDF extraction failed for {filename}: {e}")
                text = ""
        else:
            text = bytes_data.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.exception(f"Error while extracting text from {filename}: {e}")
        text = ""
    return text

# Embedding helper: supports two modes:
# 1) If OPENAI_API_KEY is present -> use OpenAI embeddings (cheaper RAM on server)
# 2) Else try to import sentence_transformers lazily and use local model
def get_embedder():
    """
    Returns a function embed_texts(list[str]) -> Iterable[list[float]]
    """
    # prefer OpenAI if key present
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            import openai
            openai.api_key = api_key
            def openai_embed(texts):
                res = openai.Embedding.create(model="text-embedding-3-small", input=texts)
                return [r["embedding"] for r in res["data"]]
            logger.info("Using OpenAI embeddings (OPENAI_API_KEY detected).")
            return openai_embed
        except Exception as e:
            logger.exception("OpenAI client import failed, falling back to local embedder: %s", e)

    # fallback: lazy-load sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL_NAME)
        def st_embed(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=False)
        logger.info("Using local SentenceTransformer model for embeddings.")
        return st_embed
    except Exception as e:
        logger.exception("Local SentenceTransformer embedding is not available: %s", e)
        raise RuntimeError("No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers and torch on the server.")

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
    logger.info(f"[build_kb] Received paths: {file_paths}")
    docs = []
    metadatas = []
    ids = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for raw_path in file_paths:
        if not isinstance(raw_path, str):
            continue
        path = raw_path.replace("/", os.sep).replace("\\\\", "\\")
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

    # Create embeddings using chosen backend
    try:
        embed_fn = get_embedder()
        embeddings = embed_fn(docs)
    except Exception as e:
        logger.exception("Embedding generation failed: %s", e)
        return {"status": "error", "error": "embedding_failed", "details": str(e)}

    # Some embedders return numpy arrays, some lists - normalize to list of floats
    try:
        emb_list = [list(map(float, e)) for e in embeddings]
    except Exception:
        emb_list = list(embeddings)

    # Add to chroma collection
    try:
        collection.add(documents=docs, metadatas=metadatas, ids=ids, embeddings=emb_list)
    except Exception as e:
        logger.exception("Chroma add failed: %s", e)
        return {"status": "error", "error": "chroma_add_failed", "details": str(e)}

    logger.info(f"[build_kb] ingested {len(docs)} chunks from {len(set([m['source'] for m in metadatas]))} files")
    return {"status": "kb_built", "num_chunks": len(docs), "ingested_files": list({m['source'] for m in metadatas})}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/generate_testcases/")
async def generate_testcases(req: QueryRequest):
    try:
        results = collection.query(query_texts=[req.query], n_results=req.top_k)
    except Exception as e:
        logger.exception("Chroma query failed: %s", e)
        return {"status": "error", "error": "chroma_query_failed", "details": str(e)}

    retrieved_docs = []
    try:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_docs.append({"text": doc, "meta": meta})
    except Exception:
        logger.warning("Unexpected results shape from collection.query()")
        retrieved_docs = []

    testcases = []
    for idx, item in enumerate(retrieved_docs, start=1):
        tc = {
            "Test_ID": f"TC-{idx:03}",
            "Feature": "Auto-generated",
            "Test_Scenario": f"Validate: {req.query}",
            "Expected_Result": "Works as expected.",
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

class SeleniumRequest(BaseModel):
    testcase_id: str

@app.post("/generate_selenium_script/")
async def generate_selenium_script(req: SeleniumRequest):
    tc = GENERATED_TESTCASES.get(req.testcase_id)
    if not tc:
        return {"error": "testcase_not_found"}
    testcase = tc["payload"]
    # Minimal stable script
    script = f"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get('http://example.com/')

print('Running Testcase: {testcase['Test_ID']}')
print('Scenario: {testcase['Test_Scenario']}')
print('Expected: {testcase['Expected_Result']}')

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
driver.quit()
""".strip()
    return {"status": "ok", "selenium_script": script}
