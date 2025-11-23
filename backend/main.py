
import os
import json
import uuid
import logging
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, Body
from pydantic import BaseModel

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bs4 import BeautifulSoup
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="QA-Agent Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qa-agent")


CHROMA_DIR = "chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

client = PersistentClient(path=CHROMA_DIR)
collection_name = "qa_agent_docs"

try:
    collection = client.get_collection(collection_name)
    logger.info(f"Loaded collection '{collection_name}'")
except Exception:
    collection = client.create_collection(collection_name)
    logger.info(f"Created collection '{collection_name}'")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)



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
    with open(TESTCASE_FILE, "w", encoding="utf-8") as f:
        json.dump(GENERATED_TESTCASES, f, indent=2)


def extract_text_from_file(filename: str, bytes_data: bytes) -> str:
    name = filename.lower()

    try:
        if name.endswith(".html") or name.endswith(".htm"):
            soup = BeautifulSoup(bytes_data.decode("utf-8", errors="ignore"), "html.parser")
            text = soup.get_text(separator="\n")

            elements = []
            for el in soup.find_all(True):
                attrs = {k: v for k, v in el.attrs.items()
                         if k in ("id", "name", "class", "type", "placeholder")}
                if attrs:
                    elements.append(f"<{el.name}> attrs={attrs} text={el.get_text().strip()[:60]}")
            if elements:
                text += "\n\nHTML_ELEMENTS:\n" + "\n".join(elements)

            return text

        if name.endswith((".md", ".txt", ".json")):
            return bytes_data.decode("utf-8", errors="ignore")

        if name.endswith(".pdf"):
            try:
                import fitz
                doc = fitz.open(stream=bytes_data, filetype="pdf")
                return "\n".join([p.get_text() for p in doc])
            except:
                return ""

        return bytes_data.decode("utf-8", errors="ignore")

    except:
        return ""



@app.post("/upload_files/")
async def upload_files(files: List[UploadFile] = File(...)):
    saved = []
    upload_dir = "uploaded_assets"
    os.makedirs(upload_dir, exist_ok=True)

    for f in files:
        path = os.path.join(upload_dir, f.filename)
        with open(path, "wb") as fh:
            fh.write(await f.read())
        saved.append({"filename": f.filename, "path": path})

    return {"status": "ok", "saved": saved}



@app.post("/build_kb/")
async def build_kb(
    file_paths: List[str] = Body(...),
    chunk_size: int = Body(1000),
    chunk_overlap: int = Body(200)
):
    docs = []
    metadatas = []
    ids = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    for raw_path in file_paths:
        path = raw_path.replace("/", os.sep)
        if not os.path.exists(path):
            logger.warning(f"Missing file: {path}")
            continue

        with open(path, "rb") as f:
            content = f.read()

        text = extract_text_from_file(path, content)
        if not text:
            continue

        chunks = splitter.split_text(text)

        for i, c in enumerate(chunks):
            uid = str(uuid.uuid4())
            docs.append(c)
            metadatas.append({"source": os.path.basename(path), "path": path, "chunk_index": i})
            ids.append(uid)

    if not docs:
        return {"status": "no_docs_found", "received": file_paths}

    embeddings = embed_model.encode(docs, convert_to_numpy=True)

    try:
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings.tolist()
        )
    except:
        collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )

    return {
        "status": "kb_built",
        "num_chunks": len(docs),
        "ingested_files": list({m["source"] for m in metadatas})
    }


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/generate_testcases/")
async def generate_testcases(req: QueryRequest):
    try:
        result = collection.query(
            query_texts=[req.query],
            n_results=req.top_k
        )
    except Exception as e:
        return {"status": "error", "details": str(e)}

    retrieved = []
    try:
        for t, meta in zip(result["documents"][0], result["metadatas"][0]):
            retrieved.append({"text": t, "meta": meta})
    except:
        retrieved = []

    generated = []

    for i, item in enumerate(retrieved, start=1):
        tc = {
            "Test_ID": f"TC-{i:03}",
            "Feature": req.query,
            "Test_Scenario": f"Validate: {req.query}",
            "Expected_Result": "The feature should work as expected.",
            "Grounded_In": [item["meta"].get("source", "unknown")],
        }

        uid = str(uuid.uuid4())
        GENERATED_TESTCASES[uid] = {"id": uid, "payload": tc}
        generated.append({"id": uid, "payload": tc})

    save_testcases()
    return {"status": "ok", "generated": generated, "retrieved": len(retrieved)}



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

    script = f"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
driver.get("http://example.com/checkout")

print("Running Testcase: {testcase['Test_ID']}")
print("Scenario: {testcase['Test_Scenario']}")
print("Expected: {testcase['Expected_Result']}")

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

driver.quit()
""".strip()

    return {"status": "ok", "selenium_script": script}
