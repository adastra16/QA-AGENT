"""
Lightweight cosine-similarity retriever using MiniLM-L3-v2 (low RAM).
"""

import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_FILE = Path("ingest/embeddings.npy")
META_FILE = Path("ingest/embeddings_meta.json")

# Smaller model (less RAM on Render)
MODEL_NAME = "paraphrase-MiniLM-L3-v2"


class Retriever:
    def __init__(self, model_name: str = MODEL_NAME):

        if not EMBED_FILE.exists():
            raise FileNotFoundError(f"{EMBED_FILE} missing. Run ingest first.")
        if not META_FILE.exists():
            raise FileNotFoundError(f"{META_FILE} missing. Run ingest first.")

        # Load vectors
        self.vectors = np.load(EMBED_FILE)

        meta_text = META_FILE.read_text(encoding="utf-8")
        self.meta = json.loads(meta_text)

        # Normalize embeddings
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = self.vectors / norms

        # Load small model
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([text])[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else vec

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []

        qvec = self.embed_query(query)

        scores = np.dot(self.vectors, qvec)
        top_k_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_k_idx:
            m = self.meta[idx]
            results.append({
                "score": float(scores[idx]),
                "chunk_id": m.get("id") or m.get("chunk_id"),
                "source": m.get("source"),
                "index": m.get("index") or m.get("chunk_index"),
                "text": m.get("text")
            })

        return results


if __name__ == "__main__":
    r = Retriever()
    print("Retriever loaded.")
    while True:
        q = input("\nQuery> ").strip()
        if not q:
            break
        res = r.retrieve(q)
        for i, r0 in enumerate(res):
            print(f"\n[{i+1}] score={r0['score']:.4f}, {r0['source']}, idx={r0['index']}")
            print(r0['text'][:300].replace("\n", " "))
