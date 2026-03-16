"""
Ingestion pipeline.

Embeds chunks using the Jina AI Embeddings API (free tier),
builds a FAISS index for semantic search and a BM25 index for keyword search,
then saves both to INDEX_DIR.

Run once per document:
    python ingest.py path/to/document.pdf

"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from dotenv import load_dotenv
import time

import faiss
import numpy as np
import requests
from rank_bm25 import BM25Okapi

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSION,
    INDEX_DIR,
    JINA_EMBEDDING_MODEL,
)
from parser import Chunk, parse_document


# Jina AI Embeddings
load_dotenv()
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"

def _embed_texts(texts: list[str], api_key: str) -> np.ndarray:
    import time

    BATCH_SIZE = 64
    all_embeddings: list[list[float]] = []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        # Exponential backoff retry
        max_retries = 5
        wait = 10   # start with 10 seconds on first 429

        for attempt in range(max_retries):
            response = requests.post(
                JINA_EMBED_URL,
                headers=headers,
                json={
                    "model": JINA_EMBEDDING_MODEL,
                    "input": batch,
                },
            )

            if response.status_code == 429:
                print(f"  Rate limited — waiting {wait}s before retry "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                wait *= 2   # double the wait each time: 10 → 20 → 40 → 80
                continue

            response.raise_for_status()
            break
        else:
            raise RuntimeError(
                f"Failed to embed batch after {max_retries} retries. "
                f"Try again later or reduce BATCH_SIZE further."
            )

        batch_embeddings = [item["embedding"] for item in response.json()["data"]]
        all_embeddings.extend(batch_embeddings)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks...")

        time.sleep(2)   # always wait 2s between batches regardless

    return np.array(all_embeddings, dtype=np.float32)


# FAISS index

def _build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS flat inner-product index.
    Vectors are L2-normalised first so inner product == cosine similarity.
    FlatIP is exact search — appropriate for document-scale indexes.
    """
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings)
    return index


# BM25 index

def _tokenise_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def _build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    tokenised = [_tokenise_for_bm25(chunk.text) for chunk in chunks]
    return BM25Okapi(tokenised)


# Serialisation

def _save_index(
    chunks: list[Chunk],
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    index_dir: Path,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)

    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "section_heading": c.section_heading,
            "token_count": c.token_count,
        }
        for c in chunks
    ]
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)

    with open(index_dir / "meta.json", "w") as f:
        json.dump(
            {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": JINA_EMBEDDING_MODEL,
                "num_chunks": len(chunks),
            },
            f,
            indent=2,
        )

    print(f"\nIndex saved to {index_dir}/")
    print(f"  faiss.index  — {len(chunks)} vectors, dim {EMBEDDING_DIMENSION}")
    print(f"  bm25.pkl     — BM25Okapi over {len(chunks)} chunks")
    print(f"  chunks.json  — full text + metadata")


# Public API

def ingest(pdf_path: str | Path) -> None:
    """
    Full ingestion pipeline:
      parse → embed (Jina AI) → build FAISS + BM25 → save to INDEX_DIR

    Args:
        pdf_path: Path to the PDF document to ingest.
    """
    api_key = os.environ.get("JINA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "JINA_API_KEY environment variable not set.\n"
            "Get a free key at https://jina.ai — no credit card required."
        )

    chunks = parse_document(pdf_path)

    print("\nEmbedding chunks via Jina AI...")
    texts = [chunk.text for chunk in chunks]
    embeddings = _embed_texts(texts, api_key)

    print("\nBuilding indexes...")
    faiss_index = _build_faiss_index(embeddings)
    bm25_index = _build_bm25_index(chunks)

    _save_index(chunks, faiss_index, bm25_index, Path(INDEX_DIR))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)

    ingest(sys.argv[1])
