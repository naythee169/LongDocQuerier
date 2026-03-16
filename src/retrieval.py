"""
Retrieval module.

Two-stage pipeline:
  Stage 1 — Hybrid retrieval: BM25 keyword search + FAISS semantic search,
             fused with Reciprocal Rank Fusion (RRF).
  Stage 2 — Jina AI Reranker API: the top RETRIEVAL_TOP_K candidates are
             scored jointly with the query for precise relevance ordering.

All API calls go to Jina AI (free tier)

"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import faiss
import numpy as np
import requests

from config import (
    BM25_TOP_K,
    EMBEDDING_DIMENSION,
    INDEX_DIR,
    JINA_EMBEDDING_MODEL,
    JINA_RERANK_MODEL,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
    SEMANTIC_TOP_K,
)
from parser import Chunk


JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

load_dotenv()

@dataclass
class RetrievedChunk:
    """A chunk with its retrieval scores attached."""
    chunk: Chunk
    rrf_score: float
    rerank_score: float = 0.0


class Index:
    """Loads and holds all search indexes from disk."""

    def __init__(self, index_dir: str | Path = INDEX_DIR):
        index_dir = Path(index_dir)

        self.faiss_index: faiss.Index = faiss.read_index(
            str(index_dir / "faiss.index")
        )

        with open(index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks: list[Chunk] = [
            Chunk(
                text=c["text"],
                chunk_id=c["chunk_id"],
                page_start=c["page_start"],
                page_end=c["page_end"],
                section_heading=c.get("section_heading"),
                token_count=c["token_count"],
            )
            for c in raw
        ]

        print(f"Loaded index: {len(self.chunks)} chunks")


# Stage 1a: Semantic retrieval via Jina embeddings

def _semantic_search(
    query: str,
    faiss_index: faiss.Index,
    api_key: str,
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Embed the query via Jina AI and search the FAISS index.
    Returns list of (chunk_id, score) sorted by descending score.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        JINA_EMBED_URL,
        headers=headers,
        json={"model": JINA_EMBEDDING_MODEL, "input": [query]},
    )
    response.raise_for_status()
    query_vec = np.array(
        [response.json()["data"][0]["embedding"]], dtype=np.float32
    )
    faiss.normalize_L2(query_vec)

    scores, indices = faiss_index.search(query_vec, top_k)
    return list(zip(indices[0].tolist(), scores[0].tolist()))


# Stage 1b: BM25 keyword retrieval

def _bm25_search(
    query: str,
    bm25,
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Score all chunks with BM25 and return the top_k.
    Returns list of (chunk_id, score) sorted by descending score.
    """
    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)
    top_indices = np.argsort(scores)[::-1][:top_k].tolist()
    return [(idx, float(scores[idx])) for idx in top_indices]


# Stage 1c: Reciprocal Rank Fusion

def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    RRF score = sum over lists of 1 / (k + rank).
    k=60 is the standard smoothing constant (Cormack et al., 2009).
    """
    rrf_scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# Stage 2: Jina AI Reranker

def _rerank(
    query: str,
    candidates: list[Chunk],
    api_key: str,
) -> list[tuple[Chunk, float]]:
    """
    Re-rank candidate chunks using the Jina AI Reranker API.

    The reranker processes (query, document) pairs jointly — equivalent to
    a cross-encoder — giving precise relevance scores without a local model.

    Returns chunks sorted by descending relevance score.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        JINA_RERANK_URL,
        headers=headers,
        json={
            "model": JINA_RERANK_MODEL,
            "query": query,
            "documents": [c.text for c in candidates],
            "top_n": RERANK_TOP_K,
        },
    )
    response.raise_for_status()
    results = response.json()["results"]

    # Jina returns results sorted by relevance_score descending,
    # with an "index" field pointing back to the original candidates list.
    ranked = [
        (candidates[r["index"]], r["relevance_score"])
        for r in results
    ]
    return ranked


class Retriever:
    """
    Stateful retriever that holds loaded indexes.
    Instantiate once at startup; call retrieve() per query.
    """

    def __init__(self, index_dir: str | Path = INDEX_DIR):
        self._api_key = os.environ.get("JINA_API_KEY")
        if not self._api_key:
            raise EnvironmentError(
                "JINA_API_KEY environment variable not set.\n"
                "Get a free key at https://jina.ai — no credit card required."
            )
        self._index = Index(index_dir)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Full two-stage retrieval for a single query.

        Stage 1: Hybrid BM25 + semantic → RRF fusion → top RETRIEVAL_TOP_K
        Stage 2: Jina reranker → top RERANK_TOP_K

        Args:
            query: Natural language question from the user.

        Returns:
            List of RetrievedChunk objects ordered by descending relevance.
        """
        # Stage 1: Hybrid retrieval
        semantic_results = _semantic_search(
            query, self._index.faiss_index, self._api_key, SEMANTIC_TOP_K
        )
        bm25_results = _bm25_search(
            query, self._index.bm25, BM25_TOP_K
        )

        fused = _reciprocal_rank_fusion([semantic_results, bm25_results])
        top_fused = fused[:RETRIEVAL_TOP_K]

        chunk_map = {c.chunk_id: c for c in self._index.chunks}
        candidates = [
            (chunk_map[chunk_id], rrf_score)
            for chunk_id, rrf_score in top_fused
            if chunk_id in chunk_map
        ]

        # Stage 2: Rerank via Jina API
        candidate_chunks = [c for c, _ in candidates]
        rrf_score_map = {chunk_id: score for chunk_id, score in top_fused}

        reranked = _rerank(query, candidate_chunks, self._api_key)

        return [
            RetrievedChunk(
                chunk=chunk,
                rrf_score=rrf_score_map.get(chunk.chunk_id, 0.0),
                rerank_score=score,
            )
            for chunk, score in reranked
        ]
