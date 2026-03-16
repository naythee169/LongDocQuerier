"""
Diagnostic visualisation for the ingested index.

Shows:
  1. UMAP projection of chunk embeddings — reveals how the document
     clusters semantically in vector space.
  2. BM25 score distribution for a test query — shows which chunks
     the keyword retriever ranks highly.

Usage:
    python visualise_index.py
    python visualise_index.py "your test query here"

Run this after ingest.py has been executed.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import umap

from config import INDEX_DIR

INDEX_PATH = Path(INDEX_DIR)


# ---------------------------------------------------------------------------
# Load index artifacts
# ---------------------------------------------------------------------------

def load_artifacts():
    faiss_index = faiss.read_index(str(INDEX_PATH / "faiss.index"))
    
    with open(INDEX_PATH / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(INDEX_PATH / "chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Reconstruct the raw embedding matrix from the FAISS index
    n = faiss_index.ntotal
    dim = faiss_index.d
    embeddings = np.zeros((n, dim), dtype=np.float32)
    faiss_index.reconstruct_n(0, n, embeddings)

    return embeddings, bm25, chunks


# ---------------------------------------------------------------------------
# Plot 1: UMAP embedding projection
# ---------------------------------------------------------------------------

def plot_embeddings(embeddings: np.ndarray, chunks: list[dict]) -> None:
    print("Fitting UMAP projection (this takes a few seconds)...")

    reducer = umap.UMAP(
        n_neighbors=min(15, len(chunks) - 1),
        min_dist=0.1,
        random_state=42,
    )
    projected = reducer.fit_transform(embeddings)

    # Colour points by page number so you can see document progression
    pages = [c["page_start"] for c in chunks]
    colours = cm.viridis(np.array(pages) / max(pages))

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=pages,
        cmap="viridis",
        s=60,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.4,
    )

    # Annotate each point with its chunk ID and page
    for i, chunk in enumerate(chunks):
        ax.annotate(
            f"#{chunk['chunk_id']} p{chunk['page_start']}",
            (projected[i, 0], projected[i, 1]),
            fontsize=6,
            alpha=0.6,
            xytext=(4, 4),
            textcoords="offset points",
        )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Page number", fontsize=10)

    ax.set_title(
        f"Chunk embeddings — UMAP projection ({len(chunks)} chunks)",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = Path("umap_embeddings.png")
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2: BM25 score distribution for a query
# ---------------------------------------------------------------------------

def plot_bm25(bm25, chunks: list[dict], query: str) -> None:
    tokenised = query.lower().split()
    scores = bm25.get_scores(tokenised)

    # Sort by score descending
    order = np.argsort(scores)[::-1]
    top_n = min(20, len(chunks))
    top_indices = order[:top_n]
    top_scores = scores[top_indices]

    # Labels: chunk ID + first 40 chars of text
    labels = [
        f"#{chunks[i]['chunk_id']} — {chunks[i]['text'][:40].strip()}..."
        for i in top_indices
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        range(top_n),
        top_scores[::-1],   # reverse so highest score is at top
        color=cm.Blues(np.linspace(0.4, 0.85, top_n)),
        edgecolor="white",
        height=0.7,
    )

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=7)
    ax.set_xlabel("BM25 score")
    ax.set_title(
        f'BM25 scores — top {top_n} chunks\nQuery: "{query}"',
        fontsize=12,
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = Path("bm25_scores.png")
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def print_summary(chunks: list[dict], embeddings: np.ndarray) -> None:
    token_counts = [c["token_count"] for c in chunks]
    pages = [c["page_start"] for c in chunks]

    print("\n── Index summary ──────────────────────────────")
    print(f"  Total chunks     : {len(chunks)}")
    print(f"  Embedding dim    : {embeddings.shape[1]}")
    print(f"  Pages covered    : {min(pages)} – {max(pages)}")
    print(f"  Tokens per chunk : min={min(token_counts)}  "
          f"max={max(token_counts)}  "
          f"avg={int(sum(token_counts)/len(token_counts))}")

    # Show chunks with detected headings
    headed = [c for c in chunks if c.get("section_heading")]
    print(f"  Chunks with detected headings: {len(headed)}")
    if headed:
        for c in headed[:5]:
            print(f"    • Chunk #{c['chunk_id']} (p{c['page_start']}): "
                  f"{c['section_heading']}")
        if len(headed) > 5:
            print(f"    ... and {len(headed) - 5} more")
    print("───────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "main findings and results"

    if not (INDEX_PATH / "faiss.index").exists():
        print(f"No index found at {INDEX_PATH}/ — run ingest.py first.")
        sys.exit(1)

    print("Loading index artifacts...")
    embeddings, bm25, chunks = load_artifacts()

    print_summary(chunks, embeddings)
    plot_embeddings(embeddings, chunks)
    plot_bm25(bm25, chunks, query)
