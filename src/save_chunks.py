"""
Save the top 5 retrieved chunks for a query to a text file.

Useful for manually inspecting whether the pipeline is retrieving
the right content before generation happens.

Usage:
    python save_chunks.py "your question here"

Output:
    chunks_output.txt in the current directory
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from retrieval import Retriever


def save_chunks(query: str, output_path: str = "chunks_output.txt") -> None:
    retriever = Retriever()
    results = retriever.retrieve(query)

    lines = []
    lines.append("=" * 70)
    lines.append(f"Query      : {query}")
    lines.append(f"Timestamp  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Chunks     : {len(results)}")
    lines.append("=" * 70)
    lines.append("")

    for i, rc in enumerate(results, 1):
        c = rc.chunk
        location = (
            f"Page {c.page_start}"
            if c.page_start == c.page_end
            else f"Pages {c.page_start}–{c.page_end}"
        )
        heading = f" | {c.section_heading}" if c.section_heading else ""

        lines.append(f"CHUNK [{i} of {len(results)}]")
        lines.append(f"  Chunk ID     : {c.chunk_id}")
        lines.append(f"  Location     : {location}{heading}")
        lines.append(f"  Rerank score : {rc.rerank_score:.4f}")
        lines.append(f"  RRF score    : {rc.rrf_score:.4f}")
        lines.append(f"  Token count  : {c.token_count}")
        lines.append("")
        lines.append("  Full text:")
        lines.append("  " + "-" * 50)
        # Indent each line of the chunk text for readability
        for text_line in c.text.splitlines():
            lines.append(f"  {text_line}")
        lines.append("")
        lines.append("=" * 70)
        lines.append("")

    output = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Saved {len(results)} chunks to {output_path}")
    print()
    for i, rc in enumerate(results, 1):
        print(f"  [{i}] Chunk #{rc.chunk.chunk_id} | "
              f"Page {rc.chunk.page_start} | "
              f"Rerank: {rc.rerank_score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python save_chunks.py \"your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    save_chunks(query)