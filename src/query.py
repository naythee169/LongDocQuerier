"""
Query entrypoint.

Loads the pre-built index, retrieves relevant chunks, and generates a grounded answer.

Usage:
    # Interactive mode
    python query.py

    # Single question
    python query.py "What was the total revenue in fiscal year 2023?"

    # With retrieval scores
    python query.py "What were the main risk factors? --verbose"

"""

from __future__ import annotations

import sys

from generation import generate_answer
from retrieval import Retriever


def ask(retriever: Retriever, question: str, verbose: bool = False) -> None:
    """Retrieve and answer a single question, printing the result."""
    print(f"\nSearching for: {question!r}")

    retrieved = retriever.retrieve(question)

    if verbose:
        print(f"\nTop {len(retrieved)} retrieved chunks (before generation):")
        for i, rc in enumerate(retrieved, 1):
            print(
                f"  [{i}] chunk_id={rc.chunk.chunk_id} "
                f"page={rc.chunk.page_start} "
                f"rerank={rc.rerank_score:.3f} "
                f"rrf={rc.rrf_score:.4f}"
            )

    answer = generate_answer(question, retrieved)
    print("\n" + "=" * 60)
    print(answer.display())
    print("=" * 60)


def main() -> None:
    retriever = Retriever()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        verbose = "--verbose" in sys.argv
        question = question.replace("--verbose", "").strip()
        ask(retriever, question, verbose=verbose)
    else:
        print("\nDocument Q&A system ready. Type 'quit' to exit.")
        print("Add --verbose after your question for retrieval scores.\n")
        while True:
            try:
                user_input = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                break

            verbose = user_input.endswith("--verbose")
            question = user_input.removesuffix("--verbose").strip()
            ask(retriever, question, verbose=verbose)


if __name__ == "__main__":
    main()
