"""
Answer generation module.

Uses the Groq API (free tier) to generate grounded answers from retrieved chunks.

The LLM is used purely as a reasoning engine over the provided excerpts —
it is explicitly instructed not to use its training knowledge.

Requires:
    GROQ_API_KEY environment variable
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from groq import Groq

from config import GROQ_MODEL, MAX_ANSWER_TOKENS, RERANK_TOP_K
from retrieval import RetrievedChunk

# If the best rerank score is below this threshold, none of the retrieved
# chunks are likely relevant — abstain rather than risk hallucination.
ABSTAIN_SCORE_THRESHOLD = 0.1
load_dotenv()

# Data model

@dataclass
class Answer:
    """The system's response to a user question."""
    question: str
    answer_text: str
    sources: list[RetrievedChunk]
    abstained: bool = False

    def display(self) -> str:
        lines = [
            f"Q: {self.question}",
            "",
            f"A: {self.answer_text}",
            "",
            "--- Sources ---",
        ]
        for rc in self.sources:
            c = rc.chunk
            location = (
                f"Page {c.page_start}"
                if c.page_start == c.page_end
                else f"Pages {c.page_start}–{c.page_end}"
            )
            heading = f" | {c.section_heading}" if c.section_heading else ""
            lines.append(
                f"[Excerpt {c.chunk_id} | {location}{heading}] "
                f"(rerank score: {rc.rerank_score:.3f})"
            )
            lines.append(f"  {c.text[:200]}{'...' if len(c.text) > 200 else ''}")
            lines.append("")
        return "\n".join(lines)


# Prompt construction

_SYSTEM_PROMPT = """You are a precise document question-answering assistant.

Your task is to answer questions based SOLELY on the provided document excerpts.

Rules:
1. Answer only using information explicitly present in the excerpts below.
2. Do NOT use your general training knowledge. If the excerpts do not contain
   the answer, say exactly: "The document does not contain sufficient information
   to answer this question."
3. Cite your sources inline using the excerpt number in square brackets, e.g. [1]
   or [2][3] for multiple sources.
4. For numerical answers, quote the exact figure from the excerpt.
5. Keep your answer concise and direct. Do not pad with general commentary.
6. If the question asks for something that partially appears across multiple
   excerpts, synthesise them and cite each one.
"""

def _build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    excerpt_lines = []
    for i, rc in enumerate(chunks, start=1):
        excerpt_lines.append(
            rc.chunk.as_context_string().replace(
                f"[Excerpt {rc.chunk.chunk_id}", f"[{i}"
            )
        )
        excerpt_lines.append("")

    return f"""Document excerpts:

{"".join(excerpt_lines)}
---
Question: {query}

Answer (cite excerpts using [1], [2], etc.):"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(query: str, retrieved: list[RetrievedChunk]) -> Answer:
    """
    Generate a grounded answer from retrieved chunks using Groq.

    Args:
        query: The user's natural language question.
        retrieved: Ordered list of RetrievedChunk objects from the retrieval stage.

    Returns:
        An Answer object containing the response text and source metadata.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable not set.\n"
            "Get a free key at https://console.groq.com — no credit card required."
        )

    # Early abstention: if the best rerank score is very low, the retrieved
    # chunks are probably not relevant — skip the LLM call entirely.
    if not retrieved or retrieved[0].rerank_score < ABSTAIN_SCORE_THRESHOLD:
        return Answer(
            question=query,
            answer_text=(
                "The document does not contain sufficient information "
                "to answer this question."
            ),
            sources=retrieved,
            abstained=True,
        )

    client = Groq(api_key=api_key)
    user_prompt = _build_user_prompt(query, retrieved)

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=MAX_ANSWER_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    answer_text = completion.choices[0].message.content.strip()

    return Answer(
        question=query,
        answer_text=answer_text,
        sources=retrieved,
        abstained=False,
    )
