"""
Document parsing and chunking.

Responsibilities:
- Extract text from PDF page by page using pdfplumber
- Preserve page numbers and detected section headings as metadata
- Split text into overlapping chunks that respect paragraph boundaries where possible
- Return a list of Chunk objects ready for embedding
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pdfplumber

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Data model

@dataclass
class Chunk:

    text: str
    chunk_id: int
    page_start: int
    page_end: int
    section_heading: Optional[str] = None
    token_count: int = 0

    def as_context_string(self) -> str:

        location = (
            f"Page {self.page_start}"
            if self.page_start == self.page_end
            else f"Pages {self.page_start}–{self.page_end}"
        )
        heading = f" | {self.section_heading}" if self.section_heading else ""
        return f"[Excerpt {self.chunk_id} | {location}{heading}]\n{self.text}"


# Heading detection

_HEADING_PATTERNS = [
    re.compile(r"^[A-Z][A-Z\s\d\-&,:]{4,60}$"),
    re.compile(r"^(?:[A-Z][a-z]+\s){1,8}[A-Z][a-z]+$"),
    re.compile(
        r"^(?:ITEM\s+\d+[A-Z]?|NOTE\s+\d+|PART\s+[IVX]+)",
        re.IGNORECASE,
    ),
]

def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line.split()) > 12:
        return False
    return any(p.match(line) for p in _HEADING_PATTERNS)


# Per-page extraction

@dataclass
class _PageContent:
    page_number: int
    text: str


def _extract_pages(pdf_path: Path) -> list[_PageContent]:
    """
    Extract text from every page of the PDF.
    Tables are extracted as pipe-delimited rows to preserve numeric content.
    """
    pages: list[_PageContent] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""

            for table in page.extract_tables():
                rows = []
                for row in table:
                    cleaned = [cell.strip() if cell else "" for cell in row]
                    rows.append(" | ".join(cleaned))
                text += "\n" + "\n".join(rows)

            pages.append(_PageContent(page_number=i, text=text.strip()))

    return pages


# Chunking

def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text)
    if len(paragraphs) == 1:
        paragraphs = text.split("\n")
    return [p.strip() for p in paragraphs if p.strip()]


def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(text.split())


def _build_chunks(pages: list[_PageContent]) -> list[Chunk]:
    """
    Slide a window over the document's paragraphs to build overlapping chunks.
    Paragraph boundaries are respected — we never split mid-paragraph.
    """
    chunks: list[Chunk] = []

    units: list[tuple[str, int, bool]] = []
    for page in pages:
        for para in _split_into_paragraphs(page.text):
            units.append((para, page.page_number, _is_heading(para)))

    current_texts: list[str] = []
    current_pages: list[int] = []
    current_tokens = 0
    current_heading: Optional[str] = None
    chunk_id = 0

    i = 0
    while i < len(units):
        para, page_num, is_heading = units[i]

        if is_heading:
            current_heading = para

        para_tokens = _count_tokens(para)

        if current_tokens + para_tokens > CHUNK_SIZE and current_texts:
            chunk_text = "\n\n".join(current_texts)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                page_start=current_pages[0],
                page_end=current_pages[-1],
                section_heading=current_heading,
                token_count=current_tokens,
            ))
            chunk_id += 1

            # Backtrack for overlap
            overlap_tokens = 0
            overlap_texts: list[str] = []
            overlap_pages: list[int] = []
            for t, p in zip(reversed(current_texts), reversed(current_pages)):
                t_count = _count_tokens(t)
                if overlap_tokens + t_count > CHUNK_OVERLAP:
                    break
                overlap_tokens += t_count
                overlap_texts.insert(0, t)
                overlap_pages.insert(0, p)

            current_texts = overlap_texts
            current_pages = overlap_pages
            current_tokens = overlap_tokens
            continue

        current_texts.append(para)
        current_pages.append(page_num)
        current_tokens += para_tokens
        i += 1

    if current_texts:
        chunks.append(Chunk(
            text="\n\n".join(current_texts),
            chunk_id=chunk_id,
            page_start=current_pages[0],
            page_end=current_pages[-1],
            section_heading=current_heading,
            token_count=current_tokens,
        ))

    return chunks


def parse_document(pdf_path: str | Path) -> list[Chunk]:
    """
    Parse a PDF and return a list of overlapping Chunk objects.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Ordered list of Chunk objects covering the entire document.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Parsing {pdf_path.name}...")
    pages = _extract_pages(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    chunks = _build_chunks(pages)
    print(f"  Built {len(chunks)} chunks "
          f"(avg {sum(c.token_count for c in chunks) // len(chunks)} tokens each)")

    return chunks
