"""Structure-aware chunking for Federal Register documents.

Federal Register documents follow a canonical structure with labeled
sections (SUMMARY, DATES, ADDRESSES, SUPPLEMENTARY INFORMATION, ...).
This module splits documents along those section boundaries first, then
chunks long sections by token count while respecting paragraph
boundaries. Every chunk carries its section name as metadata.

This two-level strategy keeps each chunk semantically coherent (about
one thing), which directly improves embedding and retrieval quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

# Canonical Federal Register section headers, in the order they
# typically appear. Matched case-sensitively at line starts.
_SECTION_HEADERS = [
    "SUMMARY",
    "DATES",
    "ADDRESSES",
    "FOR FURTHER INFORMATION CONTACT",
    "SUPPLEMENTARY INFORMATION",
]

# Chunking parameters. ~1000 tokens holds a complete regulatory thought;
# ~150 token overlap preserves continuity across chunk boundaries.
_TARGET_CHUNK_TOKENS = 1000
_OVERLAP_TOKENS = 150

# text-embedding-3-small uses the cl100k_base encoding.
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class Chunk:
    """A single chunk of document text ready for embedding."""

    content: str
    section: str | None
    chunk_index: int


def _count_tokens(text: str) -> int:
    """Count tokens the way the embedding model does."""
    return len(_ENCODING.encode(text))


def _split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split document text into (section_name, section_text) pairs.

    Detects canonical FR section headers at line starts. Text before the
    first recognized header gets section name None.
    """
    # Build a regex matching any canonical header at a line start,
    # optionally followed by a colon.
    pattern = re.compile(
        r"^(" + "|".join(re.escape(h) for h in _SECTION_HEADERS) + r"):?\s*$",
        re.MULTILINE,
    )

    sections: list[tuple[str | None, str]] = []
    last_end = 0
    last_header: str | None = None

    for match in pattern.finditer(text):
        body = text[last_end : match.start()].strip()
        if body:
            sections.append((last_header, body))
        last_header = match.group(1)
        last_end = match.end()

    # Remaining text after the final header.
    tail = text[last_end:].strip()
    if tail:
        sections.append((last_header, tail))

    return sections


def _split_section_by_tokens(text: str) -> list[str]:
    """Split one section's text into token-bounded pieces.

    Splits at paragraph boundaries where possible, packing paragraphs
    into chunks up to the target size, with token overlap carried
    between consecutive chunks.
    """
    if _count_tokens(text) <= _TARGET_CHUNK_TOKENS:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _count_tokens(para)

        # A single paragraph larger than the target gets hard-split by tokens.
        if para_tokens > _TARGET_CHUNK_TOKENS:
            if current:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0
            token_ids = _ENCODING.encode(para)
            step = _TARGET_CHUNK_TOKENS - _OVERLAP_TOKENS
            for start in range(0, len(token_ids), step):
                window = token_ids[start : start + _TARGET_CHUNK_TOKENS]
                pieces.append(_ENCODING.decode(window))
            continue

        if current_tokens + para_tokens > _TARGET_CHUNK_TOKENS and current:
            pieces.append("\n\n".join(current))
            # Carry overlap: keep trailing paragraphs up to the overlap budget.
            overlap: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current):
                prev_tokens = _count_tokens(prev)
                if overlap_tokens + prev_tokens > _OVERLAP_TOKENS:
                    break
                overlap.insert(0, prev)
                overlap_tokens += prev_tokens
            current = overlap
            current_tokens = overlap_tokens

        current.append(para)
        current_tokens += para_tokens

    if current:
        pieces.append("\n\n".join(current))

    return pieces


def chunk_document(text: str) -> list[Chunk]:
    """Chunk a Federal Register document's full text.

    Args:
        text: The document's plain text.

    Returns:
        Ordered chunks, each tagged with the section it came from and
        its position in the document.
    """
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    index = 0

    for section_name, section_text in _split_into_sections(text):
        for piece in _split_section_by_tokens(section_text):
            chunks.append(Chunk(content=piece, section=section_name, chunk_index=index))
            index += 1

    return chunks
