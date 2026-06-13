"""Tests for structure-aware document chunking."""

from __future__ import annotations

from regradar.chunking import chunk_document

_SAMPLE_DOC = """Environmental Protection Agency
40 CFR Part 60

SUMMARY:
The EPA proposes to amend emissions standards for stationary sources.

DATES:
Comments must be received on or before August 1, 2026.

ADDRESSES:
Submit comments to the Federal eRulemaking Portal.

SUPPLEMENTARY INFORMATION:
Background paragraph one about the regulatory history.

Background paragraph two about the statutory authority.
"""


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_document("") == []
    assert chunk_document("   \n  ") == []


def test_sections_are_detected() -> None:
    chunks = chunk_document(_SAMPLE_DOC)
    sections = {c.section for c in chunks}
    assert "SUMMARY" in sections
    assert "DATES" in sections
    assert "ADDRESSES" in sections
    assert "SUPPLEMENTARY INFORMATION" in sections


def test_preamble_has_no_section() -> None:
    chunks = chunk_document(_SAMPLE_DOC)
    first = chunks[0]
    assert first.section is None
    assert "Environmental Protection Agency" in first.content


def test_chunk_indices_are_sequential() -> None:
    chunks = chunk_document(_SAMPLE_DOC)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_short_sections_stay_whole() -> None:
    chunks = chunk_document(_SAMPLE_DOC)
    dates_chunks = [c for c in chunks if c.section == "DATES"]
    assert len(dates_chunks) == 1
    assert "August 1, 2026" in dates_chunks[0].content


def test_no_content_is_lost() -> None:
    """Every meaningful line of the input appears in some chunk."""
    chunks = chunk_document(_SAMPLE_DOC)
    combined = "\n".join(c.content for c in chunks)
    for line in _SAMPLE_DOC.splitlines():
        stripped = line.strip().rstrip(":")
        if stripped and stripped not in {
            "SUMMARY",
            "DATES",
            "ADDRESSES",
            "SUPPLEMENTARY INFORMATION",
        }:
            assert stripped in combined, f"Lost line: {stripped}"


def test_long_section_is_split_with_overlap() -> None:
    """A section exceeding the target splits into multiple chunks."""
    long_para = "This is a sentence about pipeline regulation. " * 30
    paragraphs = [f"Paragraph {i}. {long_para}" for i in range(10)]
    doc = "SUPPLEMENTARY INFORMATION:\n" + "\n\n".join(paragraphs)

    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(c.section == "SUPPLEMENTARY INFORMATION" for c in chunks)
