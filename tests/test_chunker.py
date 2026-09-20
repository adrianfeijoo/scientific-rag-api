"""Chunker unit tests on synthetic markdown (no model, no network, no store)."""

import re

from rag.ingestion.chunker import chunk_document
from rag.ingestion.loader import PageMarkdown

CHUNK_SIZE = 750
CHUNK_OVERLAP = 140
SOURCE = "synthetic-study.pdf"
TITLE = "A synthetic study"

PAGES = [
    PageMarkdown(
        number=1,
        text="# Spectral detection of powdery mildew in wheat\n\n"
        "Study site: replicated field plots in Zaragoza, Spain.",
    ),
    PageMarkdown(
        number=2,
        text="## Materials and Methods\n\n"
        "Plots were surveyed weekly to record disease severity on the flag leaf.\n\n"
        "Spectral measurements used a portable spectroradiometer covering 350-2500 nm.",
    ),
    PageMarkdown(
        number=3,
        text="Leaf samples were oven-dried to determine water content gravimetrically.\n\n"
        "## Results\n\n"
        "PRI tracked chlorophyll content with R2 = 0.639.\n\n"
        "WBI showed weaker correlations with plant water content.",
    ),
]


def make_chunks(pages=None):
    return chunk_document(
        pages if pages is not None else PAGES,
        source=SOURCE,
        title=TITLE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def test_chunks_carry_basic_metadata():
    chunks = make_chunks()
    assert chunks
    for chunk in chunks:
        assert chunk.source == SOURCE
        assert chunk.title == TITLE
        assert chunk.content.strip()
        assert 1 <= chunk.page_start <= chunk.page_end <= 3
        assert re.fullmatch(rf"{re.escape(SOURCE)}_p\d+(-\d+)?_c\d+", chunk.chunk_id)


def test_section_metadata_follows_header_hierarchy():
    chunks = make_chunks()
    sections = {chunk.section for chunk in chunks}
    # Header 1 is the article title; deeper sections are nested under it.
    assert any("Materials and Methods" in section for section in sections)
    assert any(section.endswith("Results") for section in sections)


def test_merged_paragraphs_can_span_pages():
    # RecursiveCharacterTextSplitter merges adjacent small pieces, so the two
    # short Materials-and-Methods paragraphs (end of page 2, start of page 3)
    # must end up in a single chunk attributed to both pages.
    chunks = make_chunks()
    cross_page = [c for c in chunks if c.pages == [2, 3]]
    assert cross_page, "expected at least one chunk spanning pages 2 and 3"
    assert all("Materials and Methods" in c.section for c in cross_page)


def test_long_section_is_resized_within_chunk_size():
    long_page = PageMarkdown(
        number=2,
        text="## Long Section\n\n"
        + "Reflectance spectra were acquired for every plot and calibrated. " * 40,
    )
    chunks = chunk_document(
        [PageMarkdown(number=1, text="# Only title\n\nintro"), long_page],
        source=SOURCE,
        title=TITLE,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    sized = [c for c in chunks if "Reflectance" in c.content]
    assert len(sized) >= 2  # the oversized section was actually split
    assert all(len(c.content) <= CHUNK_SIZE for c in sized)


def test_document_without_headers_still_chunks():
    pages = [PageMarkdown(number=1, text="Plain text only. " * 60)]
    chunks = make_chunks(pages)
    assert chunks
    assert all(chunk.section == "" for chunk in chunks)


def test_chunk_ids_are_unique_and_sequential():
    chunks = make_chunks()
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(set(ids)) == len(ids)
    assert [chunk.chunk_index for chunk in chunks] == list(range(1, len(chunks) + 1))
