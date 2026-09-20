"""Two-phase chunking with exact page attribution.

Phase 1 (structural): ``MarkdownHeaderTextSplitter`` walks the markdown
produced by pymupdf4llm top to bottom and emits one fragment per header,
recording the header hierarchy as metadata. It never cuts by size, so a
fragment is a complete logical section (e.g. all of "Materials and Methods").

Phase 2 (recursive): ``RecursiveCharacterTextSplitter`` resizes those
sections to ~``chunk_size`` characters so embeddings stay focused on one
dense paragraph/table block and inside the encoder's sweet spot. The
overlap keeps formulas and sentence-bound conclusions from being severed.

Page attribution: pages are concatenated in order and every final chunk is
located inside that concatenation. The splitters normalise whitespace, so
matching happens on a whitespace-collapsed copy of the text with an
index map back to the original offsets.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.ingestion.loader import PageMarkdown

logger = logging.getLogger(__name__)

# Header levels emitted by pymupdf4llm's font-size heuristic
_HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]
_HEADER_KEYS = [name for _, name in _HEADERS_TO_SPLIT_ON]

_WHITESPACE = re.compile(r"\s+")
# MDPI headings arrive wrapped in emphasis (**1. Introduction**) and pymupdf
# titles can carry HTML-ish tags (<sup>1</sup>): strip both so citation
# metadata reads "1. Introduction" instead of literal markup.
_EMPHASIS = re.compile(r"[*_`]+")
_TAGS = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    content: str
    source: str
    title: str
    page_start: int
    page_end: int
    section: str
    chunk_index: int

    @property
    def pages(self) -> list[int]:
        return list(range(self.page_start, self.page_end + 1))


def chunk_document(
    pages: list[PageMarkdown],
    *,
    source: str,
    title: str,
    chunk_size: int = 750,
    chunk_overlap: int = 140,
) -> list[Chunk]:
    """Split one loaded PDF into citation-ready chunks."""
    full_text, page_spans = _concat_pages(pages)
    # Normalised copy + index map: all chunk locating happens in this domain.
    norm_full, index_map = _normalise_with_map(full_text)

    structural = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT_ON, strip_headers=True
    ).split_text(full_text)
    resizer = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks: list[Chunk] = []
    section_cursor = 0  # forward-only search keeps repeated passages from matching earlier copies
    for section_doc in structural:
        section_text = section_doc.page_content
        if not section_text.strip():
            continue
        norm_section = _normalise(section_text)
        section_start = norm_full.find(norm_section, section_cursor)
        if section_start == -1:
            # Should not happen (validated on the corpus); degrade gracefully
            # by leaving the chunk without page attribution.
            logger.warning("Page mapping failed for a section of %s", source)
        else:
            section_cursor = section_start + 1

        piece_cursor = 0
        for piece in resizer.split_text(section_text):
            local = norm_section.find(_normalise(piece), piece_cursor)
            if local != -1:
                piece_cursor = local + 1

            if section_start != -1 and local != -1:
                absolute = section_start + local
                piece_pages = _pages_for_span(
                    page_spans, index_map[absolute], index_map[absolute + len(_normalise(piece)) - 1] + 1
                )
            else:
                piece_pages = []
            if not piece_pages:
                piece_pages = [pages[0].number]
                logger.warning("Chunk without page span in %s; defaulting to page %d", source, piece_pages[0])

            page_start, page_end = min(piece_pages), max(piece_pages)
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(source, page_start, page_end, len(chunks) + 1),
                    content=piece,
                    source=source,
                    title=title,
                    page_start=page_start,
                    page_end=page_end,
                    section=_section_path(section_doc.metadata),
                    chunk_index=len(chunks) + 1,
                )
            )
    return chunks


def _concat_pages(pages: list[PageMarkdown]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join page texts in order and record each page's (page, start, end) span."""
    pieces: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for position, page in enumerate(pages):
        if position:
            pieces.append("\n\n")
            cursor += 2
        spans.append((page.number, cursor, cursor + len(page.text)))
        pieces.append(page.text)
        cursor += len(page.text)
    return "".join(pieces), spans


def _normalise(text: str) -> str:
    """Collapse every whitespace run to a single space and trim the ends."""
    return _WHITESPACE.sub(" ", text).strip()


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed text plus a map from collapsed index -> original index."""
    chars: list[str] = []
    mapping: list[int] = []
    previous_was_space = True  # leading whitespace is dropped, like in _normalise
    for index, char in enumerate(text):
        if char.isspace():
            if not previous_was_space:
                chars.append(" ")
                mapping.append(index)
            previous_was_space = True
        else:
            chars.append(char)
            mapping.append(index)
            previous_was_space = False
    return "".join(chars), mapping


def _pages_for_span(spans: list[tuple[int, int, int]], start: int, end: int) -> list[int]:
    """Pages whose text range intersects the [start, end) character span."""
    return [page for page, span_start, span_end in spans if start < span_end and end > span_start]


def _section_path(metadata: dict) -> str:
    """Flatten the header hierarchy into 'Results > Vegetation Indices'.

    Header 1 (the article title, as rendered by pymupdf4llm's font-size
    heuristic) is deliberately skipped: the title already lives in its own
    metadata field, and repeating it in every section path would only add
    noise to citations and to the embedded context prefix.
    """
    parts = (_clean_header(metadata.get(key)) for key in _HEADER_KEYS[1:])
    return " > ".join(part for part in parts if part)


def _clean_header(value: object) -> str:
    return _WHITESPACE.sub(" ", _EMPHASIS.sub("", _TAGS.sub("", str(value or "")))).strip()


def _chunk_id(source: str, page_start: int, page_end: int, index: int) -> str:
    pages_part = f"p{page_start}" if page_start == page_end else f"p{page_start}-{page_end}"
    return f"{source}_{pages_part}_c{index}"
