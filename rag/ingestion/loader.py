"""PDF loading: per-page markdown extraction that preserves page boundaries.

pymupdf4llm converts each page to markdown (font-size based `#`/`##`
headings included), which allows structural chunking.
Keeping one markdown document per page (instead of whole-document
markdown) allows citations to report exact page ranges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm


@dataclass(frozen=True)
class PageMarkdown:
    """Markdown text of a single PDF page (1-based page number)."""

    number: int
    text: str


@dataclass(frozen=True)
class LoadedPdf:
    source: str  # PDF filename; doubles as the citation identifier
    title: str
    pages: list[PageMarkdown]


def load_pdf(pdf_path: Path) -> LoadedPdf:
    """Extract one markdown document per page, in page order.

    With ``page_chunks=True`` pymupdf4llm returns exactly one dict per page
    (in order), each carrying its 1-based ``page_number`` in the embedded
    metadata. Images are ignored on purpose: writing image files to disk as
    a side effect of ingestion would be surprising and useless for RAG.
    """
    raw_pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        ignore_images=True,
        show_progress=False,
    )

    pages: list[PageMarkdown] = []
    for position, raw in enumerate(raw_pages, start=1):
        text = (raw.get("text") or "").strip()
        if not text:
            continue
        number = int((raw.get("metadata") or {}).get("page_number") or position)
        pages.append(PageMarkdown(number=number, text=text))

    if not pages:
        raise ValueError(
            f"No extractable text found in {pdf_path.name} (scanned PDF without OCR?)"
        )
    return LoadedPdf(
        source=pdf_path.name,
        title=_extract_title(pdf_path, pages),
        pages=pages,
    )


def _extract_title(pdf_path: Path, pages: list[PageMarkdown]) -> str:
    """Best-effort article title: PDF metadata, then first heading of page 1.
    """
    try:
        with pymupdf.open(str(pdf_path)) as document:
            title = str((document.metadata or {}).get("title") or "").strip()
    except Exception:  # noqa: BLE001 - cosmetic field, never fatal
        title = ""
    if title:
        return title
    for line in pages[0].text.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned:
            return cleaned
    return pdf_path.stem
