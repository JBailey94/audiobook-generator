"""
File processing utilities for PDF, EPUB, Markdown and TXT files.
Ported from ebook2audiobook's text extraction pipeline.

Two levels of extraction are provided:

- :func:`extract_text` returns one big blob of text for a whole file.
- :func:`extract_sections` splits a file into titled sections (chapters) so a
  book can be rendered as a set of playable chapter files instead of a single
  multi-hour WAV.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

# Roughly how much text goes into a section when the source has no usable
# structure (plain text, PDFs without a table of contents). ~3000 characters is
# about 8-10 minutes of narration, which is a comfortable chapter length.
DEFAULT_SECTION_CHARS = 3000

# Sections shorter than this are dropped, unless they look like real speech.
# EPUBs often contain spine items that hold nothing but a cover image or a
# download-site watermark, and reading "OceanofPDF.com" aloud is worse than
# skipping it.
MIN_SECTION_CHARS = 120

_SENTENCE_END = re.compile(r"[.!?…][\"')\]]?$")


def _is_artifact(text: str, min_chars: int) -> bool:
    """True if a section is too thin to be worth narrating."""
    stripped = text.strip()
    if len(stripped) >= min_chars:
        return False
    # A genuine (if short) section ends in sentence punctuation. Watermarks
    # and image-only pages do not, so those get dropped instead.
    return not _SENTENCE_END.search(stripped)


@dataclass
class Section:
    """A titled chunk of a book, used to produce one audio file per chapter."""

    title: str
    text: str


# ---------------------------------------------------------------------------
# Flat text extraction
# ---------------------------------------------------------------------------


def _html_to_text(soup) -> str:
    """Strip a parsed HTML document down to readable text."""
    for tag in soup(["script", "style", "nav"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        import pymupdf  # PyMuPDF >= 1.24
    except ImportError:
        import fitz as pymupdf  # older PyMuPDF releases

    doc = pymupdf.open(pdf_path)
    text_parts = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            text_parts.append(text.strip())
    doc.close()
    return "\n\n".join(text_parts)


def extract_text_from_epub(epub_path: str) -> str:
    """Extract text from an EPUB file using ebooklib and BeautifulSoup."""
    sections = _epub_sections(epub_path)
    if sections:
        return "\n\n".join(s.text for s in sections)
    # Fall back to the whole-document scan for unusual EPUB packaging.
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    book = epub.read_epub(epub_path)
    text_parts = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content().decode("utf-8", errors="ignore"), "html.parser")
            text = _html_to_text(soup)
            if text:
                text_parts.append(text)
    return "\n\n".join(text_parts)


def extract_text_from_markdown(md_path: str) -> str:
    """Extract text from a Markdown file."""
    raw = Path(md_path).read_text(encoding="utf-8")

    # Strip code blocks
    raw = re.sub(r"```[\s\S]*?```", "", raw)
    raw = re.sub(r"`[^`]*`", "", raw)
    # Strip HTML tags
    raw = re.sub(r"<[^>]+>", "", raw)
    # Strip images
    raw = re.sub(r"!\[.*?\]\(.*?\)", "", raw)
    # Strip links but keep text
    raw = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", raw)
    # Strip emphasis markers
    raw = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", raw)
    # Strip headers markers
    raw = re.sub(r"^#{1,6}\s+", "", raw, flags=re.MULTILINE)
    # Strip blockquotes
    raw = re.sub(r"^>\s?", "", raw, flags=re.MULTILINE)
    # Strip horizontal rules
    raw = re.sub(r"^[-*_]{3,}\s*$", "", raw, flags=re.MULTILINE)

    return raw.strip()


def extract_text_from_txt(txt_path: str) -> str:
    """Extract text from a plain text file, tolerating odd encodings."""
    data = Path(txt_path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".epub":
        return extract_text_from_epub(file_path)
    elif ext in (".md", ".markdown"):
        return extract_text_from_markdown(file_path)
    elif ext in (".txt", ".text"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------------------------
# Section (chapter) extraction
# ---------------------------------------------------------------------------


def _clean_title(raw: str) -> str:
    """Normalise a heading into something usable as a filename."""
    title = re.sub(r"\s+", " ", raw or "").strip()
    title = title.strip("#*_` \t")
    return title[:120] if title else ""


def _href_key(href: str) -> str:
    """Normalise an EPUB href so spine files and TOC links can be matched."""
    if not href:
        return ""
    path = urlsplit(href).path
    return unquote(Path(path).name).lower()


def _toc_titles(book) -> dict[str, str]:
    """Collect ``{file name: chapter title}`` from the EPUB table of contents."""
    titles: dict[str, str] = {}

    def walk(entries) -> None:
        for entry in entries or []:
            if isinstance(entry, (list, tuple)):
                if len(entry) >= 2 and isinstance(entry[0], str) and isinstance(entry[1], str):
                    key = _href_key(entry[1])
                    title = _clean_title(entry[0])
                    if key and title:
                        titles.setdefault(key, title)
                else:
                    walk(entry)

    try:
        walk(book.toc)
    except Exception:  # pragma: no cover - malformed TOC should not be fatal
        pass
    return titles


def _epub_spine_items(book) -> list:
    """Return spine items in reading order."""
    import ebooklib

    items = []
    for entry in getattr(book, "spine", []) or []:
        idref = entry[0] if isinstance(entry, (list, tuple)) else entry
        try:
            item = book.get_item_with_id(idref)
        except Exception:
            item = None
        if item is not None and item.get_type() == ebooklib.ITEM_DOCUMENT:
            items.append(item)
    return items


def _epub_sections(epub_path: str) -> list[Section]:
    """Split an EPUB into chapters, titled from the table of contents."""
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    try:
        book = epub.read_epub(epub_path)
    except Exception:
        return []

    toc = _toc_titles(book)
    items = _epub_spine_items(book)
    if not items:
        return []

    sections: list[Section] = []
    for item in items:
        content = item.get_content()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        text = _html_to_text(soup)
        if not text:
            continue

        title = toc.get(_href_key(getattr(item, "file_name", "")))
        if not title:
            for level in ("h1", "h2", "h3"):
                heading = soup.find(level)
                if heading and heading.get_text(strip=True):
                    title = _clean_title(heading.get_text(strip=True))
                    break
        if not title and soup.title and soup.title.string:
            title = _clean_title(soup.title.string)

        sections.append(Section(title=title or f"Section {len(sections) + 1}", text=text))

    return sections


def _markdown_sections(md_path: str) -> list[Section]:
    """Split Markdown on top-level (and second-level) headings."""
    raw = Path(md_path).read_text(encoding="utf-8")
    heading_re = re.compile(r"^(#{1,2})\s+(.+?)\s*#*\s*$")

    sections: list[Section] = []
    title = ""
    body: list[str] = []

    def flush() -> None:
        text = "\n\n".join(part for part in body if part.strip())
        if text.strip():
            sections.append(Section(title=title or f"Section {len(sections) + 1}", text=text))
        body.clear()

    for line in raw.splitlines():
        match = heading_re.match(line)
        if match:
            flush()
            title = _clean_title(match.group(2))
        else:
            body.append(line)
    flush()

    return sections


def _fallback_sections(text: str, target_chars: int) -> list[Section]:
    """Group text into fixed-size sections when a format has no structure."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    sections: list[Section] = []
    buf: list[str] = []
    length = 0
    for paragraph in paragraphs:
        buf.append(paragraph)
        length += len(paragraph) + 2
        if length >= target_chars:
            sections.append(Section(title=f"Part {len(sections) + 1}", text="\n\n".join(buf)))
            buf, length = [], 0

    if buf:
        # Avoid a tiny trailing section: fold it into the previous one.
        if sections and length < target_chars // 3:
            sections[-1].text += "\n\n" + "\n\n".join(buf)
        else:
            sections.append(Section(title=f"Part {len(sections) + 1}", text="\n\n".join(buf)))
    return sections


def extract_sections(
    file_path: str,
    target_chars: int = DEFAULT_SECTION_CHARS,
    min_chars: int = MIN_SECTION_CHARS,
) -> list[Section]:
    """
    Split a book into titled :class:`Section` objects.

    EPUB and Markdown use their own structure. PDFs and plain text have no
    reliable structure, so they are grouped into ``target_chars`` sized parts.
    Sections with less than ``min_chars`` characters are discarded.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    sections: list[Section] = []
    if ext == ".epub":
        sections = _epub_sections(str(path))
    elif ext in (".md", ".markdown"):
        sections = _markdown_sections(str(path))

    if not sections:
        sections = _fallback_sections(extract_text(str(path)), target_chars)

    if min_chars > 0:
        sections = [s for s in sections if not _is_artifact(s.text, min_chars)]

    if not sections:
        raise ValueError(
            f"No readable text found in {path.name}. "
            "It may be a scanned PDF that needs OCR."
        )
    return sections
