import hashlib
import logging
from pathlib import Path
from typing import Dict, List

import PyPDF2

from config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR

logger = logging.getLogger(__name__)


class Document:
    def __init__(self, page_content: str, metadata: Dict):
        self.page_content = page_content
        self.metadata = metadata


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_pdfs() -> List[Document]:
    documents: List[Document] = []
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    for pdf_path in pdf_files:
        try:
            reader = PyPDF2.PdfReader(str(pdf_path))
            for page_no, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                for chunk_no, chunk in enumerate(_split_text(text)):
                    documents.append(
                        Document(
                            chunk,
                            {
                                "source": pdf_path.name,
                                "page": page_no,
                                "chunk": chunk_no,
                                "type": "pdf",
                            },
                        )
                    )
        except Exception as exc:
            logger.exception("Could not load %s: %s", pdf_path, exc)

    return documents


def pdf_fingerprint() -> str:
    hasher = hashlib.sha256()
    for path in sorted(PDF_DIR.glob("*.pdf")):
        hasher.update(path.name.encode())
        hasher.update(str(path.stat().st_mtime_ns).encode())
        hasher.update(str(path.stat().st_size).encode())
    return hasher.hexdigest()
