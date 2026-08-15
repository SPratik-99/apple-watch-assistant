import logging
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, RETRIEVAL_K
from retrieval.document_loader import Document, load_pdfs, pdf_fingerprint

logger = logging.getLogger(__name__)


class StaticPDFRetriever:
    """Vector retrieval for static PDF knowledge only. No prices or live data."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = None
        self.document_count = 0
        self.fingerprint = pdf_fingerprint()
        self._prepare_collection()

    def _prepare_collection(self):
        existing = self.client.get_or_create_collection(COLLECTION_NAME)
        metadata = existing.metadata or {}
        stored_fingerprint = metadata.get("pdf_fingerprint")

        if stored_fingerprint != self.fingerprint:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            existing = self.client.get_or_create_collection(
                COLLECTION_NAME,
                metadata={"pdf_fingerprint": self.fingerprint},
            )
            self._index_documents(existing)

        self.collection = existing
        self.document_count = self.collection.count()

    def _index_documents(self, collection):
        docs: List[Document] = load_pdfs()
        if not docs:
            logger.warning("No PDFs found in %s", str(__import__('config').PDF_DIR))
            return

        batch_size = 64
        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            texts = [d.page_content for d in batch]
            embeddings = self.embedding_model.encode(texts, normalize_embeddings=True).tolist()
            ids = [f"pdf_{start+i}_{abs(hash(d.page_content))}" for i, d in enumerate(batch)]
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=[d.metadata for d in batch],
            )

    def search(self, query: str, k: int = RETRIEVAL_K) -> List[Dict]:
        if not self.collection or self.collection.count() == 0:
            return []

        embedding = self.embedding_model.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(
            query_embeddings=embedding,
            n_results=min(k, self.collection.count()),
        )

        output = []
        for text, metadata, distance in zip(
            result.get("documents", [[]])[0],
            result.get("metadatas", [[]])[0],
            result.get("distances", [[]])[0],
        ):
            output.append({"text": text, "metadata": metadata or {}, "distance": distance})
        return output

    def status(self):
        return {
            "documents": self.collection.count() if self.collection else 0,
            "pdf_fingerprint": self.fingerprint,
        }
