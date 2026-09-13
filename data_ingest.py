"""
Document loading, text chunking, and Chroma vector-store construction.

The module keeps ordinary text and extracted tables as separate content types
so table semantics are not lost during PDF ingestion.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from models import EMBEDDING_MODEL

CHROMA_PATH = "chroma"


def load_documents(data_path: str):
    """Load Markdown and PDF pages with source/page metadata."""
    documents: list[Document] = []
    root = Path(data_path)

    for path in root.rglob("*.md"):
        docs = TextLoader(str(path), encoding="utf-8").load()
        for doc in docs:
            doc.metadata.setdefault("source", str(path))
            doc.metadata["content_type"] = "text"
        documents.extend(docs)

    for path in root.rglob("*.pdf"):
        docs = PyPDFLoader(str(path)).load()
        for page_index, doc in enumerate(docs):
            doc.metadata.setdefault("source", str(path))
            doc.metadata["page"] = int(doc.metadata.get("page", page_index))
            doc.metadata["content_type"] = "text"
        documents.extend(docs)

    return documents


def split_text(documents):
    """Split ordinary text; tables are handled separately."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["content_type"] = "text"
        chunk.metadata["chunk_index"] = i

    print(
        f"Split {len(documents)} document pages into "
        f"{len(chunks)} text chunks."
    )
    return chunks


def _table_as_text(rows: list[list[object]]) -> str:
    cleaned = []
    for row in rows:
        values = ["" if v is None else str(v).strip() for v in row]
        while values and not values[-1]:
            values.pop()
        if any(values):
            cleaned.append(" | ".join(values))
    return "\n".join(cleaned)


def build_embedded_dataset(
    data_path: str,
    *,
    documents=None,
    chunks=None,
    table_documents=None,
) -> Chroma:
    """
    Build Chroma from precomputed content when supplied.

    Existing callers can still pass only data_path, preserving the original
    interface.
    """
    if chunks is None:
        documents = documents if documents is not None else load_documents(data_path)
        chunks = split_text(documents)

    vector_documents = list(chunks)

    if table_documents:
        vector_documents.extend(table_documents)

    embedding_function = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    existing = db.get()
    if existing.get("ids"):
        db.delete(ids=existing["ids"])

    if vector_documents:
        db.add_documents(vector_documents)

    print(
        f"Saved {len(vector_documents)} searchable items to {CHROMA_PATH} "
        f"({len(chunks)} text chunks, {len(table_documents or [])} table records)."
    )
    return db


# Kept public for compatibility with any existing imports.
__all__ = [
    "CHROMA_PATH",
    "load_documents",
    "split_text",
    "build_embedded_dataset",
]
