"""Utilities for building a local FAISS index from text documents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from agent.tools.retrieval import _index_file, _metadata_file, _normalize


SUPPORTED_SUFFIXES = {".txt", ".md"}
"""Plain-text file types supported by the ingestion script."""


def _read_documents(input_dir: Path) -> list[tuple[str, str]]:
    """Load supported text documents from a directory recursively."""

    documents: list[tuple[str, str]] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8").strip()
        if content:
            documents.append((str(path), content))
    return documents


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a document into overlapping chunks for embedding."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def build_faiss_index(
    input_dir: str,
    index_path: str = "faiss_index",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> tuple[int, int]:
    """Build a FAISS index from local text files.

    Returns:
        A tuple of `(document_count, chunk_count)`.
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")

    source_dir = Path(input_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {source_dir}")

    documents = _read_documents(source_dir)
    if not documents:
        raise ValueError(f"No supported documents found in {source_dir}")

    chunk_texts: list[str] = []
    chunk_sources: list[str] = []
    for source_path, content in documents:
        for chunk in _chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunk_texts.append(chunk)
            chunk_sources.append(source_path)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    vectors = embeddings.embed_documents(chunk_texts)
    matrix = np.asarray(vectors, dtype="float32")
    for row_index in range(matrix.shape[0]):
        matrix[row_index] = _normalize(matrix[row_index])

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_file = _index_file(index_path)
    metadata_file = _metadata_file(index_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_file))
    metadata_file.write_text(
        json.dumps(
            {
                "chunks": chunk_texts,
                "sources": chunk_sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return len(documents), len(chunk_texts)


def main() -> None:
    """CLI entrypoint for building a local retrieval index."""

    parser = argparse.ArgumentParser(description="Build a FAISS index from local text documents.")
    parser.add_argument(
        "--input-dir",
        default="data/seed_docs",
        help="Directory containing .txt or .md documents to ingest.",
    )
    parser.add_argument(
        "--index-path",
        default="faiss_index",
        help="Base path for the FAISS index and metadata files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum number of characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
        help="Overlapping characters between consecutive chunks.",
    )
    args = parser.parse_args()

    document_count, chunk_count = build_faiss_index(
        input_dir=args.input_dir,
        index_path=args.index_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"[ingest] Indexed {document_count} documents into {chunk_count} chunks.")
    print(f"[ingest] FAISS index written to {_index_file(args.index_path)}")
    print(f"[ingest] Metadata written to {_metadata_file(args.index_path)}")


if __name__ == "__main__":
    main()
