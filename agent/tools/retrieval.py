"""Document retrieval tool backed by a local FAISS index."""

from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from agent.tools import ToolResult


def _index_file(index_path: str) -> Path:
    """Return the FAISS index file path for the configured base path."""

    base_path = Path(index_path)
    if base_path.suffix == ".index":
        return base_path
    return base_path.with_suffix(".index")


def _metadata_file(index_path: str) -> Path:
    """Return the metadata file path paired with the FAISS index."""

    base_path = Path(index_path)
    if base_path.suffix:
        return base_path.with_suffix(".json")
    return base_path.with_suffix(".json")


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector for cosine-similarity-style inner-product search."""

    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _load_or_create_index(index_path: str, dimension: int) -> tuple[faiss.Index, list[str], list[str]]:
    """Load an on-disk FAISS index and chunk metadata, or create empty storage."""

    index_file = _index_file(index_path)
    metadata_file = _metadata_file(index_path)
    index_file.parent.mkdir(parents=True, exist_ok=True)

    if index_file.exists():
        index = faiss.read_index(str(index_file))
    else:
        index = faiss.IndexFlatIP(dimension)
        faiss.write_index(index, str(index_file))

    if metadata_file.exists():
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
        chunks = payload.get("chunks", [])
        sources = payload.get("sources", [])
    else:
        chunks = []
        sources = []
        metadata_file.write_text(json.dumps({"chunks": chunks, "sources": sources}), encoding="utf-8")

    return index, chunks, sources


def run_retrieval(query: str, index_path: str = "faiss_index", top_k: int = 3) -> ToolResult:
    """Retrieve top document chunks from a local FAISS index for a query.

    The function embeds the query with `text-embedding-3-small`, loads a
    disk-backed FAISS index, creates an empty one if needed, and returns
    the top matching chunks. Confidence is derived from the best
    normalized similarity score.
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ToolResult(
            content="",
            sources=[],
            raw_confidence=0.0,
            tool_name="document_retrieval",
        )

    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key,
        )
        query_embedding = np.asarray(embeddings.embed_query(query), dtype="float32")
        query_embedding = _normalize(query_embedding)

        index, chunks, sources = _load_or_create_index(
            index_path=index_path,
            dimension=query_embedding.shape[0],
        )
        if index.ntotal == 0 or not chunks:
            return ToolResult(
                content="",
                sources=[],
                raw_confidence=0.0,
                tool_name="document_retrieval",
            )

        search_k = min(top_k, len(chunks))
        scores, indices = index.search(np.asarray([query_embedding], dtype="float32"), search_k)
        selected_chunks: list[str] = []
        selected_sources: list[str] = []

        for raw_index in indices[0]:
            if 0 <= raw_index < len(chunks):
                chunk_text = chunks[raw_index]
                selected_chunks.append(chunk_text)
                if raw_index < len(sources):
                    selected_sources.append(sources[raw_index])
                else:
                    selected_sources.append(f"local://chunk/{raw_index}")

        top_score = float(scores[0][0]) if len(scores[0]) > 0 else 0.0
        confidence = max(0.0, min(1.0, top_score))
        source_content: dict[str, str] = {}
        for source, chunk in zip(selected_sources, selected_chunks):
            source_content[source] = source_content.get(source, "") + "\n" + chunk
        return ToolResult(
            source_content=source_content,
            content="\n\n".join(selected_chunks),
            sources=selected_sources,
            raw_confidence=confidence,
            tool_name="document_retrieval",
        )
    except Exception:
        return ToolResult(
            content="",
            sources=[],
            raw_confidence=0.0,
            tool_name="document_retrieval",
        )
