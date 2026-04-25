"""Internal: ChromaDB-backed web content store.

Each document follows the schema::

    {
        "id":          str,   # deterministic hash of URL
        "title":       str,
        "description": str,
        "url":         str,
        "text":        str,   # page markdown — or error / placeholder message
    }
"""

from __future__ import annotations

import hashlib
from typing import Optional

import chromadb

_STORE_PATH = "/tmp/copilot_web_store"
_COLLECTION_NAME = "web_pages"

NOT_FETCHED_MSG = "Content not fetched — use web_store_get_text(id) or re-search with fetch=True to retrieve."


def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=_STORE_PATH)


def _collection() -> chromadb.Collection:
    return _client().get_or_create_collection(_COLLECTION_NAME)


def url_to_id(url: str) -> str:
    """Deterministic short ID from a URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


# ── write operations ─────────────────────────────────────────────────────

def upsert_item(
    doc_id: str,
    *,
    title: str,
    description: str,
    url: str,
    text: str,
) -> None:
    """Insert or update a document in the store."""
    col = _collection()
    col.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[{"title": title, "description": description, "url": url}],
    )


def update_text(doc_id: str, text: str) -> None:
    """Update only the *text* field of an existing document."""
    col = _collection()
    col.update(ids=[doc_id], documents=[text])


# ── read operations ──────────────────────────────────────────────────────

def get_item(doc_id: str) -> Optional[dict]:
    """Return full item ``{id, title, description, url, text}`` or ``None``."""
    col = _collection()
    result = col.get(ids=[doc_id])
    if not result["ids"]:
        return None
    meta = result["metadatas"][0] if result["metadatas"] else {}
    return {
        "id": doc_id,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "url": meta.get("url", ""),
        "text": result["documents"][0] if result["documents"] else "",
    }


def list_items() -> list[dict]:
    """Return all stored items (without full text for brevity)."""
    col = _collection()
    result = col.get()
    items = []
    for i, doc_id in enumerate(result["ids"]):
        meta = result["metadatas"][i] if result["metadatas"] else {}
        text = result["documents"][i] if result["documents"] else ""
        has_content = bool(text.strip()) and not text.startswith(("Content not fetched", "Failed to fetch"))
        items.append({
            "id": doc_id,
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "has_content": has_content,
        })
    return items


def semantic_search(query: str, n_results: int = 5) -> list[dict]:
    """Semantic search across stored page content."""
    col = _collection()
    result = col.query(query_texts=[query], n_results=n_results)

    items = []
    ids = result["ids"][0] if result["ids"] else []
    docs = result["documents"][0] if result["documents"] else []
    metas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result.get("distances") else []

    for i, doc_id in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        text = docs[i] if i < len(docs) else ""
        items.append({
            "id": doc_id,
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "distance": distances[i] if i < len(distances) else None,
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
        })
    return items
