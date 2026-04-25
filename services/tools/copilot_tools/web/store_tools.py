"""Agent-facing ChromaDB store tools for web content.

These tools let an agent retrieve, browse, and search across
previously fetched web pages stored in ChromaDB.
"""

from __future__ import annotations

from copilot_tools.web._fetcher import fetch_url_as_markdown
from copilot_tools.web._store import (
    get_item,
    list_items,
    semantic_search,
    update_text,
)


def web_store_get(doc_id: str) -> dict | str:
    """Get a stored web page by ID — returns all fields.

    Parameters
    ----------
    doc_id:
        The ChromaDB document ID (returned by ``web_search``).

    Returns
    -------
    ``{id, title, description, url, text}`` or an error string.
    """
    item = get_item(doc_id)
    if item is None:
        return f"Error: no item with id '{doc_id}'. Run web_search first."
    return item


def web_store_get_text(doc_id: str) -> str:
    """Get just the text content for a stored page.

    If content hasn't been fetched yet, fetches it now and updates the store.

    Parameters
    ----------
    doc_id:
        The ChromaDB document ID.

    Returns
    -------
    The page markdown content, or an error message.
    """
    item = get_item(doc_id)
    if item is None:
        return f"Error: no item with id '{doc_id}'. Run web_search first."

    text = item["text"]

    # If text is a placeholder or error, try fetching now
    if text.startswith("Content not fetched"):
        url = item["url"]
        if not url:
            return f"Error: no URL stored for id '{doc_id}'."
        text = fetch_url_as_markdown(url)
        update_text(doc_id, text)

    return text


def web_store_search(query: str, n_results: int = 5) -> list[dict]:
    """Semantic search across all stored web pages.

    Parameters
    ----------
    query:
        Natural language query.
    n_results:
        Max results to return.

    Returns
    -------
    List of ``{id, title, url, distance, text_preview}`` dicts,
    ranked by relevance.
    """
    return semantic_search(query, n_results=n_results)


def web_store_list() -> list[dict]:
    """List all stored web pages.

    Returns
    -------
    List of ``{id, title, url, has_content}`` dicts.
    """
    return list_items()
