"""``web_search`` — search the web, store results in ChromaDB, optionally fetch pages."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Optional

from copilot_tools.web._fetcher import fetch_url_as_markdown
from copilot_tools.web._store import (
    NOT_FETCHED_MSG,
    get_item,
    update_text,
    upsert_item,
    url_to_id,
)


def _ddg_search(query: str, max_results: int = 4) -> list[dict]:
    """Run a DuckDuckGo text search.  Returns raw result dicts."""
    from ddgs import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _fetch_and_store(doc_id: str, url: str) -> str:
    """Fetch a page and update its ChromaDB entry.  Returns status message."""
    text = fetch_url_as_markdown(url)
    update_text(doc_id, text)
    return text


def web_search(
    query: str,
    *,
    fetch: bool = True,
    max_results: int = 4,
) -> str:
    """Search the web and store results in ChromaDB.

    Parameters
    ----------
    query:
        The search query.
    fetch:
        If ``True`` (default), fetch full page content for every result
        concurrently and store it.  If ``False``, store metadata only —
        use ``web_store_get_text(id)`` later to retrieve content.
    max_results:
        Number of search results (default 4).

    Returns
    -------
    A summary string listing each result with its ChromaDB ID.
    The agent can then use ``web_store_get_text(id)`` to read full content.
    """
    try:
        raw = _ddg_search(query, max_results=max_results)
    except Exception as e:
        return f"Search error: {e}"

    if not raw:
        return "No results found."

    # Build items and upsert stubs
    items: list[dict] = []
    for r in raw:
        url = r.get("href", "")
        doc_id = url_to_id(url)
        title = r.get("title", "Untitled")
        description = r.get("body", "")

        upsert_item(
            doc_id,
            title=title,
            description=description,
            url=url,
            text=NOT_FETCHED_MSG,
        )
        items.append({"id": doc_id, "title": title, "description": description, "url": url})

    # Fetch pages concurrently if requested
    if fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_results) as pool:
            futures = {
                pool.submit(_fetch_and_store, item["id"], item["url"]): item
                for item in items
            }
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                except Exception:
                    pass  # errors already captured in text field by _fetch_and_store

    # Build response
    lines = []
    for item in items:
        stored = get_item(item["id"])
        has_text = False
        if stored:
            txt = stored["text"]
            has_text = bool(txt.strip()) and not txt.startswith(("Content not fetched", "Failed to fetch"))

        status = "✅ content fetched" if has_text else "⏳ content not yet fetched"
        lines.append(
            f"[{item['id']}] {item['title']}\n"
            f"  {item['description']}\n"
            f"  url: {item['url']}\n"
            f"  status: {status}"
        )

    header = f"Found {len(items)} results for: {query!r}"
    hint = "Use web_store_get_text(id) to read full page content."
    return f"{header}\n{hint}\n\n" + "\n\n".join(lines)
