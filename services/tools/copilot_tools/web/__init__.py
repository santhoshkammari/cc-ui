"""
Web Tools — search, fetch, and store web content.

    web_search          — Search the web, store results in ChromaDB (fetch=True fetches pages)
    web_fetch           — Fetch a single URL → markdown
    web_store_get       — Get a stored page by ID (all fields)
    web_store_get_text  — Get just the text for a stored page (auto-fetches if missing)
    web_store_search    — Semantic search across stored pages
    web_store_list      — List all stored pages
"""

from .fetch import web_fetch
from .search import web_search
from .store_tools import (
    web_store_get,
    web_store_get_text,
    web_store_list,
    web_store_search,
)

__all__ = [
    "web_search",
    "web_fetch",
    "web_store_get",
    "web_store_get_text",
    "web_store_search",
    "web_store_list",
]

tools = {
    "web_search": web_search,
    "web_fetch": web_fetch,
    "web_store_get": web_store_get,
    "web_store_get_text": web_store_get_text,
    "web_store_search": web_store_search,
    "web_store_list": web_store_list,
}