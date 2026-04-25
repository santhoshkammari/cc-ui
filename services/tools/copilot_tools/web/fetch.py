"""``web_fetch`` — fetch a single URL and return markdown content."""

from __future__ import annotations

from copilot_tools.web._fetcher import fetch_url_as_markdown


def web_fetch(url: str) -> str:
    """Fetch a URL and return its content as markdown.

    Parameters
    ----------
    url:
        The URL to fetch.

    Returns
    -------
    Markdown content string, or an error message on failure.
    """
    return fetch_url_as_markdown(url)
