"""Internal: fetch a URL and convert to markdown content."""

from __future__ import annotations


def fetch_url_as_markdown(url: str) -> str:
    """Fetch *url* and return main-content markdown.

    Returns the markdown string on success, or an error message string
    prefixed with ``"Failed to fetch content:"`` on failure.
    """
    try:
        from typing import cast
        from scrapling.fetchers import Fetcher
        from scrapling.core.shell import Convertor
        from scrapling.engines._browsers._types import ImpersonateType
        from scrapling.core._types import extraction_types

        page = Fetcher.get(
            url,
            timeout=30,
            retries=3,
            retry_delay=1,
            impersonate=cast(ImpersonateType, "chrome"),
        )
        parts = list(
            Convertor._extract_content(
                page,
                css_selector=None,
                extraction_type=cast(extraction_types, "markdown"),
                main_content_only=True,
            )
        )
        content = "\n".join(parts).strip()
        if not content:
            return f"Failed to fetch content: page at {url} returned empty body"
        return content
    except Exception as e:
        return f"Failed to fetch content: {e}"
