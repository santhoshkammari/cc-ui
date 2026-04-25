"""
Markdown Analysis Tools — structured extraction from markdown documents.

    get_overview        — Full eagle-eye overview (stats, structure, intro)
    get_headers         — Extract all headers with line numbers
    get_section         — Extract section content under a specific header
    get_intro           — Extract introduction/abstract/summary
    get_links           — Extract HTTP/HTTPS links
    get_tables_metadata — Table metadata (headers, line numbers, column counts)
    get_table           — Extract and format a specific table by line number
"""

from .tools import (
    markdown_analyzer_get_headers as get_headers,
    markdown_analyzer_get_header_by_line as get_section,
    markdown_analyzer_get_intro as get_intro,
    markdown_analyzer_get_overview as get_overview,
    markdown_analyzer_get_links as get_links,
    markdown_analyzer_get_table_by_line as get_table,
    markdown_analyzer_get_tables_metadata as get_tables_metadata,
)

__all__ = [
    "get_overview",
    "get_headers",
    "get_section",
    "get_intro",
    "get_links",
    "get_tables_metadata",
    "get_table",
]

tools = {
    "markdown_get_overview": get_overview,
    "markdown_get_headers": get_headers,
    "markdown_get_section": get_section,
    "markdown_get_intro": get_intro,
    "markdown_get_links": get_links,
    "markdown_get_tables_metadata": get_tables_metadata,
    "markdown_get_table": get_table,
}