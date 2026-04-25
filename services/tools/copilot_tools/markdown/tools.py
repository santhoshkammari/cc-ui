import json
import re
import textwrap
from pathlib import Path
from tabulate import tabulate

from .mrkdwn_analysis import MarkdownAnalyzer


def format_beautiful_table(table_data, max_width=40, tablefmt='grid'):
    def wrap_text(text, width=max_width):
        if not isinstance(text, str):
            text = str(text)
        return '\n'.join(textwrap.wrap(text, width=width, break_long_words=True))

    wrapped_headers = [wrap_text(header) for header in table_data['header']]
    wrapped_rows = [[wrap_text(cell) for cell in row] for row in table_data['rows']]
    return tabulate(wrapped_rows, headers=wrapped_headers, tablefmt=tablefmt, stralign='left')


def _get_analyzer(content: str) -> MarkdownAnalyzer:
    """Create a MarkdownAnalyzer from a file path or raw markdown string."""
    p = Path(content)
    if len(content) <= 260 and p.exists():
        return MarkdownAnalyzer(content)
    return MarkdownAnalyzer.from_string(content)


def markdown_analyzer_get_headers(content: str):
    """Extract all headers with line numbers from markdown content or file path.

    Args:
        content: Raw markdown string or path to a markdown file
    """
    try:
        analyzer = _get_analyzer(content)
        headers = analyzer.identify_headers()
        header_list = headers.get('Header', [])
        return headers if header_list else "No headers found"
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_paragraphs(content: str):
    """Extract all paragraphs with line numbers from markdown content or file path.

    Args:
        content: Raw markdown string or path to a markdown file
    """
    try:
        analyzer = _get_analyzer(content)
        result = {"Paragraph": []}
        for token in analyzer.tokens:
            if token.type == 'paragraph':
                result["Paragraph"].append({"line": token.line, "content": token.content.strip()})
        return result if result["Paragraph"] else "No paragraphs found"
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_links(content: str):
    """Extract HTTP/HTTPS links with line numbers from markdown content or file path.

    Args:
        content: Raw markdown string or path to a markdown file
    """
    try:
        analyzer = _get_analyzer(content)
        links = analyzer.identify_links()
        filter_links = [x for x in links.get('Text link', []) if x.get('url', '').lower().startswith('http')]
        return filter_links if filter_links else "No HTTP links found"
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_code_blocks(content: str):
    """Extract all code blocks with line numbers from markdown content or file path.

    Args:
        content: Raw markdown string or path to a markdown file
    """
    try:
        analyzer = _get_analyzer(content)
        code_blocks = analyzer.identify_code_blocks()
        code_list = code_blocks.get('Code block', [])
        return code_blocks if code_list else "No code blocks found"
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_tables_metadata(content: str):
    """Extract table metadata (headers, line numbers) from markdown content or file path.

    Args:
        content: Raw markdown string or path to a markdown file
    """
    try:
        analyzer = _get_analyzer(content)
        tables_metadata = []
        table_index = 1
        for token in analyzer.tokens:
            if token.type == 'table':
                headers = token.meta.get("header", [])
                tables_metadata.append([
                    str(table_index),
                    str(token.line),
                    str(len(headers)),
                    str(len(token.meta.get("rows", []))),
                    ", ".join(headers)
                ])
                table_index += 1
        if not tables_metadata:
            return "No tables found"
        return format_beautiful_table({
            "header": ["Table #", "Line", "Columns", "Rows", "Headers Preview"],
            "rows": tables_metadata
        })
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_table_by_line(content: str, line_number: int):
    """Extract and format a specific table at the given line number.

    Args:
        content: Raw markdown string or path to a markdown file
        line_number: Line number where the table starts
    """
    try:
        analyzer = _get_analyzer(content)
        for token in analyzer.tokens:
            if token.type == 'table' and token.line == line_number:
                headers = token.meta.get("header", [])
                rows = token.meta.get("rows", [])
                clean = lambda s: re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s).strip()
                return format_beautiful_table({
                    "header": [clean(h) for h in headers],
                    "rows": [[clean(c) for c in row] for row in rows]
                }, max_width=15, tablefmt='grid')
        return f"No table at line {line_number}"
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_header_by_line(content: str, line_number: int):
    """Extract the section content under a specific header line.

    Args:
        content: Raw markdown string or path to a markdown file
        line_number: Line number where the header is located
    """
    try:
        analyzer = _get_analyzer(content)
        headers = analyzer.identify_headers()
        header_list = headers.get('Header', [])

        target_header = next((h for h in header_list if h.get('line') == line_number), None)
        if not target_header:
            return f"No header at line {line_number}"

        raw = content if '\n' in content else Path(content).read_text(encoding='utf-8')
        content_lines = raw.splitlines(keepends=True)

        target_level = target_header.get('level', 1)
        start_line = target_header.get('line', 1)
        end_line = len(content_lines)
        for h in header_list:
            if h.get('line', 1) > start_line and h.get('level', 1) <= target_level:
                end_line = h['line'] - 1
                break

        section_text = '\n'.join(
            content_lines[i].rstrip('\n') for i in range(start_line, min(end_line, len(content_lines)))
        ).strip()

        return {
            "header": {"line": start_line, "level": target_level, "text": target_header.get('text', '')},
            "content": section_text,
            "content_lines": f"{start_line + 1}-{min(end_line, len(content_lines))}",
            "word_count": len(section_text.split()) if section_text else 0
        }
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_intro(content: str):
    """Extract introduction/abstract/summary from markdown content or file path.

    Args:
        content: Raw markdown string or path to a markdown file
    """
    try:
        raw = content if '\n' in content else Path(content).read_text(encoding='utf-8')
        analyzer = _get_analyzer(content)
        headers = analyzer.identify_headers()
        header_list = headers.get('Header', [])

        intro_patterns = [
            'abstract', 'summary', 'executive summary',
            'introduction', 'intro', 'overview', 'about',
            'preface', 'foreword', 'background', 'context',
            'getting started', 'what is', 'description'
        ]

        intro_header = next(
            (h for h in header_list if any(p in h.get('text', '').lower() for p in intro_patterns)),
            None
        )

        if intro_header:
            content_lines = raw.split('\n')
            target_level = intro_header.get('level', 1)
            start_line = intro_header.get('line', 1)
            end_line = len(content_lines)
            for h in header_list:
                if h.get('line', 1) > start_line and h.get('level', 1) <= target_level:
                    end_line = h['line'] - 1
                    break
            section_text = '\n'.join(content_lines[start_line:min(end_line, len(content_lines))]).strip()
            return {
                "type": "explicit_header",
                "header": {"line": start_line, "level": target_level, "text": intro_header.get('text', '')},
                "content": section_text,
                "content_lines": f"{start_line + 1}-{min(end_line, len(content_lines))}",
                "word_count": len(section_text.split()) if section_text else 0
            }

        if not header_list:
            return "No structure found for intro extraction"

        first_header = header_list[0]
        title_line = first_header.get('line', 1)
        paragraphs = [
            {"line": t.line, "content": t.content.strip()}
            for t in analyzer.tokens if t.type == 'paragraph' and t.line > title_line
        ]
        if not paragraphs:
            return "No introductory content found"

        intro_paragraphs, word_count = [], 0
        for para in paragraphs:
            para_words = len(para['content'].split())
            if word_count + para_words > 300:
                break
            intro_paragraphs.append(para)
            word_count += para_words
            if len(intro_paragraphs) >= 3:
                break

        return {
            "type": "inferred_paragraphs",
            "header": {"line": title_line, "level": first_header.get('level', 1), "text": first_header.get('text', '')},
            "content": '\n\n'.join(p['content'] for p in intro_paragraphs),
            "content_lines": f"{intro_paragraphs[0]['line']}-{intro_paragraphs[-1]['line']}",
            "word_count": word_count,
            "paragraphs_count": len(intro_paragraphs)
        }
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_lists(content: str):
    """Extract all ordered and unordered lists from markdown content or file path.

    Args:
        content: Raw markdown string or path to a markdown file
    """
    try:
        analyzer = _get_analyzer(content)
        result = {"Ordered list": [], "Unordered list": []}
        for token in analyzer.tokens:
            if token.type == 'ordered_list':
                result["Ordered list"].append({"line": token.line, "items": token.meta["items"]})
            elif token.type == 'unordered_list':
                result["Unordered list"].append({"line": token.line, "items": token.meta["items"]})
        return result if (result["Ordered list"] or result["Unordered list"]) else "No lists found"
    except Exception:
        return "Failed to analyze content"


def markdown_analyzer_get_overview(content: str):
    """Get complete eagle-eye overview of markdown content or file path.

    Args:
        content: Raw markdown string or path to a markdown file
    """
    try:
        analyzer = _get_analyzer(content)
        headers = analyzer.identify_headers()
        paragraphs = analyzer.identify_paragraphs()
        links = analyzer.identify_links()
        code_blocks = analyzer.identify_code_blocks()
        tables = analyzer.identify_tables()
        lists = analyzer.identify_lists()
        intro_result = markdown_analyzer_get_intro(content)

        http_links = [x for x in links.get('Text link', []) if x.get('url', '').lower().startswith('http')]
        paragraph_list = paragraphs.get('Paragraph', [])
        word_count = sum(len(p.split()) for p in paragraph_list)
        header_list = headers.get('Header', [])

        structure = [
            f"{'  ' * (h.get('level', 1) - 1)}H{h.get('level', 1)}: {h.get('text', '')} (line {h.get('line', 'N/A')})"
            for h in header_list
        ]
        code_block_summary = [
            f"{cb.get('language', 'unknown')} code block (lines {cb.get('start_line', 'N/A')}-{cb.get('end_line', 'N/A')})"
            for cb in code_blocks.get('Code block', [])
        ]
        table_summary = [f"Table at line {t.get('line', 'N/A')}" for t in tables.get('Table', [])]

        intro_info = {
            "found": isinstance(intro_result, dict),
            "type": intro_result.get('type', 'none') if isinstance(intro_result, dict) else 'none',
            "word_count": intro_result.get('word_count', 0) if isinstance(intro_result, dict) else 0,
            "content": intro_result.get('content', '') if isinstance(intro_result, dict) else 'No introduction found'
        }

        ds = "\n".join(structure)
        hl = "\n".join(f"- {h.get('text', '')}" for h in header_list)
        cb_detail = "\n".join(f"- {cb}" for cb in code_block_summary) if code_block_summary else "None"
        tbl_detail = "\n".join(f"- {t}" for t in table_summary) if table_summary else "None"
        n_lists = len(lists.get('Ordered list', [])) + len(lists.get('Unordered list', []))

        return f"""# Document Overview: {header_list[0].get('text', 'Untitled') if header_list else 'Untitled'}

## Introduction/Abstract
- **Found**: {'Yes' if intro_info['found'] else 'No'}
- **Type**: {intro_info['type'].replace('_', ' ').title()}
- **Word Count**: {intro_info['word_count']}

{intro_info['content']}

## Document Structure
{ds}

## Content Statistics
- **Total Sections**: {len(header_list)}
- **Paragraphs**: {len(paragraph_list)}
- **Estimated Words**: {word_count}
- **Code Blocks**: {len(code_blocks.get('Code block', []))}
- **Tables**: {len(tables.get('Table', []))}
- **Lists**: {n_lists}
- **External Links**: {len(http_links)}

## Code Blocks Detail
{cb_detail}

## Tables Detail
{tbl_detail}

## All Headers List
{hl}
""" if header_list else "Empty document found"
    except Exception:
        return "Failed to analyze content"

