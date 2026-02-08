from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ...core.utils import normalize_whitespace
from .generic_static import enrich_detail as _enrich_detail


def _text(node: Any) -> str:
    if not node:
        return ""
    return normalize_whitespace(node.get_text(" ", strip=True))


def fetch_list(
    url: str, html: str, *, selectors: Dict[str, str] | None = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """EGU Jobs list parser (article.media cards)."""
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []
    for article in soup.select("article.media"):
        title_node = article.select_one("h2.h4 a")
        if not title_node:
            continue
        title = _text(title_node)
        href = (title_node.get("href") or "").strip()
        if not href:
            continue
        full_url = urljoin(url, href)

        org = ""
        location = ""
        posted_at = ""
        li_nodes = article.select("ul.list-inline li")
        if len(li_nodes) >= 1:
            org = _text(li_nodes[0])
        if len(li_nodes) >= 2:
            location = _text(li_nodes[1])
        if len(li_nodes) >= 3:
            posted_at = _text(li_nodes[2])

        items.append(
            {
                "title": title,
                "url": full_url,
                "org": org,
                "location": location,
                "posted_at": posted_at,
                "deadline": "",
            }
        )
        if len(items) >= limit:
            break
    return items


def enrich_detail(
    detail_html: str,
    *,
    selectors: Dict[str, str] | None = None,
    min_len: int = 120,
    enable_fallback: bool = True,
) -> Dict[str, Any]:
    """Parse EGU job detail rows; fallback to generic extractor if needed."""
    soup = BeautifulSoup(detail_html, "html.parser")
    container = soup.select_one("#content") or soup
    rows = container.select("div.row")

    title = ""
    org = ""
    location = ""
    posted_at = ""
    deadline = ""
    description = ""

    for row in rows:
        label_node = row.select_one("div.col-md-3")
        value_node = row.select_one("div.col-md-9")
        if not label_node or not value_node:
            continue
        label = _text(label_node).lower()
        if not label:
            continue

        if label == "position":
            title = _text(value_node)
        elif label == "employer":
            p = value_node.find("p")
            org = _text(p) if p else _text(value_node)
        elif label == "location":
            location = _text(value_node)
        elif label == "application deadline":
            deadline = _text(value_node)
        elif label == "posted":
            posted_at = _text(value_node)
        elif label == "job description":
            description = _text(value_node)

    if not description or len(description) < min_len:
        fallback = _enrich_detail(
            detail_html,
            selectors=selectors,
            min_len=min_len,
            enable_fallback=enable_fallback,
        )
        if not description and fallback.get("description"):
            description = fallback.get("description", "")
        extract_method = fallback.get("extract_method", "") or "egu"
        extract_warnings = list(fallback.get("extract_warnings") or [])
    else:
        extract_method = "egu"
        extract_warnings = []

    return {
        "description": description,
        "extract_method": extract_method,
        "extract_warnings": extract_warnings,
        "title": title,
        "org": org,
        "location": location,
        "posted_at": posted_at,
        "deadline": deadline,
        "url": "",
    }
