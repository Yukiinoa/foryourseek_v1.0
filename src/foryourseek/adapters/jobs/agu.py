from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ...core.utils import normalize_whitespace
from .generic_static import enrich_detail as _enrich_detail
from .generic_static import fetch_list as _fetch_list


def _text(node: Any) -> str:
    if not node:
        return ""
    return normalize_whitespace(node.get_text(" ", strip=True))


def _clean_href(href: str) -> str:
    return "".join((href or "").split())


def fetch_list(
    url: str, html: str, *, selectors: Dict[str, str] | None = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """AGU Career Center listing parser (lister__item cards)."""
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []
    seen = set()

    for card in soup.select("li.lister__item"):
        title_node = card.select_one(
            "h3.lister__header a, h2.lister__header a, a.js-clickable-area-link"
        )
        if not title_node:
            continue
        title = _text(title_node)
        href = _clean_href(title_node.get("href") or "")
        if not href:
            continue
        full_url = urljoin(url, href)
        if full_url in seen:
            continue
        seen.add(full_url)

        org = _text(card.select_one("li.lister__meta-item--recruiter"))
        location = _text(card.select_one("li.lister__meta-item--location"))

        posted_at = ""
        for li in card.select("li.job-actions__action"):
            t = _text(li)
            if re.search(r"\b(ago|today|yesterday)\b", t.lower()):
                posted_at = t
                break

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

    if not items:
        return _fetch_list(url, html, selectors=selectors, limit=limit)
    return items


def enrich_detail(
    detail_html: str,
    *,
    selectors: Dict[str, str] | None = None,
    min_len: int = 120,
    enable_fallback: bool = True,
) -> Dict[str, Any]:
    return _enrich_detail(
        detail_html,
        selectors=selectors,
        min_len=min_len,
        enable_fallback=enable_fallback,
    )
