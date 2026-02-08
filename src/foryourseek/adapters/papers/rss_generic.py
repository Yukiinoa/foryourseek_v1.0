from __future__ import annotations

from typing import Any, Dict, List

import feedparser
from bs4 import BeautifulSoup

from ...core.utils import find_doi, normalize_whitespace


def parse_rss(text: str, source_name: str) -> List[Dict[str, Any]]:
    feed = feedparser.parse(text)
    items: List[Dict[str, Any]] = []
    for e in feed.entries or []:
        title = normalize_whitespace(getattr(e, "title", "") or "")
        link = getattr(e, "link", "") or ""
        published = getattr(e, "published", "") or getattr(e, "updated", "") or ""
        raw_summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
        # Strip HTML tags commonly embedded in RSS summaries
        summary = normalize_whitespace(
            BeautifulSoup(raw_summary, "html.parser").get_text(" ", strip=True)
        )
        raw_blob = f"{title}\n{link}\n{summary}\n{getattr(e, 'id', '')}"
        doi = find_doi(raw_blob)
        items.append(
            {
                "title": title,
                "url": link,
                "published_at": published,
                "summary": summary,
                "doi": doi,
                "source": source_name,
            }
        )
    return items
