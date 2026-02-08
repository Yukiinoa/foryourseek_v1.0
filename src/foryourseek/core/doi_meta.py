from __future__ import annotations

import json
from urllib.parse import quote

from bs4 import BeautifulSoup

from .http import HttpClient
from .utils import normalize_doi, normalize_whitespace


def fetch_crossref_abstract(doi: str, http: HttpClient) -> str:
    """Fetch abstract from Crossref by DOI.

    Returns empty string if unavailable.
    """
    if not doi:
        return ""
    url = f"https://api.crossref.org/works/{quote(doi)}"
    try:
        text = http.get_text(url, headers={"Accept": "application/json"})
        data = json.loads(text)
        abstract = (data.get("message", {}) or {}).get("abstract") or ""
    except Exception:
        return ""

    if not abstract:
        return ""
    # Crossref often returns JATS/HTML; strip tags to plain text.
    cleaned = BeautifulSoup(abstract, "html.parser").get_text(" ", strip=True)
    return normalize_whitespace(cleaned)


def search_crossref_by_title(title: str, journal: str, http: HttpClient) -> tuple[str, str]:
    """Search Crossref by title (+ optional journal) to get DOI and abstract.

    Returns (doi, abstract); empty strings if not found.
    """
    if not title:
        return "", ""
    q_title = quote(title)
    q_journal = quote(journal) if journal else ""
    query = f"query.title={q_title}"
    if q_journal:
        query += f"&query.container-title={q_journal}"
    url = f"https://api.crossref.org/works?{query}&rows=1"
    try:
        text = http.get_text(url, headers={"Accept": "application/json"})
        data = json.loads(text)
        items = (data.get("message", {}) or {}).get("items") or []
        if not items:
            return "", ""
        item = items[0] or {}
        doi = normalize_doi(item.get("DOI") or "")
        abstract = item.get("abstract") or ""
    except Exception:
        return "", ""

    if abstract:
        abstract = BeautifulSoup(abstract, "html.parser").get_text(" ", strip=True)
        abstract = normalize_whitespace(abstract)
    return doi, abstract


def _openalex_inverted_index_to_text(inv: dict) -> str:
    if not inv:
        return ""
    max_pos = -1
    for positions in inv.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    if max_pos < 0:
        return ""
    words = [""] * (max_pos + 1)
    for word, positions in inv.items():
        for p in positions:
            if 0 <= p <= max_pos:
                words[p] = word
    return normalize_whitespace(" ".join(words))


def search_openalex_by_title(title: str, journal: str, http: HttpClient) -> tuple[str, str]:
    """Search OpenAlex by title (+ optional journal) to get DOI and abstract."""
    if not title:
        return "", ""
    q_title = quote(title)
    url = f"https://api.openalex.org/works?search={q_title}&per-page=1"
    if journal:
        q_journal = quote(journal)
        url += f"&filter=primary_location.source.display_name:{q_journal}"
    try:
        text = http.get_text(url, headers={"Accept": "application/json"})
        data = json.loads(text)
        results = data.get("results") or []
        if not results:
            return "", ""
        item = results[0] or {}
        doi = normalize_doi(item.get("doi") or "")
        inv = item.get("abstract_inverted_index") or {}
        abstract = _openalex_inverted_index_to_text(inv)
    except Exception:
        return "", ""
    return doi, abstract


def search_openalex_by_title_source_id(
    title: str, source_id: str, http: HttpClient
) -> tuple[str, str]:
    """Search OpenAlex by title restricted to a source id (Sxxxx)."""
    if not title:
        return "", ""
    sid = _normalize_openalex_source_id(source_id)
    if not sid:
        return "", ""
    q_title = quote(title)
    url = f"https://api.openalex.org/works?search={q_title}&per-page=1&filter=primary_location.source.id:{sid}"
    try:
        text = http.get_text(url, headers={"Accept": "application/json"})
        data = json.loads(text)
        results = data.get("results") or []
        if not results:
            return "", ""
        item = results[0] or {}
        doi = normalize_doi(item.get("doi") or "")
        inv = item.get("abstract_inverted_index") or {}
        abstract = _openalex_inverted_index_to_text(inv)
    except Exception:
        return "", ""
    return doi, abstract


def _normalize_openalex_source_id(source_id: str) -> str:
    if not source_id:
        return ""
    sid = source_id.strip()
    if sid.startswith("https://openalex.org/"):
        sid = sid.split("/")[-1]
    return sid


def resolve_openalex_source_id(journal: str, http: HttpClient) -> str:
    """Resolve OpenAlex source id (Sxxxx) from journal name."""
    if not journal:
        return ""
    q = quote(journal)
    url = f"https://api.openalex.org/sources?search={q}&per-page=1"
    try:
        text = http.get_text(url, headers={"Accept": "application/json"})
        data = json.loads(text)
        results = data.get("results") or []
        if not results:
            return ""
        sid = (results[0].get("id") or "").strip()
        return _normalize_openalex_source_id(sid)
    except Exception:
        return ""


def search_openalex_by_source_id(source_id: str, http: HttpClient, limit: int = 20) -> list[dict]:
    """Fetch recent works from OpenAlex by source id (Sxxxx)."""
    sid = _normalize_openalex_source_id(source_id)
    if not sid:
        return []
    url = (
        "https://api.openalex.org/works"
        f"?filter=primary_location.source.id:{sid}"
        f"&sort=publication_date:desc&per-page={int(limit)}"
    )
    try:
        text = http.get_text(url, headers={"Accept": "application/json"})
        data = json.loads(text)
        results = data.get("results") or []
    except Exception:
        return []

    items: list[dict] = []
    for item in results:
        title = normalize_whitespace(item.get("title") or "")
        doi = normalize_doi(item.get("doi") or "")
        published_at = item.get("publication_date") or ""
        inv = item.get("abstract_inverted_index") or {}
        abstract = _openalex_inverted_index_to_text(inv)
        primary = item.get("primary_location") or {}
        landing = primary.get("landing_page_url") or ""
        if not landing:
            landing = (item.get("ids") or {}).get("doi") or ""
        items.append(
            {
                "title": title,
                "url": landing,
                "published_at": published_at,
                "summary": abstract,
                "doi": doi,
                "source": sid,
            }
        )
    return items


def search_openalex_by_journal(journal: str, http: HttpClient, limit: int = 20) -> list[dict]:
    """Fetch recent works from OpenAlex by journal name (resolved to source id)."""
    sid = resolve_openalex_source_id(journal, http=http)
    if not sid:
        return []
    return search_openalex_by_source_id(sid, http=http, limit=limit)
