from __future__ import annotations

from typing import Any, Dict, List

from bs4 import BeautifulSoup

from ...core.extract import fetch_content_smart
from ...core.utils import find_doi, normalize_doi, normalize_whitespace


def enrich_paper_from_html(
    html: str,
    *,
    abstract_selector: str = "",
    min_len: int = 120,
    enable_fallback: bool = True,
) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    meta = {
        m.get("name") or m.get("property"): m.get("content")
        for m in soup.find_all("meta")
        if (m.get("name") or m.get("property")) and m.get("content")
    }

    # DOI
    doi = normalize_doi(meta.get("citation_doi") or meta.get("dc.identifier") or "") or find_doi(
        " ".join([str(v) for v in meta.values() if v])
    )
    # Authors
    authors = [
        m.get("content")
        for m in soup.find_all("meta", attrs={"name": "citation_author"})
        if m.get("content")
    ]
    authors_str = ", ".join([normalize_whitespace(a) for a in authors if a])[:800]

    # Abstract: selector -> meta -> smart extract
    abstract = ""
    abstract_source = ""
    extract_method = ""
    extract_warnings: List[str] = []
    if abstract_selector:
        r = fetch_content_smart(
            html,
            specific_selector=abstract_selector,
            min_len=min_len,
            kind="paper",
            enable_fallback=enable_fallback,
        )
        abstract = r.text
        if abstract:
            abstract_source = r.method
            extract_method = r.method
            extract_warnings = list(r.warnings)
    if not abstract:
        abstract = (
            meta.get("citation_abstract")
            or meta.get("description")
            or meta.get("og:description")
            or ""
        )
        abstract = normalize_whitespace(abstract)
        if abstract:
            abstract_source = "meta"
            extract_method = "meta"
    if not abstract:
        r = fetch_content_smart(
            html,
            specific_selector=None,
            min_len=min_len,
            kind="paper",
            enable_fallback=enable_fallback,
        )
        abstract = r.text
        if abstract:
            abstract_source = r.method
            extract_method = r.method
            extract_warnings = list(r.warnings)

    return {
        "doi": doi,
        "authors": authors_str,
        "abstract": abstract,
        "abstract_source": abstract_source,
        "extract_method": extract_method or abstract_source,
        "extract_warnings": extract_warnings,
    }
