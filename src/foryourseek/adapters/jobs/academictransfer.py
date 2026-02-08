from __future__ import annotations

import html as htmlmod
import json
from typing import Any, Dict, Iterable, List

from bs4 import BeautifulSoup

from ...core.extract import fetch_content_smart
from ...core.utils import normalize_whitespace


def fetch_list(
    url: str,
    html: str,
    *,
    selectors: Dict[str, str] | None = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    # AcademicTransfer listings are loaded via API; list scraping is handled in job_agent.
    return []


def _iter_jsonld_objects(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, dict):
        if "@graph" in raw and isinstance(raw["@graph"], list):
            for n in raw["@graph"]:
                if isinstance(n, dict):
                    yield n
            return
        yield raw
        return
    if isinstance(raw, list):
        for n in raw:
            if isinstance(n, dict):
                yield n


def _jsonld_jobposting(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = s.get("children") or s.string or s.get_text() or ""
        if not raw:
            continue
        raw = htmlmod.unescape(raw)
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for obj in _iter_jsonld_objects(data):
            t = obj.get("@type") or obj.get("type")
            if isinstance(t, list):
                is_job = any(str(x).lower() == "jobposting" for x in t)
            else:
                is_job = str(t or "").lower() == "jobposting"
            if is_job:
                return obj
    return {}


def _format_address(addr: Any) -> str:
    if isinstance(addr, str):
        return normalize_whitespace(addr)
    if not isinstance(addr, dict):
        return ""
    parts = [
        addr.get("streetAddress", ""),
        addr.get("addressLocality", ""),
        addr.get("addressRegion", ""),
        addr.get("postalCode", ""),
        addr.get("addressCountry", ""),
    ]
    return normalize_whitespace(", ".join([p for p in parts if p]))


def enrich_detail(
    detail_html: str,
    *,
    selectors: Dict[str, str] | None = None,
    min_len: int = 120,
    enable_fallback: bool = True,
) -> Dict[str, Any]:
    selectors = selectors or {}
    r = fetch_content_smart(
        detail_html,
        specific_selector=selectors.get("description") or selectors.get("detail_description"),
        min_len=min_len,
        kind="job",
        enable_fallback=enable_fallback,
    )

    job_json = _jsonld_jobposting(detail_html)
    jsonld_desc = ""
    if job_json:
        raw_desc = job_json.get("description") or ""
        if raw_desc:
            jsonld_desc = normalize_whitespace(
                BeautifulSoup(raw_desc, "html.parser").get_text(" ", strip=True)
            )

    description = r.text
    method = r.method
    warnings = list(r.warnings)

    if jsonld_desc and (not description or method in ("bs4_text", "empty")):
        description = jsonld_desc if len(jsonld_desc) >= min_len or not description else description
        method = "jsonld"
        warnings.append("jsonld_preferred")

    title = normalize_whitespace(job_json.get("title") or job_json.get("name") or "")

    org = ""
    hiring = job_json.get("hiringOrganization") or {}
    if isinstance(hiring, list) and hiring:
        hiring = hiring[0]
    if isinstance(hiring, dict):
        org = normalize_whitespace(hiring.get("name") or "")

    location = ""
    loc = job_json.get("jobLocation") or {}
    if isinstance(loc, list) and loc:
        loc = loc[0]
    if isinstance(loc, dict):
        addr = loc.get("address") or {}
        location = _format_address(addr) or normalize_whitespace(loc.get("name") or "")

    posted_at = normalize_whitespace(job_json.get("datePosted") or "")
    deadline = normalize_whitespace(job_json.get("validThrough") or "")

    return {
        "description": description,
        "extract_method": method,
        "extract_warnings": warnings,
        "title": title,
        "org": org,
        "location": location,
        "posted_at": posted_at,
        "deadline": deadline,
        "url": normalize_whitespace(job_json.get("url") or ""),
    }
