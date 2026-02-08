from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ...core.extract import fetch_content_smart
from ...core.utils import normalize_whitespace


def _pick_selector(selectors: Dict[str, str], *keys: str) -> str:
    for k in keys:
        v = selectors.get(k)
        if v:
            return v
    return ""


def _node_text(node: Any, selector: str) -> str:
    if not selector:
        return ""
    n = node.select_one(selector)
    if not n:
        return ""
    return normalize_whitespace(n.get_text(" ", strip=True))


def _node_attr(node: Any, selector: str, attr: str = "href") -> str:
    if not selector:
        return ""
    n = node.select_one(selector)
    if not n or not n.has_attr(attr):
        return ""
    return str(n.get(attr) or "").strip()


def _first_link_url(node: Any) -> str:
    a = node if getattr(node, "name", "") == "a" else node.find("a", href=True)
    return (a.get("href") or "").strip() if a else ""


def fetch_list(
    url: str, html: str, *, selectors: Dict[str, str] | None = None, limit: int = 50
) -> List[Dict[str, Any]]:
    selectors = selectors or {}
    soup = BeautifulSoup(html, "html.parser")
    item_sel = _pick_selector(selectors, "list_item", "item", "item_selector", "list") or "a"
    nodes = soup.select(item_sel)
    items: List[Dict[str, Any]] = []
    seen_urls = set()

    title_sel = _pick_selector(selectors, "title", "list_title")
    url_sel = _pick_selector(selectors, "url", "list_url")
    org_sel = _pick_selector(selectors, "org", "list_org")
    location_sel = _pick_selector(selectors, "location", "list_location")
    deadline_sel = _pick_selector(selectors, "deadline", "list_deadline")
    posted_sel = _pick_selector(selectors, "posted_at", "list_posted_at")

    for n in nodes:
        title = _node_text(n, title_sel) if title_sel else ""
        if not title:
            a = n if getattr(n, "name", "") == "a" else n.find("a")
            title = normalize_whitespace(a.get_text(" ", strip=True)) if a else ""

        href = _node_attr(n, url_sel, "href") if url_sel else ""
        if not href and title_sel:
            tnode = n.select_one(title_sel)
            if tnode:
                a = tnode if tnode.name == "a" else tnode.find("a", href=True)
                href = (a.get("href") or "").strip() if a else ""
        if not href:
            href = _first_link_url(n)
        if not href:
            continue

        full_url = urljoin(url, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        if not title:
            continue

        org = _node_text(n, org_sel)
        location = _node_text(n, location_sel)
        deadline = _node_text(n, deadline_sel)
        posted_at = _node_text(n, posted_sel)

        if (not posted_at or not deadline) and n.find_all("time"):
            times = n.find_all("time")
            if not posted_at and times:
                posted_at = normalize_whitespace(
                    times[0].get("datetime") or times[0].get_text(" ", strip=True)
                )
            if not deadline and len(times) > 1:
                deadline = normalize_whitespace(
                    times[1].get("datetime") or times[1].get_text(" ", strip=True)
                )

        items.append(
            {
                "title": title,
                "url": full_url,
                "org": org,
                "location": location,
                "deadline": deadline,
                "posted_at": posted_at,
            }
        )
        if len(items) >= limit:
            break
    return items


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
        raw = (s.string or s.get_text() or "").strip()
        if not raw:
            continue
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
            if not is_job:
                continue
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


def _extract_meta_description(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for key in ["description", "og:description", "twitter:description"]:
        m = soup.find("meta", attrs={"name": key}) or soup.find("meta", attrs={"property": key})
        if m and m.get("content"):
            return normalize_whitespace(m.get("content"))
    return ""


def enrich_detail(
    detail_html: str,
    *,
    selectors: Dict[str, str] | None = None,
    min_len: int = 120,
    enable_fallback: bool = True,
) -> Dict[str, Any]:
    selectors = selectors or {}
    desc_sel = _pick_selector(selectors, "description", "detail_description") or ""
    r = fetch_content_smart(
        detail_html,
        specific_selector=desc_sel or None,
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

    meta_desc = _extract_meta_description(detail_html)

    description = r.text
    method = r.method
    warnings = list(r.warnings)

    if jsonld_desc and (not description or method in ("bs4_text", "empty")):
        description = jsonld_desc if len(jsonld_desc) >= min_len or not description else description
        method = "jsonld"
        warnings.append("jsonld_preferred")
    elif meta_desc and (not description or method in ("bs4_text", "empty")):
        description = meta_desc if len(meta_desc) >= min_len or not description else description
        method = "meta"
        warnings.append("meta_preferred")

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
