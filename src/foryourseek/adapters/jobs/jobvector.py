from __future__ import annotations

import json
from typing import Any, Dict, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ...core.utils import normalize_whitespace
from .generic_static import enrich_detail as _enrich_detail


def _text(node: Any) -> str:
    if not node:
        return ""
    return normalize_whitespace(node.get_text(" ", strip=True))


def _jsonld_jobposting(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = s.get_text() or ""
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        objs: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            objs = [data]
        elif isinstance(data, list):
            objs = [d for d in data if isinstance(d, dict)]
        for obj in objs:
            if str(obj.get("@type") or "").lower() == "jobposting":
                return obj
    return {}


def _format_location(loc: Any) -> str:
    if not loc:
        return ""
    if isinstance(loc, list) and loc:
        loc = loc[0]
    if isinstance(loc, dict):
        addr = loc.get("address") or {}
        if isinstance(addr, dict):
            parts = [
                addr.get("addressLocality") or "",
                addr.get("addressRegion") or "",
                addr.get("addressCountry") or "",
            ]
            return normalize_whitespace(", ".join([p for p in parts if p]))
    return ""


def _format_salary(job: Dict[str, Any]) -> str:
    base = job.get("baseSalary")
    if isinstance(base, dict):
        currency = base.get("currency") or job.get("salaryCurrency") or ""
        value = base.get("value")
        if isinstance(value, dict):
            amount = value.get("value")
            unit = value.get("unitText") or ""
        else:
            amount = value
            unit = ""
        if amount:
            amount_str = str(amount)
            if currency and unit:
                return f"{currency} {amount_str} / {unit}"
            if currency:
                return f"{currency} {amount_str}"
            return amount_str
    return ""


def fetch_list(
    url: str, html: str, *, selectors: Dict[str, str] | None = None, limit: int = 50
) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []
    for article in soup.select("article.list-item-job"):
        if limit and len(items) >= limit:
            break
        title = _text(article.select_one(".job-description h2")) or _text(article.select_one("h2"))
        a = article.select_one("a.vacancy-title-anchor[href]") or article.select_one(
            'a[href*="/en/job/"]'
        )
        job_url = a.get("href") if a else ""
        if job_url:
            job_url = urljoin(url, job_url)
        org = _text(article.select_one(".company-name-text")) or _text(
            article.select_one(".company-name")
        )
        location = _text(article.select_one(".locations-loop-inside-wrapper")) or _text(
            article.select_one(".locations-wrapper")
        )
        deadline = _text(article.select_one(".date"))
        salary = ""
        euro = article.select_one("svg.fa-euro-sign")
        if euro:
            parent = euro.find_parent("span")
            salary = _text(parent)
        research_field = _text(article.select_one(".subjects"))
        items.append(
            {
                "title": title,
                "url": job_url,
                "org": org,
                "location": location,
                "deadline": deadline,
                "salary": salary,
                "research_field": research_field,
            }
        )
    return items


def enrich_detail(
    html: str,
    *,
    selectors: Dict[str, str] | None = None,
    min_len: int = 200,
    enable_fallback: bool = True,
) -> Dict[str, Any]:
    job = _jsonld_jobposting(html)
    if job:

        def _as_text(val: Any) -> str:
            if not val:
                return ""
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                for k in ("description", "name", "text", "value"):
                    v = val.get(k)
                    if isinstance(v, str) and v.strip():
                        return v
                return ""
            if isinstance(val, list):
                return " ".join([_as_text(v) for v in val if v])
            return ""

        parts = [
            _as_text(job.get("description")),
            _as_text(job.get("responsibilities")),
            _as_text(job.get("qualifications")),
            _as_text(job.get("experienceRequirements")),
            _as_text(job.get("jobBenefits")),
        ]
        desc_html = " ".join([p for p in parts if p])
        description = normalize_whitespace(
            BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)
        )
        org = ""
        hiring = job.get("hiringOrganization") or {}
        if isinstance(hiring, dict):
            org = normalize_whitespace(hiring.get("name") or "")
        posted_at = normalize_whitespace(job.get("datePosted") or "")
        deadline = normalize_whitespace(job.get("validThrough") or "")
        location = _format_location(job.get("jobLocation"))
        salary = _format_salary(job)
        education = job.get("educationRequirements")
        education_level = ""
        if isinstance(education, dict):
            education_level = normalize_whitespace(education.get("credentialCategory") or "")
        elif isinstance(education, str):
            education_level = normalize_whitespace(education)
        employment = job.get("employmentType")
        contract_type = ""
        if isinstance(employment, list):
            contract_type = normalize_whitespace(", ".join([str(e) for e in employment if e]))
        elif isinstance(employment, str):
            cleaned = employment.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            contract_type = (
                normalize_whitespace(", ".join(parts)) if parts else normalize_whitespace(cleaned)
            )
        return {
            "description": description,
            "extract_method": "jobvector_jsonld",
            "extract_warnings": [],
            "title": normalize_whitespace(job.get("title") or ""),
            "org": org,
            "location": location,
            "posted_at": posted_at,
            "deadline": deadline,
            "salary": salary,
            "education_level": education_level,
            "contract_type": contract_type,
        }
    if not enable_fallback:
        return {
            "description": "",
            "extract_method": "",
            "extract_warnings": ["jsonld_not_found"],
        }
    return _enrich_detail(
        html,
        selectors=selectors or {},
        min_len=min_len,
        enable_fallback=enable_fallback,
    )
