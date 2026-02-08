from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ...core.utils import normalize_whitespace
from .generic_static import enrich_detail as _enrich_detail


def _text(node: Any) -> str:
    if not node:
        return ""
    return normalize_whitespace(node.get_text(" ", strip=True))


def _strip_prefix(text: str, prefix: str) -> str:
    if not text:
        return ""
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix) :].strip()
    return text


def _extract_deadline(text: str) -> str:
    if not text:
        return ""
    m = re.search(
        r"Application Deadline\\s*:?\\s*(.+?)(?:Research Field|Researcher Profile|Funding Programme|$)",
        text,
        flags=re.IGNORECASE,
    )
    return normalize_whitespace(m.group(1)) if m else ""


def fetch_list(
    url: str, html: str, *, selectors: Dict[str, str] | None = None, limit: int = 50
) -> List[Dict[str, Any]]:
    """EURAXESS list parser (job teaser blocks)."""
    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict[str, Any]] = []

    for teaser in soup.select("div#job-teaser-content"):
        title_node = teaser.select_one(
            "h3.ecl-content-block__title a, h1.ecl-content-block__title a"
        )
        if not title_node:
            continue
        title = _text(title_node)
        href = (title_node.get("href") or "").strip()
        if not href:
            continue
        full_url = urljoin(url, href)

        org = ""
        posted_at = ""
        for li in teaser.select("ul.ecl-content-block__primary-meta-container li"):
            t = _text(li)
            if not t:
                continue
            if t.lower().startswith("posted on"):
                posted_at = _strip_prefix(t, "Posted on:")
            elif not org:
                org = t

        location = _text(teaser.select_one("span.ecl-label--highlight"))
        desc = _text(teaser.select_one("div.ecl-content-block__description"))

        secondary_text = " ".join(
            _text(li) for li in teaser.select("ul.ecl-content-block__secondary-meta-container li")
        )
        deadline = _extract_deadline(secondary_text)

        items.append(
            {
                "title": title,
                "url": full_url,
                "org": org,
                "location": location,
                "posted_at": posted_at,
                "deadline": deadline,
                "description": desc,
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
    soup = BeautifulSoup(detail_html, "html.parser")

    title = _text(soup.select_one("h1.ecl-content-block__title"))

    info_map: Dict[str, str] = {}
    for dt in soup.select("dt.ecl-description-list__term"):
        key = _text(dt)
        dd = dt.find_next_sibling("dd")
        if key and dd:
            info_map[key] = _text(dd)

    org = info_map.get("Organisation/Company", "")
    deadline = info_map.get("Application Deadline", "")
    location = info_map.get("Country", "")
    department = info_map.get("Department", "")
    research_field = info_map.get("Research Field", "")
    researcher_profile = info_map.get("Researcher Profile", "")
    positions = info_map.get("Positions", "")
    education_level = info_map.get("Education Level", "")
    contract_type = info_map.get("Type of Contract", "")
    job_status = info_map.get("Job Status", "")
    funding_programme = info_map.get("Funding Programme", "")
    reference_number = info_map.get("Reference Number", "")
    languages = info_map.get("Languages", "")
    salary = (
        info_map.get("Salary", "")
        or info_map.get("Monthly Salary", "")
        or info_map.get("Gross Salary", "")
        or info_map.get("Net Salary", "")
        or info_map.get("Salary or benefits", "")
    )

    posted_at = ""
    for li in soup.select("li.ecl-content-block__primary-meta-item"):
        t = _text(li)
        if t.lower().startswith("posted on"):
            posted_at = _strip_prefix(t, "Posted on:")
            break

    description = ""
    desc_heading = soup.find("h2", id="offer-description")
    if desc_heading and desc_heading.parent:
        parent = desc_heading.parent
        content = parent.find("div", class_="ecl") or parent
        description = _text(content)
        description = _strip_prefix(description, "Offer Description")

    if not description or len(description) < min_len:
        fallback = _enrich_detail(
            detail_html,
            selectors=selectors,
            min_len=min_len,
            enable_fallback=enable_fallback,
        )
        if not description and fallback.get("description"):
            description = fallback.get("description", "")
        extract_method = fallback.get("extract_method", "") or "euraxess"
        extract_warnings = list(fallback.get("extract_warnings") or [])
    else:
        extract_method = "euraxess"
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
        "department": department,
        "research_field": research_field,
        "researcher_profile": researcher_profile,
        "positions": positions,
        "education_level": education_level,
        "contract_type": contract_type,
        "job_status": job_status,
        "funding_programme": funding_programme,
        "salary": salary,
        "reference_number": reference_number,
        "languages": languages,
        "url": "",
    }
