from __future__ import annotations

from typing import Any, Dict

try:
    from dateutil import parser as date_parser
except Exception:  # pragma: no cover
    date_parser = None

from .utils import normalize_doi, normalize_whitespace

_DATE_SKIP_TOKENS = (
    "open until",
    "open-ended",
    "open ended",
    "rolling",
    "tbd",
    "not specified",
)


def normalize_date(raw: str) -> str:
    raw = normalize_whitespace(raw)
    if not raw:
        return ""
    low = raw.lower()
    if any(tok in low for tok in _DATE_SKIP_TOKENS):
        return ""
    if date_parser is None:
        return ""
    try:
        dt = date_parser.parse(raw, fuzzy=True)
        return dt.date().isoformat()
    except Exception:
        return ""


def normalize_salary(raw: str) -> str:
    raw = normalize_whitespace(raw)
    if not raw:
        return ""
    return raw


def _split_tokens(raw: str) -> list[str]:
    raw = raw.replace("/", " ").replace(",", " ").replace(";", " ").replace("|", " ")
    raw = raw.replace("_", " ")
    return [t for t in raw.lower().split() if t]


def normalize_contract_type(raw: str) -> str:
    raw = normalize_whitespace(raw)
    if not raw or raw.isdigit():
        return ""
    tokens = _split_tokens(raw)
    labels: list[str] = []
    if "full" in tokens and "time" in tokens:
        labels.append("Full-time")
    if "part" in tokens and "time" in tokens:
        labels.append("Part-time")
    if any(t in tokens for t in ["temporary", "fixed", "fixed-term", "limited"]):
        labels.append("Temporary")
    if "permanent" in tokens:
        labels.append("Permanent")
    if any(t.startswith("intern") for t in tokens):
        labels.append("Internship")
    if not labels:
        return raw
    return ", ".join(dict.fromkeys(labels))


def normalize_education_level(raw: str, *, title: str = "", description: str = "") -> str:
    raw = normalize_whitespace(raw)
    if raw and raw.isdigit():
        raw = ""
    hay = f"{title}\n{description}\n{raw}".lower()
    if any(k in hay for k in ["phd", "ph.d", "doctoral"]):
        return "PhD"
    if any(k in hay for k in ["master", "msc", "m.sc", "m.s."]):
        return "Master"
    if any(k in hay for k in ["bachelor", "bsc", "b.sc", "b.s."]):
        return "Bachelor"
    return raw


def normalize_job_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    title = normalize_whitespace(item.get("title", ""))
    description = normalize_whitespace(item.get("description", ""))
    return {
        "title_norm": title,
        "org_norm": normalize_whitespace(item.get("org", "")),
        "location_norm": normalize_whitespace(item.get("location", "")),
        "deadline_norm": normalize_date(item.get("deadline", "")),
        "posted_at_norm": normalize_date(item.get("posted_at", "")),
        "salary_norm": normalize_salary(item.get("salary", "")),
        "contract_type_norm": normalize_contract_type(item.get("contract_type", "")),
        "education_level_norm": normalize_education_level(
            item.get("education_level", ""),
            title=title,
            description=description,
        ),
    }


def normalize_paper_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doi_norm": normalize_doi(item.get("doi", "")),
        "journal_norm": normalize_whitespace(item.get("journal", "")),
        "published_at_norm": normalize_date(item.get("published_at", "")),
    }
