from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)

TRACKING_QUERY_KEYS = {
    "gclid",
    "fbclid",
    "dclid",
    "msclkid",
    "gclsrc",
    "gbraid",
    "wbraid",
    "yclid",
    "_hsenc",
    "_hsmi",
    "mc_cid",
    "mc_eid",
    "igshid",
    "mkt_tok",
    "spm",
    "spm_id_from",
}
TRACKING_QUERY_PREFIXES = ("utm_", "pk_")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def normalize_doi(raw: str) -> str:
    if not raw:
        return ""
    doi = raw.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "")
    doi = doi.strip().lower()
    return doi


def find_doi(text: str) -> str:
    if not text:
        return ""
    m = DOI_RE.search(text)
    return normalize_doi(m.group(0)) if m else ""


def _is_tracking_param(key: str) -> bool:
    if not key:
        return False
    k = key.lower()
    if k in TRACKING_QUERY_KEYS:
        return True
    return any(k.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)


def canonicalize_url(url: str) -> str:
    """Normalize URL for dedup: strip tracking params and fragments, trim trailing slash."""
    if not url:
        return ""
    url = url.strip()
    try:
        p = urlparse(url)
        # Drop fragment
        fragment = ""
        # Remove tracking params
        query_pairs = [
            (k, v)
            for k, v in parse_qsl(p.query, keep_blank_values=True)
            if not _is_tracking_param(k)
        ]
        query = urlencode(query_pairs)
        # Normalize path: remove trailing slash except root
        path = p.path or ""
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        np = p._replace(query=query, fragment=fragment, path=path)
        return urlunparse(np)
    except Exception:
        return url


def make_paper_id(doi: str, url: str, title: str) -> str:
    doi_n = normalize_doi(doi)
    if doi_n:
        return f"doi:{doi_n}"
    url_n = canonicalize_url(url)
    if url_n:
        return f"url:{sha1_hex(url_n)}"
    title_n = normalize_whitespace(title).lower()
    return f"title:{sha1_hex(title_n)}"


def make_paper_ids(doi: str, url: str, title: str) -> list[str]:
    ids: list[str] = []
    doi_n = normalize_doi(doi)
    if doi_n:
        ids.append(f"doi:{doi_n}")
    url_n = canonicalize_url(url)
    if url_n:
        ids.append(f"url:{sha1_hex(url_n)}")
    title_n = normalize_whitespace(title).lower()
    if title_n:
        ids.append(f"title:{sha1_hex(title_n)}")
    # De-dup while preserving order
    return list(dict.fromkeys(ids))


def make_job_id(url: str, semantic_key: str = "") -> str:
    semantic_key = normalize_whitespace(semantic_key).lower()
    if semantic_key:
        return f"sem:{sha1_hex(semantic_key)}"
    url_n = canonicalize_url(url)
    if url_n:
        return f"url:{sha1_hex(url_n)}"
    return f"sem:{sha1_hex(semantic_key)}"


def safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default
