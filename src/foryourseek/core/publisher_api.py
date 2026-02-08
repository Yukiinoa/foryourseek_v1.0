from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from bs4 import BeautifulSoup

from .http import HttpClient
from .utils import normalize_doi, normalize_whitespace

_PII_RE = re.compile(r"/pii/([A-Z0-9]+)", re.IGNORECASE)
_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _extract_pii(url: str) -> str:
    if not url:
        return ""
    m = _PII_RE.search(url)
    return (m.group(1) if m else "") or ""


def _expand_env(value: str) -> str:
    def repl(m: re.Match) -> str:
        return os.getenv(m.group(1), "") or ""

    return _ENV_RE.sub(repl, value)


def _expand_mapping(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        if v is None:
            continue
        if isinstance(v, str):
            vv = _expand_env(v)
            if vv == "":
                continue
            out[k] = vv
        else:
            out[k] = v
    return out


def _get_json_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _find_abstract_in_json(obj: Any, paths: list[str]) -> str:
    for p in paths:
        val = _get_json_path(obj, p)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _clean_abstract(text: str) -> str:
    if not text:
        return ""
    cleaned = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return normalize_whitespace(cleaned)


def fetch_publisher_abstract(
    *,
    cfg: Dict[str, Any],
    doi: str,
    url: str,
    title: str,
    journal: str,
    http: HttpClient,
) -> Tuple[str, str]:
    """Fetch abstract via configurable publisher APIs.

    Returns (abstract, source_name).
    """
    providers = (cfg.get("papers", {}) or {}).get("abstract_providers") or []
    if not providers:
        return "", ""

    ctx = {
        "doi": normalize_doi(doi),
        "pii": _extract_pii(url),
        "url": url or "",
        "title": title or "",
        "journal": journal or "",
    }

    for prov in providers:
        if not prov or not prov.get("enabled", True):
            continue
        name = prov.get("name") or "publisher_api"
        reqs = prov.get("requests") or []
        headers = _expand_mapping(prov.get("headers"))
        paths = prov.get("response_paths") or ["abstract", "dc:description"]
        abstract_is_html = bool(prov.get("abstract_is_html", True))

        for req in reqs:
            if not req:
                continue
            url_t = _expand_env(req.get("url") or "")
            requires = req.get("requires") or []
            if any(not ctx.get(r) for r in requires):
                continue
            try:
                target = url_t.format_map(ctx)
            except Exception:
                continue
            if not target:
                continue
            if "${" in target or not target.startswith("http"):
                continue
            try:
                raw = http.get_text(target, headers=headers or None)
                data = json.loads(raw)
                abstract = _find_abstract_in_json(data, paths)
                if abstract:
                    return (
                        _clean_abstract(abstract)
                        if abstract_is_html
                        else normalize_whitespace(abstract)
                    ), name
            except Exception:
                continue

    return "", ""
