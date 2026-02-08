from __future__ import annotations

import json
import re
import time
import traceback
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup

try:
    from dateutil import parser as date_parser
except Exception:
    date_parser = None

from ..core.db import Database, JobRow
from ..core.http import HttpClient
from ..core.llm import LLMClient
from ..core.logging import log_error, log_event
from ..core.schema import normalize_job_fields
from ..core.utils import canonicalize_url, make_job_id, normalize_whitespace


def _match_any(hay: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    h = hay.lower()
    return any(k.lower() in h for k in keywords if k)


def _match_required(hay: str, keywords: List[str]) -> bool:
    if not keywords:
        return True
    h = hay.lower()
    return any(k.lower() in h for k in keywords if k)


def _prefilter_score(
    title: str,
    text: str,
    keywords: List[str],
    *,
    title_weight: int,
    desc_weight: int,
    allow_if_no_keywords: bool,
) -> tuple[int, List[str], List[str]]:
    hay_title = (title or "").lower()
    hay_text = (text or "").lower()
    if not keywords:
        return (1 if allow_if_no_keywords else 0), [], []

    t_hits = [k for k in keywords if k and k.lower() in hay_title]
    d_hits = [k for k in keywords if k and k.lower() in hay_text]
    score = len(set(t_hits)) * int(title_weight) + len(set(d_hits)) * int(desc_weight)
    # keep order for display
    t_hits = list(dict.fromkeys(t_hits))
    d_hits = list(dict.fromkeys(d_hits))
    return score, t_hits, d_hits


_DEDUP_TITLE_STOPWORDS = {
    "phd",
    "ph.d",
    "doctoral",
    "doctorate",
    "position",
    "positions",
    "opening",
    "openings",
    "job",
    "jobs",
    "opportunity",
    "opportunities",
    "project",
    "program",
    "programme",
    "student",
    "studentship",
    "studentship",
    "candidate",
    "candidates",
    "fellow",
    "fellowship",
    "assistant",
    "associate",
    "intern",
    "internship",
    "researcher",
    "research",
}

_DEDUP_ORG_STOPWORDS = {
    "the",
    "of",
    "and",
    "university",
    "college",
    "school",
    "institute",
    "institution",
    "department",
    "dept",
    "faculty",
    "laboratory",
    "lab",
    "centre",
    "center",
}

_DEDUP_TITLE_SYNONYMS = [
    (r"\bgeo[-\s]?ai\b", "geospatial ai"),
]


def _build_degree_stopwords(degree_keywords: List[str]) -> set[str]:
    tokens: set[str] = set()
    for kw in degree_keywords:
        kw_n = normalize_whitespace(kw).lower()
        for part in kw_n.split():
            if part:
                tokens.add(part)
    return tokens


def _normalize_dedup_title(title: str, degree_stopwords: set[str]) -> str:
    text = normalize_whitespace(title).lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    for pattern, repl in _DEDUP_TITLE_SYNONYMS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t]
    stop = _DEDUP_TITLE_STOPWORDS | degree_stopwords
    filtered = [t for t in tokens if t not in stop]
    if not filtered:
        filtered = tokens
    return " ".join(filtered)


def _normalize_dedup_org(org: str) -> str:
    text = normalize_whitespace(org).lower()
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t]
    filtered = [t for t in tokens if t not in _DEDUP_ORG_STOPWORDS]
    if not filtered:
        filtered = tokens
    return " ".join(filtered)


def _normalize_dedup_location(location: str) -> str:
    text = normalize_whitespace(location).lower()
    if not text:
        return ""
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return " ".join(tokens)


def _build_search_keywords(search_cfg: Dict[str, Any], profile: Dict[str, Any]) -> str:
    raw = search_cfg.get("keywords")
    if not raw:
        raw = profile.get("include_keywords", [])
    if isinstance(raw, str):
        raw = [raw]
    keywords = [str(k).strip() for k in (raw or []) if str(k).strip()]
    if not keywords:
        return ""
    op = str(search_cfg.get("keyword_operator") or "OR").upper()
    if op == "AND":
        joiner = " AND "
    elif op == "SPACE":
        joiner = " "
    else:
        joiner = " OR "
    return joiner.join(keywords)


_ETH_CSRF_RE = re.compile(r"apiCsrfToken\s*[:=]\s*['\"]([^'\"]+)['\"]")
_ACADEMICTRANSFER_TOKEN_INDEX_RE = re.compile(r"\$satDataApiPublicAccessToken\":(\d+)")


def _eth_gethired_root(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def _eth_gethired_token(html: str) -> str:
    m = _ETH_CSRF_RE.search(html or "")
    return m.group(1) if m else ""


def _eth_gethired_list_paging(
    http: HttpClient,
    root: str,
    *,
    token: str,
    page_index: int,
    page_size: int,
) -> Dict[str, Any]:
    url = urljoin(root, "Umbraco/iTalent/Jobs/ListPaging")
    payload = {"paging": {"pageIndex": page_index, "pageSize": page_size}}
    headers = {
        "User-Agent": http.user_agent,
        "RequestVerificationToken": token,
        "Content-Type": "application/json",
    }
    resp = http.session.post(url, json=payload, headers=headers, timeout=http.timeout_sec)
    resp.raise_for_status()
    return resp.json()


def _eth_gethired_detail(
    http: HttpClient,
    root: str,
    *,
    token: str,
    job_id: str,
) -> Dict[str, Any]:
    url = urljoin(root, f"Umbraco/iTalent/Jobs/Detail?id={job_id}")
    headers = {"RequestVerificationToken": token}
    resp = http.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()


def _eth_match_filter(obj: Any, wanted: List[str]) -> bool:
    if not wanted:
        return True
    if not isinstance(obj, dict):
        return False
    obj_id = str(obj.get("id") or "").lower()
    obj_name = str(obj.get("name") or "").lower()
    for w in wanted:
        w_norm = str(w or "").strip().lower()
        if not w_norm:
            continue
        if w_norm == obj_id or w_norm == obj_name:
            return True
    return False


def _academictransfer_extract_token(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", attrs={"type": "application/json"})
    if not scripts:
        return ""
    payload = max(scripts, key=lambda s: len(s.get_text() or "")).get_text() or ""
    if not payload:
        return ""
    m = _ACADEMICTRANSFER_TOKEN_INDEX_RE.search(payload)
    if not m:
        return ""
    try:
        idx = int(m.group(1))
        data = json.loads(payload)
        if idx < 0 or idx >= len(data):
            return ""
        token = data[idx]
        return str(token) if token else ""
    except Exception:
        return ""


def _academictransfer_build_queries(
    base_url: str,
    search_cfg: Dict[str, Any],
    profile: Dict[str, Any],
    jobs_cfg: Dict[str, Any],
) -> List[str]:
    query_mode = str(search_cfg.get("query_mode") or "combined").lower()

    base_query = str(search_cfg.get("query") or "").strip()
    if not base_query:
        q = dict(parse_qsl(urlparse(base_url).query)).get("q", "")
        base_query = str(q or "").strip()

    if query_mode == "per_keyword":
        raw = search_cfg.get("keywords")
        if not raw:
            raw = jobs_cfg.get("domain_keywords") or profile.get("include_keywords") or []
        if isinstance(raw, str):
            if "," in raw:
                queries = [p.strip() for p in raw.split(",") if p.strip()]
            else:
                queries = [raw.strip()] if raw.strip() else []
        else:
            queries = [str(k).strip() for k in raw if str(k).strip()]
        if not queries and base_query:
            queries = [base_query]
    else:
        if base_query:
            query = base_query
        else:
            raw = search_cfg.get("keywords")
            if not raw:
                raw = jobs_cfg.get("domain_keywords") or profile.get("include_keywords") or []
            if isinstance(raw, str):
                query = raw.strip()
            else:
                query = " ".join([str(k).strip() for k in raw if str(k).strip()])
        queries = [query] if query else [""]

    if search_cfg.get("require_phd"):
        expanded = []
        for q in queries:
            if "phd" in q.lower():
                expanded.append(q)
            else:
                expanded.append(f"{q} phd".strip())
        queries = expanded

    return list(dict.fromkeys([q for q in queries if q is not None]))


def _academictransfer_payload_text(payload: Dict[str, Any]) -> str:
    parts = [
        payload.get("description") or "",
        payload.get("requirements") or "",
        payload.get("contract_terms") or "",
        payload.get("additional_info") or "",
        payload.get("organisation_description") or "",
        payload.get("department_description") or "",
        payload.get("extra_info") or "",
    ]
    html = "\n".join([p for p in parts if p])
    if not html:
        return ""
    return normalize_whitespace(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))


def _normalize_academictransfer_meta(
    education_level: str,
    contract_type: str,
    *,
    title: str,
    description: str,
) -> tuple[str, str]:
    edu = normalize_whitespace(education_level or "")
    contract = normalize_whitespace(contract_type or "")
    if edu.isdigit():
        hay = f"{title} {description}".lower()
        edu = "PhD" if "phd" in hay else ""
    if contract.isdigit():
        contract = ""
    return edu, contract


def _academictransfer_match_structured(item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True

    def _norm(val: Any) -> str:
        if val is None:
            return ""
        if isinstance(val, bool):
            return "true" if val else "false"
        return str(val).strip().lower()

    for key, wanted in filters.items():
        if wanted is None or wanted == "" or wanted == []:
            continue
        raw = item.get(key)
        if raw is None:
            return False
        wanted_list = wanted if isinstance(wanted, list) else [wanted]
        wanted_norm = {_norm(w) for w in wanted_list if _norm(w)}
        if not wanted_norm:
            continue
        if isinstance(raw, list):
            raw_norm = {_norm(v) for v in raw if _norm(v)}
            if not raw_norm.intersection(wanted_norm):
                return False
        else:
            if _norm(raw) not in wanted_norm:
                return False
    return True


def _academictransfer_candidates(
    http: HttpClient,
    base_url: str,
    *,
    search_cfg: Dict[str, Any],
    profile: Dict[str, Any],
    jobs_cfg: Dict[str, Any],
    max_candidates: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    html = http.get_text_smart(base_url)
    token = _academictransfer_extract_token(html)
    if not token:
        raise RuntimeError("academictransfer: missing public access token")
    api_base = str(search_cfg.get("api_base_url") or "https://api.academictransfer.com").rstrip("/")
    queries = _academictransfer_build_queries(base_url, search_cfg, profile, jobs_cfg)
    limit = int(search_cfg.get("page_size") or 25)
    if limit <= 0:
        limit = 25
    structured_filters = search_cfg.get("structured_filters") or {}
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json; version=2",
    }
    candidates: List[Dict[str, Any]] = []
    page_count = 1
    pages_fetched = 0
    seen_ids: set[str] = set()

    per_query_cap_raw = search_cfg.get("max_candidates_per_query")
    if per_query_cap_raw is None:
        per_query_cap = 0
    else:
        per_query_cap = int(per_query_cap_raw)
        if per_query_cap < 0:
            per_query_cap = 0

    for q in queries:
        per_query_count = 0
        offset = 0
        while True:
            params = {
                "limit": limit,
                "offset": offset,
            }
            if q:
                params["q"] = q
            resp = http.session.get(
                f"{api_base}/vacancies/",
                headers=headers,
                params=params,
                timeout=http.timeout_sec,
            )
            resp.raise_for_status()
            payload = resp.json()
            pages_fetched += 1
            count = int(payload.get("count") or 0)
            if count and limit:
                page_count = max(page_count, (count + limit - 1) // limit)

            results = payload.get("results") or []
            for item in results:
                if structured_filters and not _academictransfer_match_structured(
                    item, structured_filters
                ):
                    continue
                external_id = str(item.get("external_id") or "")
                internal_id = str(item.get("id") or "")
                key = external_id or internal_id
                if not key or key in seen_ids:
                    continue
                seen_ids.add(key)
                title = normalize_whitespace(item.get("title") or "")
                org = normalize_whitespace(item.get("organisation_name") or "")
                city = normalize_whitespace(item.get("city") or "")
                country = normalize_whitespace(item.get("country_code") or "")
                location = ", ".join([p for p in [city, country] if p])
                posted_at = normalize_whitespace(item.get("created_datetime") or "")
                deadline = normalize_whitespace(item.get("end_date") or "")
                description = _academictransfer_payload_text(item)

                min_salary = item.get("min_salary")
                max_salary = item.get("max_salary")
                salary = ""
                if min_salary or max_salary:
                    salary = (
                        f"{min_salary}-{max_salary}"
                        if min_salary and max_salary
                        else f"{min_salary or max_salary}"
                    )

                department = normalize_whitespace(item.get("department_name") or "")
                research_field = ""
                if item.get("research_fields"):
                    research_field = ", ".join([str(x) for x in item.get("research_fields") or []])
                education_level = str(item.get("education_level") or "")
                contract_type = str(item.get("contract_type") or "")
                positions = str(item.get("available_positions") or "")

                url = item.get("absolute_url") or ""
                if not url and external_id:
                    url = f"https://www.academictransfer.com/en/jobs/{external_id}/"

                candidates.append(
                    {
                        "title": title,
                        "url": url,
                        "org": org,
                        "location": location,
                        "deadline": deadline,
                        "posted_at": posted_at,
                        "description": description,
                        "department": department,
                        "research_field": research_field,
                        "education_level": education_level,
                        "contract_type": contract_type,
                        "salary": salary,
                        "positions": positions,
                        "at_payload": item,
                    }
                )
                per_query_count += 1
                if max_candidates > 0 and len(candidates) >= max_candidates:
                    break
                if per_query_cap > 0 and per_query_count >= per_query_cap:
                    break
            if max_candidates > 0 and len(candidates) >= max_candidates:
                break
            if per_query_cap > 0 and per_query_count >= per_query_cap:
                break
            if not payload.get("next"):
                break
            offset += limit
        if max_candidates > 0 and len(candidates) >= max_candidates:
            break

    ctx = {
        "token": token,
        "api_base": api_base,
        "page_count": page_count,
        "pages_fetched": pages_fetched,
    }
    return candidates, ctx


def _eth_to_list(val: Any) -> List[str]:
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    return []


def _eth_gethired_candidates(
    http: HttpClient,
    base_url: str,
    *,
    search_cfg: Dict[str, Any],
    max_candidates: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    html = http.get_text_smart(base_url)
    token = _eth_gethired_token(html)
    if not token:
        raise RuntimeError("eth_gethired: missing apiCsrfToken")
    root = _eth_gethired_root(base_url)

    jobtype_filters = _eth_to_list(search_cfg.get("jobtypes"))
    jobtypesub_filters = _eth_to_list(search_cfg.get("jobtypessub") or search_cfg.get("jobtypesub"))
    page_size = int(search_cfg.get("page_size") or 25)
    if page_size <= 0:
        page_size = 25

    candidates: List[Dict[str, Any]] = []
    page_index = 1
    page_count = 1
    pages_fetched = 0
    while page_index <= page_count:
        payload = _eth_gethired_list_paging(
            http,
            root,
            token=token,
            page_index=page_index,
            page_size=page_size,
        )
        pages_fetched += 1
        paging = payload.get("paging") or {}
        page_count = int(paging.get("pageCount") or page_count or 1)
        results = payload.get("results") or []
        for job in results:
            if jobtype_filters and not _eth_match_filter(job.get("jobtype"), jobtype_filters):
                continue
            if jobtypesub_filters and not _eth_match_filter(
                job.get("jobtypesub"), jobtypesub_filters
            ):
                continue

            job_id = str(job.get("id") or "").strip()
            if not job_id:
                continue
            title = normalize_whitespace(job.get("name") or "")
            org = normalize_whitespace(job.get("customerDisplayname") or "")
            region = job.get("region") or {}
            location = normalize_whitespace(region.get("name") or "")
            posted_at = normalize_whitespace(job.get("dateStart") or "")
            public_url = urljoin(root, f"/en/jobs/details/?id={job_id}")
            candidates.append(
                {
                    "title": title,
                    "url": public_url,
                    "org": org,
                    "location": location,
                    "posted_at": posted_at,
                    "deadline": "",
                    "eth_id": job_id,
                }
            )
            if max_candidates > 0 and len(candidates) >= max_candidates:
                break
        if max_candidates > 0 and len(candidates) >= max_candidates:
            break
        if not results:
            break
        page_index += 1

    ctx = {
        "token": token,
        "root": root,
        "page_count": page_count,
        "pages_fetched": pages_fetched,
        "page_size": page_size,
    }
    return candidates, ctx


def _prepare_euraxess_search(
    html: str,
    base_url: str,
    *,
    keywords: str,
    search_cfg: Dict[str, Any],
) -> tuple[str, Dict[str, str]] | tuple[None, None]:
    soup = BeautifulSoup(html, "html.parser")
    form = None
    for f in soup.find_all("form"):
        if f.find("input", {"name": "keywords"}):
            form = f
            break
    if not form:
        return None, None
    action = (form.get("action") or "").strip() or base_url
    action_url = urljoin(base_url, action)
    payload: Dict[str, str] = {}
    for inp in form.find_all("input"):
        name = inp.get("name")
        if not name or name == "keywords":
            continue
        payload[name] = str(inp.get("value") or "")
    search_entity = str(search_cfg.get("search_entity") or "").strip()
    if not search_entity:
        sel = form.find("select", {"name": "search_entity"})
        if sel:
            for opt in sel.find_all("option"):
                if "jobs" in (opt.get_text() or "").lower():
                    search_entity = str(opt.get("value") or "").strip()
                    break
    if not search_entity:
        search_entity = "13934"
    payload["search_entity"] = search_entity
    payload["keywords"] = keywords
    return action_url, payload


def _extract_filter_query(html: str) -> str:
    params: List[tuple[str, str]] = []
    seen = set()
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = str(a.get("href") or "")
        if "f%5B" not in href and "f[" not in href:
            continue
        q = urlparse(href).query
        if not q and "?" in href:
            q = href.split("?", 1)[1]
        for k, v in parse_qsl(q, keep_blank_values=True):
            if not k.startswith("f["):
                continue
            key = (k, v)
            if key in seen:
                continue
            seen.add(key)
            params.append(key)
    if params:
        return urlencode(params, doseq=True)

    matches = re.findall(r"f%5B\\d+%5D=[^&\"'\\s]+", html)
    if matches:
        q = "&".join(matches)
        params = [(k, v) for k, v in parse_qsl(q, keep_blank_values=True) if k.startswith("f[")]
        if params:
            return urlencode(params, doseq=True)
    return ""


def _fetch_detail_html(
    http: HttpClient,
    url: str,
    *,
    max_attempts: int = 3,
    backoff_sec: float = 1.5,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        resp = http.get(url, headers=http._browser_headers)
        if resp.status_code == 429:
            time.sleep(backoff_sec * (2**attempt))
            last_exc = RuntimeError(f"rate_limited:{resp.status_code}")
            continue
        try:
            resp.raise_for_status()
        except Exception as ex:
            last_exc = ex
            break
        resp.encoding = resp.encoding or "utf-8"
        return resp.text
    if last_exc:
        raise last_exc
    raise RuntimeError("detail_fetch_failed")


def _deadline_is_active(raw: str, *, allow_without_deadline: bool) -> bool:
    if not raw:
        return allow_without_deadline
    low = raw.lower().strip()
    if any(
        x in low
        for x in [
            "open until",
            "open-ended",
            "open ended",
            "rolling",
            "tbd",
            "not specified",
        ]
    ):
        return allow_without_deadline
    if date_parser is None:
        return True
    today = datetime.now(timezone.utc)
    try:
        dt = date_parser.parse(
            raw,
            fuzzy=True,
            default=datetime(today.year, today.month, today.day, tzinfo=today.tzinfo),
        )
        return dt.date() >= today.date()
    except Exception:
        return allow_without_deadline


def _load_adapter(adapter_name: str):
    mod = import_module(f"foryourseek.adapters.jobs.{adapter_name}")
    return mod


def _set_query_param(url: str, key: str, value: Any, extra: Dict[str, Any] | None = None) -> str:
    p = urlparse(url)
    query = dict(parse_qsl(p.query, keep_blank_values=True))
    query[key] = str(value)
    for k, v in (extra or {}).items():
        if v is None:
            continue
        query[k] = str(v)
    return urlunparse(p._replace(query=urlencode(query)))


def _build_page_urls(base_url: str, pagination: Dict[str, Any] | None) -> List[str]:
    if not pagination:
        return [base_url]
    if isinstance(pagination, list):
        return [str(u) for u in pagination if u]

    explicit = pagination.get("page_urls") or []
    if explicit:
        return [str(u) for u in explicit if u]

    template = (pagination.get("url_template") or "").strip()
    if template:
        start = int(pagination.get("start", 1))
        end = pagination.get("end")
        pages = pagination.get("pages")
        if end is None and pages is not None:
            end = start + int(pages) - 1
        if end is None:
            end = start
        return [template.format(page=i) for i in range(start, int(end) + 1)]

    page_param = (pagination.get("page_param") or "").strip()
    if page_param:
        start = int(pagination.get("start", 1))
        end = pagination.get("end")
        pages = pagination.get("pages")
        if end is None and pages is not None:
            end = start + int(pages) - 1
        if end is None:
            end = start
        extra = {}
        size_param = pagination.get("page_size_param")
        size_value = pagination.get("page_size")
        if size_param and size_value is not None:
            extra[str(size_param)] = size_value
        return [
            _set_query_param(base_url, page_param, i, extra=extra)
            for i in range(start, int(end) + 1)
        ]

    return [base_url]


def _extract_page_numbers(
    base_url: str,
    html: str,
    *,
    page_param: str = "",
    max_pages: int | None = None,
) -> Tuple[str, int, Dict[str, Any]]:
    """Return (page_param, max_page, extra_params) discovered from pagination links."""
    soup = BeautifulSoup(html, "html.parser")
    base_parts = urlparse(base_url)
    base_path = base_parts.path.rstrip("/") or "/"

    base_prefix = base_path.rstrip("/")
    if base_prefix == "/":
        base_prefix = ""

    def _path_page_num(href: str) -> Tuple[int | None, bool]:
        full = urljoin(base_url, href)
        parts = urlparse(full)
        path = parts.path or ""
        path_no = path.rstrip("/")
        if base_prefix:
            if not path_no.startswith(base_prefix + "/"):
                return None, False
            remainder = path_no[len(base_prefix) + 1 :]
        else:
            if not path_no.startswith("/"):
                return None, False
            remainder = path_no[1:]
        if remainder.isdigit():
            return int(remainder), path.endswith("/")
        return None, False

    link_hrefs: List[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a.get("href") or "").strip()
        if not href:
            continue
        if page_param:
            if page_param not in href and _path_page_num(href)[0] is None:
                continue
        else:
            if (
                "page" not in href
                and "p=" not in href
                and "pg=" not in href
                and _path_page_num(href)[0] is None
            ):
                continue
        link_hrefs.append(href)

    # also include rel=next/last if present
    for link_tag in soup.find_all("link", href=True):
        rel = link_tag.get("rel") or []
        if any(str(r).lower() in ("next", "last") for r in rel):
            href = str(link_tag.get("href") or "").strip()
            if href:
                link_hrefs.append(href)

    candidates: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    param_keys = [page_param] if page_param else ["page", "p", "pg"]

    for href in link_hrefs:
        full = urljoin(base_url, href)
        parts = urlparse(full)
        path = parts.path.rstrip("/") or "/"
        if path != base_path:
            continue
        q = dict(parse_qsl(parts.query, keep_blank_values=True))
        for key in param_keys:
            if key in q:
                try:
                    page_num = int(str(q.get(key) or "").strip())
                except Exception:
                    continue
                extra = {k: v for k, v in q.items() if k != key}
                candidates.setdefault(key, []).append((page_num, extra))

    if not candidates:
        path_pages: List[int] = []
        trailing = False
        for href in link_hrefs:
            page_num, has_trailing = _path_page_num(href)
            if page_num is None:
                continue
            path_pages.append(page_num)
            trailing = trailing or has_trailing
        if not path_pages:
            return "", 0, {}
        return "__path__", max(path_pages), {"path_trailing_slash": trailing}

    # pick the key with the highest max page
    best_key = ""
    best_max = 1
    best_extra: Dict[str, Any] = {}
    for key, vals in candidates.items():
        max_page = max(v[0] for v in vals) if vals else 1
        if max_page > best_max:
            best_key = key
            best_max = max_page
            best_extra = vals[0][1] if vals else {}

    if max_pages is not None and max_pages > 0:
        best_max = min(best_max, int(max_pages))

    return best_key or (page_param or "page"), best_max, best_extra


def _build_page_urls_auto(base_url: str, html: str, pagination: Dict[str, Any] | None) -> List[str]:
    pagination = pagination or {}
    page_param = str(pagination.get("page_param") or "").strip()
    max_pages = pagination.get("max_pages")
    key, max_page, extra = _extract_page_numbers(
        base_url,
        html,
        page_param=page_param,
        max_pages=int(max_pages) if max_pages else None,
    )
    start = int(pagination.get("start", 1))
    if max_page < start or not key:
        return [base_url]
    if key == "__path__":
        base_parts = urlparse(base_url)
        base_path = base_parts.path.rstrip("/")
        if base_path == "/":
            base_path = ""
        root = f"{base_parts.scheme}://{base_parts.netloc}"
        trailing = bool(extra.get("path_trailing_slash", True))
        urls: List[str] = []
        last_seg = base_path.rsplit("/", 1)[-1] if base_path else ""
        base_has_page = last_seg.isdigit()
        if not base_has_page and start <= 1:
            urls.append(base_url)
        page_start = start if base_has_page else max(start, 2)
        for i in range(page_start, max_page + 1):
            path = f"{base_path}/{i}"
            if not path.startswith("/"):
                path = "/" + path
            if trailing:
                path = path.rstrip("/") + "/"
            urls.append(root + path)
        return urls
    return [_set_query_param(base_url, key, i, extra=extra) for i in range(start, max_page + 1)]


def _extract_total_count(
    html: str,
    *,
    pattern: str = "",
    selector: str = "",
) -> int:
    text = ""
    if selector:
        try:
            soup = BeautifulSoup(html, "html.parser")
            node = soup.select_one(selector)
            if node:
                text = node.get_text(" ", strip=True)
        except Exception:
            text = ""
    hay = text if text else html
    if pattern:
        m = re.search(pattern, hay, flags=re.IGNORECASE)
        if m:
            if m.groupdict():
                val = m.groupdict().get("count") or ""
            else:
                val = m.group(1) if m.groups() else m.group(0)
            digits = re.sub(r"[^0-9]", "", val)
            if digits:
                return int(digits)
    return 0


def _count_items(html: str, selector: str) -> int:
    if not selector:
        return 0
    try:
        soup = BeautifulSoup(html, "html.parser")
        return len(soup.select(selector))
    except Exception:
        return 0


def _build_page_urls_count(
    base_url: str, html: str, pagination: Dict[str, Any] | None
) -> List[str]:
    pagination = pagination or {}
    page_param = str(pagination.get("page_param") or "").strip() or "page"
    start = int(pagination.get("start", 1))
    max_pages = pagination.get("max_pages")
    total = _extract_total_count(
        html,
        pattern=str(pagination.get("count_pattern") or ""),
        selector=str(pagination.get("count_selector") or ""),
    )
    page_size = int(pagination.get("page_size") or 0)
    if page_size <= 0:
        page_size = _count_items(html, str(pagination.get("item_selector") or ""))
    if total <= 0 or page_size <= 0:
        return [base_url]
    max_page = (total + page_size - 1) // page_size
    if max_pages:
        max_page = min(max_page, int(max_pages))
    if max_page < start:
        return [base_url]
    return [_set_query_param(base_url, page_param, i) for i in range(start, max_page + 1)]


def run_job_agent(
    *,
    run_id: str,
    cfg: Dict[str, Any],
    db: Database,
    http: HttpClient,
    llm: LLMClient,
) -> Dict[str, Any]:
    jobs_cfg = cfg["jobs"]
    extract_cfg = cfg.get("extract", {})
    min_len = int(extract_cfg.get("min_len", 120))
    enable_fallback = extract_cfg.get("fallback", "trafilatura") != "none"

    max_per_day = int(jobs_cfg.get("max_per_day", 40))
    if "max_candidates_per_run" in jobs_cfg:
        max_candidates = int(jobs_cfg.get("max_candidates_per_run") or 0)
    else:
        max_candidates = 0
    max_llm = int(jobs_cfg.get("max_llm_per_run", max_per_day))
    debug_cfg = jobs_cfg.get("debug", {}) or {}
    export_candidates = bool(debug_cfg.get("export_candidates", False))
    candidates_limit = int(debug_cfg.get("candidates_limit", 0) or 0)
    export_audit = bool(debug_cfg.get("export_audit", export_candidates))
    audit_limit = int(debug_cfg.get("audit_limit", candidates_limit) or 0)
    resurface_cfg = jobs_cfg.get("resurface", {}) or {}
    resurface_mode = (resurface_cfg.get("mode") or "never").lower()
    if resurface_mode not in ("never", "unexpired", "always"):
        resurface_mode = "never"
    allow_without_deadline = bool(resurface_cfg.get("allow_without_deadline", True))
    pre_cfg = jobs_cfg.get("prefilter", {}) or {}
    pre_title_w = int(pre_cfg.get("title_weight", 3))
    pre_desc_w = int(pre_cfg.get("desc_weight", 1))
    pre_min_score = int(pre_cfg.get("min_score_for_llm", 2))
    pre_allow_no_kw = bool(pre_cfg.get("allow_if_no_keywords", True))
    pre_keywords = pre_cfg.get("keywords") or []
    degree_kw = jobs_cfg.get("degree_keywords", [])
    exclude_kw = jobs_cfg.get("exclude_keywords", [])
    degree_stopwords = _build_degree_stopwords(degree_kw)
    output_language = (getattr(llm, "output_language", "zh") or "zh").lower()
    if not pre_keywords:
        pre_keywords = (jobs_cfg.get("domain_keywords") or []) + (
            cfg.get("profile", {}).get("include_keywords") or []
        )
    pre_keywords = list(dict.fromkeys([k for k in pre_keywords if k]))

    dedup_cfg = jobs_cfg.get("dedup", {"method": "url"})
    dedup_method = (dedup_cfg.get("method") or "url").lower()
    sem_fields = dedup_cfg.get("semantic_key_fields", ["title", "org", "location", "deadline"])

    found = 0
    new_count = 0
    budget_hit_count = 0
    llm_input_tokens_sum = 0
    llm_output_tokens_sum = 0
    llm_total_tokens_sum = 0
    llm_usage_by_provider: Dict[str, Dict[str, Any]] = {}
    new_items_for_email: List[Dict[str, Any]] = []
    site_stats: Dict[str, Dict[str, Any]] = {}
    seen_skipped_total = 0
    rule_skipped_total = 0
    rule_skipped_exclude = 0
    rule_skipped_degree = 0
    llm_used_total = 0
    prefilter_candidates_total = 0
    prefilter_eligible_total = 0
    prefilter_skipped_total = 0
    final_items: List[Dict[str, Any]] = []
    prefilter_candidates: List[Dict[str, Any]] = []
    processed_total = 0
    seen_job_ids_run: set[str] = set()
    candidates_preview: List[Dict[str, Any]] = []

    def _bump_usage(
        provider: str, model: str, in_tokens: int, out_tokens: int, total_tokens: int
    ) -> None:
        if not provider:
            return
        entry = llm_usage_by_provider.setdefault(
            provider,
            {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "models": {},
            },
        )
        entry["calls"] += 1
        entry["input_tokens"] += int(in_tokens or 0)
        entry["output_tokens"] += int(out_tokens or 0)
        entry["total_tokens"] += int(total_tokens or 0)
        if model:
            models = entry.setdefault("models", {})
            models[model] = int(models.get(model, 0)) + 1

    def _build_semantic_key(
        values: Dict[str, Any], title_key: str, org_key: str, loc_key: str
    ) -> str:
        parts: List[str] = []
        for f in sem_fields:
            if f == "title":
                parts.append(title_key)
            elif f == "org":
                parts.append(org_key)
            elif f == "location":
                parts.append(loc_key)
            else:
                parts.append(normalize_whitespace(str(values.get(f, ""))))
        return "|".join(parts)

    for site in jobs_cfg.get("sites", []):
        site_name = site.get("name") or "(unknown)"
        url = site.get("url") or ""
        adapter_name = site.get("adapter") or "generic_static"
        selectors = site.get("selectors") or {}

        if not url:
            continue

        site_stat = {
            "site": site_name,
            "url": url,
            "adapter": adapter_name,
            "pages": 1,
            "pages_fetched": 0,
            "page_urls_sample": [],
            "candidates": 0,
            "processed": 0,
            "new": 0,
            "seen_skipped": 0,
            "rule_skipped_total": 0,
            "rule_skipped": {"exclude": 0, "degree": 0},
            "prefilter_candidates": 0,
            "prefilter_selected": 0,
            "prefilter_skipped": 0,
            "llm_used": 0,
            "truncated_by_max_candidates": False,
            "sample_titles": [],
            "error": "",
        }

        try:
            pagination_cfg = site.get("pagination") or {}
            search_cfg = site.get("search") or {}
            base_url = url
            is_eth_gethired = (adapter_name == "eth_gethired") or (
                str(search_cfg.get("method") or "").lower() == "eth_gethired"
            )
            is_academictransfer = (adapter_name == "academictransfer") or (
                str(search_cfg.get("method") or "").lower() == "academictransfer"
            )
            eth_ctx: Dict[str, Any] | None = None
            at_ctx: Dict[str, Any] | None = None
            page_urls: List[str] = []
            page_html_cache: Dict[str, str] = {}
            first_html = ""
            used_search = False
            filter_url = str(search_cfg.get("filter_url") or "").strip()
            if filter_url:
                base_url = filter_url
                first_html = http.get_text(filter_url)
                page_html_cache[base_url] = first_html
                used_search = True
            elif search_cfg and (search_cfg.get("method") or "").lower() == "post":
                search_keywords = _build_search_keywords(search_cfg, cfg.get("profile", {}))
                if search_keywords:
                    seed_html = http.get_text(url)
                    action_url, payload = _prepare_euraxess_search(
                        seed_html,
                        url,
                        keywords=search_keywords,
                        search_cfg=search_cfg,
                    )
                    if action_url and payload:
                        first_html = http.post_text(
                            action_url, data=payload, headers=http._browser_headers
                        )
                        filter_query = _extract_filter_query(first_html)
                        if not filter_query:
                            filter_query = urlencode([("f[0]", f"keywords:{search_keywords}")])
                        if filter_query:
                            base_url = url + ("&" if "?" in url else "?") + filter_query
                        page_html_cache[base_url] = first_html
                        used_search = True
            if is_eth_gethired:
                candidates, eth_ctx = _eth_gethired_candidates(
                    http,
                    base_url,
                    search_cfg=search_cfg,
                    max_candidates=max_candidates,
                )
                site_stat["pages"] = int(eth_ctx.get("page_count") or 1)
                site_stat["pages_fetched"] = int(eth_ctx.get("pages_fetched") or 0)
                site_stat["page_urls_sample"] = [
                    f"{eth_ctx.get('root')}Umbraco/iTalent/Jobs/ListPaging?pageIndex=1",
                    f"{eth_ctx.get('root')}Umbraco/iTalent/Jobs/ListPaging?pageIndex=2",
                ]
            elif is_academictransfer:
                candidates, at_ctx = _academictransfer_candidates(
                    http,
                    base_url,
                    search_cfg=search_cfg,
                    profile=cfg.get("profile", {}),
                    jobs_cfg=jobs_cfg,
                    max_candidates=max_candidates,
                )
                site_stat["pages"] = int(at_ctx.get("page_count") or 1)
                site_stat["pages_fetched"] = int(at_ctx.get("pages_fetched") or 0)
                at_queries = _academictransfer_build_queries(
                    base_url, search_cfg, cfg.get("profile", {}), jobs_cfg
                )
                sample_q = at_queries[0] if at_queries else ""
                site_stat["page_urls_sample"] = [
                    f"{at_ctx.get('api_base')}/vacancies/?q={sample_q}",
                ]
            elif (pagination_cfg.get("mode") or "").lower() == "count":
                if not first_html:
                    first_html = http.get_text(base_url)
                    page_html_cache[base_url] = first_html
                page_urls = _build_page_urls_count(base_url, first_html, pagination_cfg)
            elif (pagination_cfg.get("mode") or "").lower() == "auto" or pagination_cfg.get("auto"):
                if not first_html:
                    first_html = http.get_text(base_url)
                    page_html_cache[base_url] = first_html
                page_urls = _build_page_urls_auto(base_url, first_html, pagination_cfg)
                if not page_urls:
                    page_urls = [base_url]
                if used_search and base_url not in page_urls:
                    page_urls = [base_url] + page_urls
                if page_urls and page_urls[0] not in page_html_cache and first_html:
                    page_html_cache[page_urls[0]] = first_html
            else:
                page_urls = _build_page_urls(base_url, pagination_cfg)
                if used_search and base_url not in page_urls:
                    page_urls = [base_url] + page_urls
            if not is_eth_gethired and not is_academictransfer:
                site_stat["pages"] = len(page_urls)
                site_stat["page_urls_sample"] = page_urls[:3]

                adapter = _load_adapter(adapter_name)
                candidates = []
                seen_candidate_urls = set()
                list_limit = max_candidates if max_candidates > 0 else 10000
                for page_url in page_urls:
                    html = page_html_cache.get(page_url)
                    if html is None:
                        html = http.get_text(page_url)
                    site_stat["pages_fetched"] += 1
                    page_candidates = adapter.fetch_list(
                        page_url, html, selectors=selectors, limit=list_limit
                    )
                    for c in page_candidates:
                        raw_url = c.get("url", "") or ""
                        url_key = canonicalize_url(raw_url)
                        if url_key and url_key in seen_candidate_urls:
                            continue
                        if url_key:
                            seen_candidate_urls.add(url_key)
                        candidates.append(c)
                        if max_candidates > 0 and len(candidates) >= max_candidates:
                            site_stat["truncated_by_max_candidates"] = True
                            break
                    if max_candidates > 0 and len(candidates) >= max_candidates:
                        break
            else:
                adapter = _load_adapter(adapter_name)
                if max_candidates > 0 and len(candidates) >= max_candidates:
                    site_stat["truncated_by_max_candidates"] = True

            site_stat["candidates"] = len(candidates)
            found += len(candidates)

            if export_candidates:
                for c in candidates:
                    if candidates_limit > 0 and len(candidates_preview) >= candidates_limit:
                        break
                    candidates_preview.append(
                        {
                            "site": site_name,
                            "title": normalize_whitespace(c.get("title", "")),
                            "org": normalize_whitespace(c.get("org", "")),
                            "location": normalize_whitespace(c.get("location", "")),
                            "url": c.get("url", ""),
                            "posted_at": normalize_whitespace(c.get("posted_at", "")),
                            "deadline": normalize_whitespace(c.get("deadline", "")),
                        }
                    )

            for c in candidates:
                if max_candidates > 0 and processed_total >= max_candidates:
                    site_stat["truncated_by_max_candidates"] = True
                    break

                site_stat["processed"] += 1
                title = normalize_whitespace(c.get("title", ""))
                job_url = c.get("url", "")
                org = normalize_whitespace(c.get("org", ""))
                location = normalize_whitespace(c.get("location", ""))
                deadline = normalize_whitespace(c.get("deadline", ""))
                posted_at = normalize_whitespace(c.get("posted_at", ""))
                department = normalize_whitespace(c.get("department", ""))
                research_field = normalize_whitespace(c.get("research_field", ""))
                researcher_profile = normalize_whitespace(c.get("researcher_profile", ""))
                positions = normalize_whitespace(c.get("positions", ""))
                education_level = normalize_whitespace(c.get("education_level", ""))
                contract_type = normalize_whitespace(c.get("contract_type", ""))
                job_status = normalize_whitespace(c.get("job_status", ""))
                funding_programme = normalize_whitespace(c.get("funding_programme", ""))
                salary = normalize_whitespace(c.get("salary", ""))
                reference_number = normalize_whitespace(c.get("reference_number", ""))
                languages = normalize_whitespace(c.get("languages", ""))

                if title and len(site_stat["sample_titles"]) < 3:
                    site_stat["sample_titles"].append(title)

                semantic_key = ""
                if dedup_method == "semantic_key":
                    title_key = _normalize_dedup_title(title, degree_stopwords)
                    org_key = _normalize_dedup_org(org)
                    loc_key = _normalize_dedup_location(location)
                    semantic_key = _build_semantic_key(
                        {
                            "title": title,
                            "org": org,
                            "location": location,
                            "deadline": deadline,
                            "posted_at": posted_at,
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
                        },
                        title_key,
                        org_key,
                        loc_key,
                    )
                job_id = make_job_id(job_url, semantic_key=semantic_key)

                if job_id in seen_job_ids_run:
                    continue
                seen_job_ids_run.add(job_id)

                seen_before = db.is_seen(job_id)
                if seen_before:
                    if resurface_mode == "never":
                        site_stat["seen_skipped"] += 1
                        seen_skipped_total += 1
                        continue
                    if (
                        resurface_mode == "unexpired"
                        and deadline
                        and not _deadline_is_active(
                            deadline,
                            allow_without_deadline=allow_without_deadline,
                        )
                    ):
                        site_stat["seen_skipped"] += 1
                        seen_skipped_total += 1
                        continue

                # Fetch detail page to extract description
                description = normalize_whitespace(c.get("description", ""))
                extract_method = ""
                extract_warnings: List[str] = []
                try:
                    if is_eth_gethired and eth_ctx and c.get("eth_id"):
                        detail = _eth_gethired_detail(
                            http,
                            eth_ctx["root"],
                            token=eth_ctx["token"],
                            job_id=str(c.get("eth_id") or ""),
                        )
                        body_html = detail.get("body") or ""
                        if body_html:
                            description = normalize_whitespace(
                                BeautifulSoup(body_html, "html.parser").get_text(" ", strip=True)
                            )
                        extract_method = "eth_api"
                        if not title and detail.get("name"):
                            title = normalize_whitespace(detail.get("name") or "")
                        if not org and detail.get("customerDisplayname"):
                            org = normalize_whitespace(detail.get("customerDisplayname") or "")
                        if not posted_at and detail.get("dateStart"):
                            posted_at = normalize_whitespace(detail.get("dateStart") or "")
                        if not deadline and detail.get("dateEnd"):
                            deadline = normalize_whitespace(detail.get("dateEnd") or "")
                        if not location:
                            addr = detail.get("addressLocation") or {}
                            if isinstance(addr, dict):
                                addr_full = addr.get("addressFull") or ""
                                if not addr_full and addr.get("addressFullHtml"):
                                    addr_full = BeautifulSoup(
                                        addr.get("addressFullHtml"), "html.parser"
                                    ).get_text(" ", strip=True)
                                if not addr_full:
                                    parts = [
                                        addr.get("city") or "",
                                        (
                                            (addr.get("country") or {}).get("name")
                                            if isinstance(addr.get("country"), dict)
                                            else ""
                                        ),
                                        (
                                            (addr.get("region") or {}).get("name")
                                            if isinstance(addr.get("region"), dict)
                                            else ""
                                        ),
                                    ]
                                    addr_full = ", ".join([p for p in parts if p])
                                location = normalize_whitespace(addr_full)
                    elif is_academictransfer and c.get("at_payload"):
                        detail = c.get("at_payload") or {}
                        description = _academictransfer_payload_text(detail)
                        extract_method = "academictransfer_api"
                        if not title and detail.get("title"):
                            title = normalize_whitespace(detail.get("title") or "")
                        if not org and detail.get("organisation_name"):
                            org = normalize_whitespace(detail.get("organisation_name") or "")
                        if not posted_at and detail.get("created_datetime"):
                            posted_at = normalize_whitespace(detail.get("created_datetime") or "")
                        if not deadline and detail.get("end_date"):
                            deadline = normalize_whitespace(detail.get("end_date") or "")
                        if not location:
                            city = normalize_whitespace(detail.get("city") or "")
                            country = normalize_whitespace(detail.get("country_code") or "")
                            location = ", ".join([p for p in [city, country] if p])
                        if not department and detail.get("department_name"):
                            department = normalize_whitespace(detail.get("department_name") or "")
                        if not research_field and detail.get("research_fields"):
                            research_field = ", ".join(
                                [str(x) for x in detail.get("research_fields") or []]
                            )
                        if not education_level and detail.get("education_level") is not None:
                            education_level = str(detail.get("education_level") or "")
                        if not contract_type and detail.get("contract_type") is not None:
                            contract_type = str(detail.get("contract_type") or "")
                        if not positions and detail.get("available_positions") is not None:
                            positions = str(detail.get("available_positions") or "")
                        if not salary:
                            min_salary = detail.get("min_salary")
                            max_salary = detail.get("max_salary")
                            if min_salary or max_salary:
                                salary = (
                                    f"{min_salary}-{max_salary}"
                                    if min_salary and max_salary
                                    else f"{min_salary or max_salary}"
                                )
                    else:
                        detail_retries = int(site.get("detail_retries", 3) or 3)
                        detail_backoff = float(site.get("detail_backoff_sec", 1.5) or 1.5)
                        detail_html = _fetch_detail_html(
                            http,
                            job_url,
                            max_attempts=detail_retries,
                            backoff_sec=detail_backoff,
                        )
                        enriched = adapter.enrich_detail(
                            detail_html,
                            selectors=selectors,
                            min_len=min_len,
                            enable_fallback=enable_fallback,
                        )
                        description = enriched.get("description", "")
                        extract_method = enriched.get("extract_method", "") or ""
                        extract_warnings = list(enriched.get("extract_warnings") or [])
                        if not title and enriched.get("title"):
                            title = normalize_whitespace(enriched.get("title", ""))
                        if not org and enriched.get("org"):
                            org = normalize_whitespace(enriched.get("org", ""))
                        if not location and enriched.get("location"):
                            location = normalize_whitespace(enriched.get("location", ""))
                        if not deadline and enriched.get("deadline"):
                            deadline = normalize_whitespace(enriched.get("deadline", ""))
                        if not posted_at and enriched.get("posted_at"):
                            posted_at = normalize_whitespace(enriched.get("posted_at", ""))
                        if not department and enriched.get("department"):
                            department = normalize_whitespace(enriched.get("department", ""))
                        if not research_field and enriched.get("research_field"):
                            research_field = normalize_whitespace(
                                enriched.get("research_field", "")
                            )
                        if not researcher_profile and enriched.get("researcher_profile"):
                            researcher_profile = normalize_whitespace(
                                enriched.get("researcher_profile", "")
                            )
                        if not positions and enriched.get("positions"):
                            positions = normalize_whitespace(enriched.get("positions", ""))
                        if not education_level and enriched.get("education_level"):
                            education_level = normalize_whitespace(
                                enriched.get("education_level", "")
                            )
                        if not contract_type and enriched.get("contract_type"):
                            contract_type = normalize_whitespace(enriched.get("contract_type", ""))
                        if not job_status and enriched.get("job_status"):
                            job_status = normalize_whitespace(enriched.get("job_status", ""))
                        if not funding_programme and enriched.get("funding_programme"):
                            funding_programme = normalize_whitespace(
                                enriched.get("funding_programme", "")
                            )
                        if not salary and enriched.get("salary"):
                            salary = normalize_whitespace(enriched.get("salary", ""))
                        if not reference_number and enriched.get("reference_number"):
                            reference_number = normalize_whitespace(
                                enriched.get("reference_number", "")
                            )
                        if not languages and enriched.get("languages"):
                            languages = normalize_whitespace(enriched.get("languages", ""))
                except Exception as ex:
                    extract_warnings.append("detail_fetch_failed")
                    log_error(
                        "job_enrich_failed",
                        run_id=run_id,
                        source=site_name,
                        url=job_url,
                        error=repr(ex),
                    )

                if is_academictransfer:
                    education_level, contract_type = _normalize_academictransfer_meta(
                        education_level,
                        contract_type,
                        title=title,
                        description=description,
                    )

                if dedup_method == "semantic_key":
                    title_key_after = _normalize_dedup_title(title, degree_stopwords)
                    org_key_after = _normalize_dedup_org(org)
                    loc_key_after = _normalize_dedup_location(location)
                    semantic_key_after = _build_semantic_key(
                        {
                            "title": title,
                            "org": org,
                            "location": location,
                            "deadline": deadline,
                            "posted_at": posted_at,
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
                        },
                        title_key_after,
                        org_key_after,
                        loc_key_after,
                    )
                    if semantic_key_after and semantic_key_after != semantic_key:
                        job_id_after = make_job_id(job_url, semantic_key=semantic_key_after)
                        if job_id_after in seen_job_ids_run:
                            continue
                        seen_job_ids_run.add(job_id_after)
                        job_id = job_id_after
                        seen_before = seen_before or db.is_seen(job_id)

                if (
                    seen_before
                    and resurface_mode == "unexpired"
                    and not _deadline_is_active(
                        deadline,
                        allow_without_deadline=allow_without_deadline,
                    )
                ):
                    site_stat["seen_skipped"] += 1
                    seen_skipped_total += 1
                    continue

                hay_title = title
                hay_all = f"{title}\n{org}\n{location}\n{description}"

                # Hard rules: exclude only on title; degree on full text by default
                if _match_any(hay_title, exclude_kw):
                    reasons = (
                        ["Excluded by negative keywords (rule-based skip)."]
                        if output_language == "en"
                        else ["命中排除关键词（规则跳过）"]
                    )
                    final_items.append(
                        {
                            "order": processed_total,
                            "id": job_id,
                            "title": title,
                            "org": org,
                            "location": location,
                            "url": job_url,
                            "posted_at": posted_at,
                            "deadline": deadline,
                            "site": site_name,
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
                            "description": description,
                            "extract_method": extract_method,
                            "extract_warnings": extract_warnings,
                            "summary_bullets": [],
                            "relevance_score": 0,
                            "recommendation": "skip",
                            "reasons": reasons,
                            "rule_skip": True,
                            "rule_skip_reason": "exclude",
                            "prefilter_score": 0,
                            "prefilter_title_hits": [],
                            "prefilter_desc_hits": [],
                            "prefilter_skip_reason": "rule_exclude",
                            "llm_status": "skipped",
                            "llm_input_chars": 0,
                            "llm_input_tokens": 0,
                            "llm_output_tokens": 0,
                            "llm_total_tokens": 0,
                            "llm_provider": "",
                            "llm_model": "",
                            "budget_hit": 0,
                        }
                    )
                    site_stat["rule_skipped_total"] += 1
                    site_stat["rule_skipped"]["exclude"] += 1
                    rule_skipped_total += 1
                    rule_skipped_exclude += 1
                else:
                    degree_required = True
                    if is_academictransfer and (search_cfg.get("structured_filters") or {}).get(
                        "education_level"
                    ):
                        degree_required = False
                    if is_eth_gethired and (
                        search_cfg.get("jobtypessub") or search_cfg.get("jobtypesub")
                    ):
                        degree_required = False
                    if adapter_name == "euraxess" and (education_level or positions):
                        degree_required = False
                    if degree_required and degree_kw and not _match_required(hay_all, degree_kw):
                        reasons = (
                            [
                                "Missing degree keywords (rule-based skip; relax in config if needed)."
                            ]
                            if output_language == "en"
                            else ["未命中学位关键词（规则跳过，可在 config 放宽）"]
                        )
                        final_items.append(
                            {
                                "order": processed_total,
                                "id": job_id,
                                "title": title,
                                "org": org,
                                "location": location,
                                "url": job_url,
                                "posted_at": posted_at,
                                "deadline": deadline,
                                "site": site_name,
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
                                "description": description,
                                "extract_method": extract_method,
                                "extract_warnings": extract_warnings,
                                "summary_bullets": [],
                                "relevance_score": 0,
                                "recommendation": "skip",
                                "reasons": reasons,
                                "rule_skip": True,
                                "rule_skip_reason": "degree",
                                "prefilter_score": 0,
                                "prefilter_title_hits": [],
                                "prefilter_desc_hits": [],
                                "prefilter_skip_reason": "rule_degree",
                                "llm_status": "skipped",
                                "llm_input_chars": 0,
                                "llm_input_tokens": 0,
                                "llm_output_tokens": 0,
                                "llm_total_tokens": 0,
                                "llm_provider": "",
                                "llm_model": "",
                                "budget_hit": 0,
                            }
                        )
                        site_stat["rule_skipped_total"] += 1
                        site_stat["rule_skipped"]["degree"] += 1
                        rule_skipped_total += 1
                        rule_skipped_degree += 1
                    else:
                        pre_score, pre_title_hits, pre_desc_hits = _prefilter_score(
                            title,
                            description,
                            pre_keywords,
                            title_weight=pre_title_w,
                            desc_weight=pre_desc_w,
                            allow_if_no_keywords=pre_allow_no_kw,
                        )
                        prefilter_candidates.append(
                            {
                                "order": processed_total,
                                "site": site_name,
                                "id": job_id,
                                "title": title,
                                "org": org,
                                "location": location,
                                "url": job_url,
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
                                "description": description,
                                "extract_method": extract_method,
                                "extract_warnings": extract_warnings,
                                "prefilter_score": pre_score,
                                "prefilter_title_hits": pre_title_hits,
                                "prefilter_desc_hits": pre_desc_hits,
                            }
                        )
                    site_stat["prefilter_candidates"] += 1
                    prefilter_candidates_total += 1

                processed_total += 1

            site_stats[site_name] = site_stat
        except Exception as ex:
            tb = traceback.format_exc()
            db.record_error(
                run_id,
                module="job",
                source=site_name,
                error_type=type(ex).__name__,
                message=tb,
            )
            log_error(
                "job_agent_source_failed",
                run_id=run_id,
                source=site_name,
                error=repr(ex),
            )
            site_stat["error"] = repr(ex)
            site_stats[site_name] = site_stat

    # Prefilter scoring + Top-N LLM
    prefilter_candidates.sort(key=lambda x: x.get("prefilter_score", 0), reverse=True)
    llm_selected: List[Dict[str, Any]] = []
    prefilter_skipped: List[Dict[str, Any]] = []
    llm_used = 0
    for c in prefilter_candidates:
        pre_score = int(c.get("prefilter_score", 0))
        eligible_for_llm = (pre_score >= pre_min_score) or (not pre_keywords and pre_allow_no_kw)
        if eligible_for_llm:
            prefilter_eligible_total += 1
        if eligible_for_llm and llm_used < max_llm:
            llm_selected.append(c)
            llm_used += 1
            if c["site"] in site_stats:
                site_stats[c["site"]]["prefilter_selected"] += 1
                site_stats[c["site"]]["llm_used"] += 1
            llm_used_total += 1
        else:
            c["prefilter_skip_reason"] = (
                "score_below_threshold" if not eligible_for_llm else "llm_quota_full"
            )
            prefilter_skipped.append(c)
            prefilter_skipped_total += 1
            if c["site"] in site_stats:
                site_stats[c["site"]]["prefilter_skipped"] += 1

    # LLM analyze selected candidates
    for c in llm_selected:
        llm_out = llm.analyze_job(
            profile=cfg["profile"],
            thresholds={
                "strong_threshold": int(cfg["papers"].get("strong_threshold", 8)),  # reuse
                "normal_threshold": int(cfg["papers"].get("normal_threshold", 5)),
            },
            title=c["title"],
            org=c["org"],
            location=c["location"],
            deadline=c["deadline"],
            description=c["description"],
            url=c["url"],
        )
        rec = llm_out.recommendation
        score = llm_out.relevance_score
        summary_bullets = llm_out.summary_bullets
        reasons = llm_out.reasons
        llm_status = llm_out.llm_status
        llm_input_chars = llm_out.llm_input_chars
        llm_input_tokens = llm_out.llm_input_tokens
        llm_output_tokens = llm_out.llm_output_tokens
        llm_total_tokens = llm_out.llm_total_tokens
        llm_provider = getattr(llm_out, "llm_provider", "") or ""
        llm_model = getattr(llm_out, "llm_model", "") or ""
        budget_hit = llm_out.budget_hit
        if budget_hit:
            budget_hit_count += 1
        llm_input_tokens_sum += llm_input_tokens
        llm_output_tokens_sum += llm_output_tokens
        llm_total_tokens_sum += llm_total_tokens
        _bump_usage(
            llm_provider,
            llm_model,
            llm_input_tokens,
            llm_output_tokens,
            llm_total_tokens,
        )

        final_items.append(
            {
                "order": c["order"],
                "id": c["id"],
                "title": c["title"],
                "org": c["org"],
                "location": c["location"],
                "url": c["url"],
                "posted_at": c["posted_at"],
                "deadline": c["deadline"],
                "site": c["site"],
                "department": c.get("department", ""),
                "research_field": c.get("research_field", ""),
                "researcher_profile": c.get("researcher_profile", ""),
                "positions": c.get("positions", ""),
                "education_level": c.get("education_level", ""),
                "contract_type": c.get("contract_type", ""),
                "job_status": c.get("job_status", ""),
                "funding_programme": c.get("funding_programme", ""),
                "salary": c.get("salary", ""),
                "reference_number": c.get("reference_number", ""),
                "languages": c.get("languages", ""),
                "description": c["description"],
                "extract_method": c["extract_method"],
                "extract_warnings": c["extract_warnings"],
                "summary_bullets": summary_bullets,
                "relevance_score": score,
                "recommendation": rec,
                "reasons": reasons,
                "rule_skip": False,
                "rule_skip_reason": "",
                "prefilter_score": int(c.get("prefilter_score", 0)),
                "prefilter_title_hits": c.get("prefilter_title_hits") or [],
                "prefilter_desc_hits": c.get("prefilter_desc_hits") or [],
                "prefilter_skip_reason": "",
                "llm_status": llm_status,
                "llm_input_chars": llm_input_chars,
                "llm_input_tokens": llm_input_tokens,
                "llm_output_tokens": llm_output_tokens,
                "llm_total_tokens": llm_total_tokens,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "budget_hit": budget_hit,
            }
        )

    if export_candidates:
        Path("logs").mkdir(parents=True, exist_ok=True)
        out_path = Path("logs") / f"job_candidates_{run_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in candidates_preview:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        log_event(
            "job_candidates_exported",
            run_id=run_id,
            path=str(out_path),
            count=len(candidates_preview),
        )

    # Prefilter skipped candidates (no LLM call)
    for c in prefilter_skipped:
        pre_score = int(c.get("prefilter_score", 0))
        if c.get("prefilter_skip_reason") == "llm_quota_full":
            reason = (
                [f"Prefilter rank not in top {max_llm} (LLM slots used up)."]
                if output_language == "en"
                else [f"粗筛排序未入选（LLM 名额 {max_llm} 已用完）"]
            )
        else:
            if not pre_keywords and not pre_allow_no_kw:
                reason = (
                    ["Prefilter disabled: no keywords configured and allow_if_no_keywords=false."]
                    if output_language == "en"
                    else ["粗筛未启用：未配置关键词且 allow_if_no_keywords=false"]
                )
            else:
                reason = (
                    [f"Prefilter score {pre_score} < threshold {pre_min_score}."]
                    if output_language == "en"
                    else [f"粗筛得分 {pre_score} < 阈值 {pre_min_score}"]
                )
        title_hits = c.get("prefilter_title_hits") or []
        desc_hits = c.get("prefilter_desc_hits") or []
        if title_hits or desc_hits:
            reason.append(
                f"Hits: title {title_hits[:3]} / desc {desc_hits[:3]}"
                if output_language == "en"
                else f"命中：title {title_hits[:3]} / desc {desc_hits[:3]}"
            )

        final_items.append(
            {
                "order": c["order"],
                "id": c["id"],
                "title": c["title"],
                "org": c["org"],
                "location": c["location"],
                "url": c["url"],
                "posted_at": c["posted_at"],
                "deadline": c["deadline"],
                "site": c["site"],
                "department": c.get("department", ""),
                "research_field": c.get("research_field", ""),
                "researcher_profile": c.get("researcher_profile", ""),
                "positions": c.get("positions", ""),
                "education_level": c.get("education_level", ""),
                "contract_type": c.get("contract_type", ""),
                "job_status": c.get("job_status", ""),
                "funding_programme": c.get("funding_programme", ""),
                "salary": c.get("salary", ""),
                "reference_number": c.get("reference_number", ""),
                "languages": c.get("languages", ""),
                "description": c["description"],
                "extract_method": c["extract_method"],
                "extract_warnings": c["extract_warnings"],
                "summary_bullets": [],
                "relevance_score": 0,
                "recommendation": "skip",
                "reasons": reason,
                "rule_skip": False,
                "rule_skip_reason": "",
                "prefilter_score": pre_score,
                "prefilter_title_hits": title_hits,
                "prefilter_desc_hits": desc_hits,
                "prefilter_skip_reason": c.get("prefilter_skip_reason", ""),
                "llm_status": "prefilter_skip",
                "llm_input_chars": 0,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "llm_total_tokens": 0,
                "llm_provider": "",
                "llm_model": "",
                "budget_hit": 0,
            }
        )

    # Sort by score (desc), keep stable order for ties
    final_items.sort(key=lambda x: (-int(x.get("relevance_score", 0)), int(x.get("order", 0))))

    # Persist + prepare email
    for item in final_items:
        seen_before = db.is_seen(item["id"])
        db.mark_seen(item["id"], item_type="job", source=item["site"])
        db.upsert_job(
            JobRow(
                id=item["id"],
                title=item["title"],
                org=item["org"],
                location=item["location"],
                url=item["url"],
                posted_at=item["posted_at"],
                deadline=item["deadline"],
                site=item["site"],
                department=item.get("department", ""),
                research_field=item.get("research_field", ""),
                researcher_profile=item.get("researcher_profile", ""),
                positions=item.get("positions", ""),
                education_level=item.get("education_level", ""),
                contract_type=item.get("contract_type", ""),
                job_status=item.get("job_status", ""),
                funding_programme=item.get("funding_programme", ""),
                salary=item.get("salary", ""),
                reference_number=item.get("reference_number", ""),
                languages=item.get("languages", ""),
                description=item["description"],
                extract_method=item["extract_method"],
                extract_warnings=item["extract_warnings"],
                summary_bullets=item["summary_bullets"],
                relevance_score=item["relevance_score"],
                recommendation=item["recommendation"],
                reasons=item["reasons"],
                llm_status=item["llm_status"],
                llm_input_chars=item["llm_input_chars"],
                llm_input_tokens=item["llm_input_tokens"],
                llm_output_tokens=item["llm_output_tokens"],
                llm_total_tokens=item["llm_total_tokens"],
                budget_hit=item["budget_hit"],
            )
        )
        if not seen_before:
            new_count += 1
        if item["site"] in site_stats:
            if not seen_before:
                site_stats[item["site"]]["new"] += 1
        item_email = {
            "id": item["id"],
            "title": item["title"],
            "org": item["org"],
            "location": item["location"],
            "url": item["url"],
            "posted_at": item["posted_at"],
            "deadline": item["deadline"],
            "site": item["site"],
            "department": item.get("department", ""),
            "research_field": item.get("research_field", ""),
            "researcher_profile": item.get("researcher_profile", ""),
            "positions": item.get("positions", ""),
            "education_level": item.get("education_level", ""),
            "contract_type": item.get("contract_type", ""),
            "job_status": item.get("job_status", ""),
            "funding_programme": item.get("funding_programme", ""),
            "salary": item.get("salary", ""),
            "reference_number": item.get("reference_number", ""),
            "languages": item.get("languages", ""),
            "extract_method": item["extract_method"],
            "extract_warnings": item["extract_warnings"],
            "summary_bullets": item["summary_bullets"],
            "relevance_score": item["relevance_score"],
            "recommendation": item["recommendation"],
            "reasons": item["reasons"],
            "rule_skip": item.get("rule_skip", False),
            "rule_skip_reason": item.get("rule_skip_reason", ""),
            "prefilter_score": item.get("prefilter_score", 0),
            "prefilter_title_hits": item.get("prefilter_title_hits", []),
            "prefilter_desc_hits": item.get("prefilter_desc_hits", []),
            "prefilter_skip_reason": item.get("prefilter_skip_reason", ""),
            "llm_status": item["llm_status"],
            "llm_input_chars": item["llm_input_chars"],
            "llm_input_tokens": item["llm_input_tokens"],
            "llm_output_tokens": item["llm_output_tokens"],
            "llm_total_tokens": item["llm_total_tokens"],
            "llm_provider": item.get("llm_provider", ""),
            "llm_model": item.get("llm_model", ""),
            "budget_hit": item["budget_hit"],
        }
        item_email.update(normalize_job_fields(item_email))
        new_items_for_email.append(item_email)

    if export_audit:
        Path("logs").mkdir(parents=True, exist_ok=True)
        out_path = Path("logs") / f"job_audit_{run_id}.jsonl"
        count = 0
        with out_path.open("w", encoding="utf-8") as f:
            for item in new_items_for_email:
                if audit_limit > 0 and count >= audit_limit:
                    break
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
        log_event(
            "job_audit_exported",
            run_id=run_id,
            path=str(out_path),
            count=count,
        )

    log_event(
        "job_agent_done",
        run_id=run_id,
        jobs_found=found,
        jobs_new=new_count,
        llm_input_tokens=llm_input_tokens_sum,
        llm_output_tokens=llm_output_tokens_sum,
        llm_total_tokens=llm_total_tokens_sum,
        llm_usage_by_provider=llm_usage_by_provider,
    )
    log_event("job_site_capture_stats", run_id=run_id, stats=site_stats)
    log_event(
        "job_rule_skip_stats",
        run_id=run_id,
        seen_skipped=seen_skipped_total,
        rule_skipped=rule_skipped_total,
        rule_skipped_exclude=rule_skipped_exclude,
        rule_skipped_degree=rule_skipped_degree,
        prefilter_candidates=prefilter_candidates_total,
        prefilter_eligible=prefilter_eligible_total,
        prefilter_skipped=prefilter_skipped_total,
        prefilter_min_score=pre_min_score,
        prefilter_keywords_count=len(pre_keywords),
        llm_used=llm_used_total,
    )
    return {
        "jobs_found": found,
        "jobs_new": new_count,
        "budget_hit": budget_hit_count,
        "llm_input_tokens": llm_input_tokens_sum,
        "llm_output_tokens": llm_output_tokens_sum,
        "llm_total_tokens": llm_total_tokens_sum,
        "llm_usage_by_provider": llm_usage_by_provider,
        "jobs_new_items": new_items_for_email,
    }
