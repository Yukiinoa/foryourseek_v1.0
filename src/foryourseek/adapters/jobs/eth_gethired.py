from __future__ import annotations

from typing import Any, Dict, List

from .generic_static import enrich_detail as _enrich_detail


def fetch_list(
    url: str,
    html: str,
    *,
    selectors: Dict[str, str] | None = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    # ETH GetHired listings are loaded via API; list scraping is handled in job_agent.
    return []


def enrich_detail(
    detail_html: str,
    *,
    selectors: Dict[str, str] | None = None,
    min_len: int = 120,
    enable_fallback: bool = True,
) -> Dict[str, Any]:
    # Fallback only (in case API detail fails).
    return _enrich_detail(
        detail_html,
        selectors=selectors,
        min_len=min_len,
        enable_fallback=enable_fallback,
    )
