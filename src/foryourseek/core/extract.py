from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from bs4 import BeautifulSoup

try:
    import trafilatura
except Exception:  # allow running without trafilatura
    trafilatura = None


@dataclass
class ExtractResult:
    text: str
    method: str  # selector | trafilatura | bs4_text | empty
    warnings: list[str]


def extract_by_selector(html: str, selector: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one(selector)
    if not node:
        return ""
    return node.get_text(" ", strip=True)


def extract_all_text_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(" ", strip=True)


def fetch_content_smart(
    html: str,
    specific_selector: Optional[str] = None,
    min_len: int = 120,
    kind: str = "generic",  # paper | job | generic
    enable_fallback: bool = True,
) -> ExtractResult:
    warnings: list[str] = []

    # 1) precise selector
    if specific_selector:
        text = extract_by_selector(html, specific_selector)
        if len(text) >= min_len:
            return ExtractResult(text=text, method="selector", warnings=warnings)
        warnings.append("selector_too_short_or_failed")

    # 2) trafilatura fallback
    if enable_fallback and trafilatura is not None:
        include_tables = False if kind == "paper" else True
        extracted = trafilatura.extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=include_tables,
        )
        if extracted and len(extracted.strip()) >= min_len:
            return ExtractResult(
                text=extracted.strip(),
                method="trafilatura",
                warnings=warnings + ["fallback_used"],
            )
        warnings.append("trafilatura_failed_or_too_short")

    # 3) noisy fallback
    text = extract_all_text_bs4(html)
    if text:
        return ExtractResult(text=text, method="bs4_text", warnings=warnings + ["noisy_fallback"])

    return ExtractResult(text="", method="empty", warnings=warnings + ["empty"])
