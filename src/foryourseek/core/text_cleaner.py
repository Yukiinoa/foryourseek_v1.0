from __future__ import annotations

import html as _html
import re

JUNK_HINTS = [
    "cookie",
    "privacy policy",
    "terms of use",
    "subscribe",
    "sign up",
    "all rights reserved",
    "javascript",
    "accept cookies",
]

RE_MULTI_SPACE = re.compile(r"\s+")
RE_REPEAT_PUNCT = re.compile(r"([。.!?])\1+")

_MOJIBAKE_MARKERS = [
    "â€",
    "â€™",
    "â€œ",
    "â€",
    "â€“",
    "â€”",
    "â€¦",
    "â",
    "Ã",
    "Â",
    "�",
]


def _mojibake_score(text: str) -> int:
    score = 0
    for m in _MOJIBAKE_MARKERS:
        score += text.count(m)
    return score


def fix_text_encoding(text: str) -> str:
    """Best-effort fix for common mojibake (UTF-8 decoded as latin-1) and HTML entities."""
    if not text:
        return ""
    t = _html.unescape(text)
    score = _mojibake_score(t)
    if score == 0:
        return t
    try:
        cand = t.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return t
    return cand if _mojibake_score(cand) < score else t


def clean_text_for_llm(text: str) -> str:
    if not text:
        return ""

    text = fix_text_encoding(text)

    # 1) normalize whitespace
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    # 2) collapse repeated punctuation
    text = RE_REPEAT_PUNCT.sub(r"\1", text)

    # 3) light de-noise: remove very short junk fragments
    parts = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    cleaned_parts = []
    for p in parts:
        low = p.lower()
        if any(h in low for h in JUNK_HINTS) and len(p) < 120:
            continue
        cleaned_parts.append(p)

    return " ".join(cleaned_parts).strip()


def truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    # try to cut at a natural boundary
    for sep in ["。", ". ", "\n", "; ", ", "]:
        idx = cut.rfind(sep)
        if idx > max_chars * 0.7:
            return cut[: idx + len(sep)].strip()
    return cut.strip()


def head_sentences(text: str, max_chars: int = 600) -> str:
    """Simple rule-based fallback summary."""
    t = clean_text_for_llm(text)
    return truncate_text(t, max_chars)
