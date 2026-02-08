#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure imports work when running from repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from foryourseek.core.budget import Budget
from foryourseek.core.config import load_config
from foryourseek.core.extract import fetch_content_smart
from foryourseek.core.http import HttpClient
from foryourseek.core.llm import LLMClient
from foryourseek.core.text_cleaner import clean_text_for_llm, truncate_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", choices=["paper", "job"], required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--selector", default="", help="Optional CSS selector for precise extraction")
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--print-clean-text", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / "config.yaml")
    http = HttpClient(
        user_agent=cfg.get("runtime", {}).get("user_agent", "foryourseek/debug"),
        timeout_sec=20,
        retries=2,
    )

    html = http.get_text(args.url)
    r = fetch_content_smart(
        html,
        specific_selector=args.selector or None,
        min_len=120,
        kind=args.type,
        enable_fallback=True,
    )

    cleaned = clean_text_for_llm(r.text)
    cleaned = truncate_text(cleaned, args.max_chars)

    if args.print_clean_text:
        print("\n--- CLEANED TEXT (truncated) ---\n")
        print(cleaned)
        print("\n--- END ---\n")

    budget = Budget(max_calls=5)
    llm = LLMClient(cfg.get("llm", {}), budget=budget)

    if args.type == "paper":
        out = llm.analyze_paper(
            profile=cfg["profile"],
            thresholds={
                "strong_threshold": cfg["papers"].get("strong_threshold", 8),
                "normal_threshold": cfg["papers"].get("normal_threshold", 5),
            },
            title="(debug) title",
            abstract=cleaned,
            journal="(debug)",
            published_at="",
            url=args.url,
        )
    else:
        out = llm.analyze_job(
            profile=cfg["profile"],
            thresholds={
                "strong_threshold": cfg["papers"].get("strong_threshold", 8),
                "normal_threshold": cfg["papers"].get("normal_threshold", 5),
            },
            title="(debug) job title",
            org="(debug)",
            location="",
            deadline="",
            description=cleaned,
            url=args.url,
        )

    print(
        json.dumps(
            {
                "extract_method": r.method,
                "extract_warnings": r.warnings,
                "llm_status": out.llm_status,
                "llm_input_chars": out.llm_input_chars,
                "llm_input_tokens": out.llm_input_tokens,
                "llm_output_tokens": out.llm_output_tokens,
                "llm_total_tokens": out.llm_total_tokens,
                "budget_hit": out.budget_hit,
                "relevance_score": out.relevance_score,
                "recommendation": out.recommendation,
                "summary_bullets": out.summary_bullets,
                "reasons": out.reasons,
                "confidence": out.confidence,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
