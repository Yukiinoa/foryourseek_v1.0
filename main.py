#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "src"))

from zoneinfo import ZoneInfo

from dateutil import parser as date_parser

from foryourseek.agents.job_agent import run_job_agent
from foryourseek.agents.paper_agent import run_paper_agent
from foryourseek.core.budget import Budget
from foryourseek.core.config import load_config
from foryourseek.core.db import Database
from foryourseek.core.db_maintenance import maintenance_db
from foryourseek.core.emailer import render_daily_brief, send_alert_email, send_email
from foryourseek.core.http import HttpClient
from foryourseek.core.llm import LLMClient
from foryourseek.core.logging import log_error, log_event, setup_logging


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Do not send email; render HTML locally")
    ap.add_argument("--only", choices=["papers", "jobs"], default="", help="Run only one agent")
    ap.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help="Override papers.max_per_day for this run (0=use config)",
    )
    ap.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Override jobs.max_per_day for this run (0=use config)",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(ROOT / "config.yaml")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 2

    # Apply CLI overrides
    if args.max_papers > 0:
        cfg["papers"]["max_per_day"] = args.max_papers
        cfg["papers"]["max_candidates_per_run"] = args.max_papers
        cfg["papers"]["max_llm_per_run"] = args.max_papers
    if args.max_jobs > 0:
        cfg["jobs"]["max_per_day"] = args.max_jobs
        cfg["jobs"]["max_candidates_per_run"] = args.max_jobs
        cfg["jobs"]["max_llm_per_run"] = args.max_jobs

    runtime = cfg.get("runtime", {})
    env = (runtime.get("env") or "dev").lower()
    tz = ZoneInfo(runtime.get("timezone", "Asia/Tokyo"))
    start_time = time.perf_counter()

    def _make_run_id() -> str:
        base = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        candidate = base
        idx = 1
        while (log_dir / f"run_{candidate}.jsonl").exists():
            candidate = f"{base}_{idx:02d}"
            idx += 1
        return candidate

    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m{secs:02d}s"
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h{mins:02d}m{secs:02d}s"

    def _estimate_llm_cost_from_pricing(
        pricing: Dict[str, Any],
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        if not pricing:
            return 0.0
        in_rate_1m = pricing.get("input_per_1m")
        out_rate_1m = pricing.get("output_per_1m")
        if in_rate_1m is not None or out_rate_1m is not None:
            in_rate = float(in_rate_1m or 0)
            out_rate = float(out_rate_1m or 0)
            return (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate
        in_rate = float(pricing.get("input_per_1k", 0) or 0)
        out_rate = float(pricing.get("output_per_1k", 0) or 0)
        return (input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate

    def _estimate_llm_cost_by_provider(
        llm_cfg: Dict[str, Any],
        usage_by_provider: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, float], float]:
        pricing = llm_cfg.get("pricing") or {}
        providers_pricing = pricing.get("providers") or {}
        default_pricing = pricing.get("default")
        if default_pricing is None:
            default_pricing = {
                k: v for k, v in pricing.items() if k not in ("providers", "default")
            }
        cost_by_provider: Dict[str, float] = {}
        total_cost = 0.0
        for provider, usage in (usage_by_provider or {}).items():
            provider_pricing = providers_pricing.get(provider) or default_pricing or {}
            in_tokens = int(usage.get("input_tokens", 0) or 0)
            out_tokens = int(usage.get("output_tokens", 0) or 0)
            cost = _estimate_llm_cost_from_pricing(provider_pricing, in_tokens, out_tokens)
            cost_by_provider[provider] = round(float(cost), 6)
            total_cost += float(cost)
        return cost_by_provider, total_cost

    def _merge_usage(
        target: Dict[str, Dict[str, Any]],
        source: Dict[str, Dict[str, Any]],
    ) -> None:
        for provider, usage in (source or {}).items():
            entry = target.setdefault(
                provider,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "models": {},
                },
            )
            entry["calls"] += int(usage.get("calls", 0) or 0)
            entry["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            entry["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            entry["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            models = usage.get("models") or {}
            if isinstance(models, dict):
                dest_models = entry.setdefault("models", {})
                for m, count in models.items():
                    dest_models[m] = int(dest_models.get(m, 0)) + int(count or 0)

    def _llm_display(usage_by_provider: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
        if not usage_by_provider:
            return "n/a", "n/a"
        if len(usage_by_provider) != 1:
            return "mixed", "mixed"
        provider = next(iter(usage_by_provider.keys()))
        models = list((next(iter(usage_by_provider.values())).get("models") or {}).keys())
        if len(models) == 1:
            return provider, models[0]
        if not models:
            return provider, "n/a"
        return provider, "mixed"

    def _parse_deadline(raw: str, *, today: datetime) -> datetime | None:
        if not raw:
            return None
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
            return None
        try:
            dt = date_parser.parse(
                raw,
                fuzzy=True,
                default=datetime(today.year, today.month, today.day),
            )
            tz = today.tzinfo
            if tz is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=tz)
                else:
                    try:
                        dt = dt.astimezone(tz)
                    except Exception:
                        pass
            # drop past dates for "upcoming"
            if dt.date() < today.date():
                return None
            return dt
        except Exception:
            return None

    run_id = _make_run_id()
    log_path = setup_logging(run_id)

    db_path = runtime.get("state_db_path", "state/history.db")
    db = Database(db_path)
    db.create_run(run_id)

    http = HttpClient(
        user_agent=runtime.get("user_agent", "foryourseek/0.1"),
        timeout_sec=int(runtime.get("http_timeout_sec", 20)),
        retries=int(runtime.get("http_retries", 3)),
    )

    budget = Budget(max_calls=int(cfg.get("llm", {}).get("max_calls_per_run", 50)))
    llm = LLMClient(cfg.get("llm", {}), budget=budget, run_id=run_id)

    stats = {
        "papers_found": 0,
        "papers_new": 0,
        "papers_strong": 0,
        "jobs_found": 0,
        "jobs_new": 0,
        "llm_calls": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_total_tokens": 0,
        "budget_hit": 0,
        "error_count": 0,
        "duration_sec": 0.0,
    }

    papers_result = {"papers_new_items": []}
    jobs_result = {"jobs_new_items": []}

    try:
        if args.only != "jobs":
            papers_result = run_paper_agent(run_id=run_id, cfg=cfg, db=db, http=http, llm=llm)
        if args.only != "papers":
            jobs_result = run_job_agent(run_id=run_id, cfg=cfg, db=db, http=http, llm=llm)
    except Exception as ex:
        stats["error_count"] += 1
        tb = traceback.format_exc()
        db.record_error(
            run_id,
            module="main",
            source="pipeline",
            error_type=type(ex).__name__,
            message=tb,
        )
        log_error("pipeline_failed", run_id=run_id, error=repr(ex))

    # Merge stats
    for k in ["papers_found", "papers_new", "papers_strong"]:
        stats[k] = int(papers_result.get(k, 0))
    for k in ["jobs_found", "jobs_new"]:
        stats[k] = int(jobs_result.get(k, 0))
    stats["llm_calls"] = int(budget.calls_used)
    stats["llm_input_tokens"] = int(papers_result.get("llm_input_tokens", 0)) + int(
        jobs_result.get("llm_input_tokens", 0)
    )
    stats["llm_output_tokens"] = int(papers_result.get("llm_output_tokens", 0)) + int(
        jobs_result.get("llm_output_tokens", 0)
    )
    stats["llm_total_tokens"] = int(papers_result.get("llm_total_tokens", 0)) + int(
        jobs_result.get("llm_total_tokens", 0)
    )
    stats["budget_hit"] = int(papers_result.get("budget_hit", 0)) + int(
        jobs_result.get("budget_hit", 0)
    )
    papers_usage = papers_result.get("llm_usage_by_provider") or {}
    jobs_usage = jobs_result.get("llm_usage_by_provider") or {}
    usage_by_provider: Dict[str, Dict[str, Any]] = {}
    _merge_usage(usage_by_provider, papers_usage)
    _merge_usage(usage_by_provider, jobs_usage)
    cost_by_provider, total_cost = _estimate_llm_cost_by_provider(
        cfg.get("llm", {}), usage_by_provider
    )
    cost_by_provider_papers, total_cost_papers = _estimate_llm_cost_by_provider(
        cfg.get("llm", {}), papers_usage
    )
    cost_by_provider_jobs, total_cost_jobs = _estimate_llm_cost_by_provider(
        cfg.get("llm", {}), jobs_usage
    )
    stats["llm_cost_usd"] = total_cost

    # Count errors
    stats["error_count"] = int(stats.get("error_count", 0))

    stats["duration_sec"] = time.perf_counter() - start_time
    duration_str = _format_duration(stats["duration_sec"])

    provider_display, model_display = _llm_display(usage_by_provider)
    log_event(
        "run_summary",
        run_id=run_id,
        llm_calls=stats["llm_calls"],
        llm_provider=provider_display,
        llm_model=model_display,
        llm_input_tokens=stats["llm_input_tokens"],
        llm_output_tokens=stats["llm_output_tokens"],
        llm_total_tokens=stats["llm_total_tokens"],
        llm_cost_usd=round(float(stats["llm_cost_usd"]), 6),
        llm_cost_by_provider=cost_by_provider,
        llm_cost_by_agent={
            "papers": round(total_cost_papers, 6),
            "jobs": round(total_cost_jobs, 6),
        },
        llm_usage_by_provider=usage_by_provider,
        llm_usage_by_agent={"papers": papers_usage, "jobs": jobs_usage},
        duration_sec=round(stats["duration_sec"], 3),
        budget_hit=stats["budget_hit"],
    )

    # Prepare email context
    now_local = datetime.now(tz)
    date_str = now_local.strftime("%Y-%m-%d")

    papers_new_items = papers_result.get("papers_new_items", [])
    jobs_new_items = jobs_result.get("jobs_new_items", [])

    def _llm_ok(item: Dict[str, Any]) -> bool:
        return (item.get("llm_status") or "").lower() == "ok"

    # Only allow LLM-verified items into featured sections. This prevents rule-based fallback
    # from being surfaced as "top picks" when the provider timed out / returned invalid JSON.
    papers_strong = [
        p for p in papers_new_items if _llm_ok(p) and p.get("recommendation") == "strong"
    ]
    papers_normal = [
        p for p in papers_new_items if _llm_ok(p) and p.get("recommendation") == "normal"
    ]

    jobs_all = jobs_result.get("jobs_new_items", [])
    jobs_strong = [j for j in jobs_all if _llm_ok(j) and j.get("recommendation") == "strong"]
    jobs_normal = [j for j in jobs_all if _llm_ok(j) and j.get("recommendation") == "normal"]
    jobs_other = [
        j
        for j in jobs_all
        if _llm_ok(j)
        and j.get("recommendation") not in ("strong", "normal")
        and not j.get("rule_skip")
    ]

    def _deadline_key(item: Dict[str, Any]) -> tuple[int, datetime]:
        dt = _parse_deadline(item.get("deadline", ""), today=now_local)
        if not dt:
            return (1, datetime.max.replace(tzinfo=tz))
        return (0, dt)

    jobs_upcoming = sorted(jobs_other, key=_deadline_key)

    def _job_key(item: Dict[str, Any]) -> str:
        return (
            item.get("url")
            or "|".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("org") or ""),
                    str(item.get("location") or ""),
                ]
            ).strip()
        )

    if jobs_strong:
        jobs_top = jobs_strong
        jobs_list = jobs_normal + jobs_upcoming
    else:
        pool = jobs_normal + jobs_upcoming
        fallback_n = int(cfg.get("jobs", {}).get("top_fallback_count", 5) or 5)

        def _top_sort_key(item: Dict[str, Any]) -> tuple[float, int, datetime]:
            try:
                score = float(item.get("relevance_score") or 0)
            except Exception:
                score = 0.0
            missing_flag, dt = _deadline_key(item)
            return (-score, missing_flag, dt)

        pool_sorted = sorted(pool, key=_top_sort_key)
        jobs_top = pool_sorted[:fallback_n]
        top_keys = {_job_key(j) for j in jobs_top}
        jobs_list = [j for j in pool if _job_key(j) not in top_keys]

    random_review = None
    jobs_visible = bool(jobs_top or jobs_list)
    if (
        cfg["email"].get("random_review_when_empty")
        and not papers_strong
        and not papers_normal
        and not jobs_visible
    ):
        random_review = db.fetch_random_or_oldest_strong_paper(days=30)

    llm_model_display = model_display
    if provider_display not in ("mixed", "n/a"):
        llm_model_display = (
            f"{provider_display}/{model_display}"
            if model_display not in ("", "n/a")
            else provider_display
        )

    context = {
        "run_id": run_id,
        "date": date_str,
        "papers_new": stats["papers_new"],
        "jobs_new": stats["jobs_new"],
        "error_count": stats["error_count"],
        "llm_calls": stats["llm_calls"],
        "llm_provider": provider_display,
        "llm_model": llm_model_display,
        "llm_input_tokens": stats["llm_input_tokens"],
        "llm_output_tokens": stats["llm_output_tokens"],
        "llm_total_tokens": stats["llm_total_tokens"],
        "llm_tokens_display": f"{stats['llm_input_tokens']:,}/{stats['llm_output_tokens']:,}/{stats['llm_total_tokens']:,}",
        "llm_calls_display": f"{stats['llm_calls']:,}",
        "llm_cost_usd": stats["llm_cost_usd"],
        "llm_cost_by_provider": cost_by_provider,
        "llm_cost_by_agent": {
            "papers": round(total_cost_papers, 6),
            "jobs": round(total_cost_jobs, 6),
        },
        "budget_hit": stats["budget_hit"],
        "duration_sec": stats["duration_sec"],
        "duration_str": duration_str,
        "output_language": (
            getattr(llm, "output_language", None)
            or cfg.get("llm", {}).get("output_language", "zh")
            or "zh"
        ).lower(),
        "papers_strong": papers_strong,
        "papers_normal": papers_normal,
        "jobs_strong": jobs_strong,
        "jobs_normal": jobs_normal,
        "jobs_upcoming": jobs_upcoming,
        "jobs_top": jobs_top,
        "jobs_list": jobs_list,
        "random_review": random_review,
    }

    subject = (
        cfg["email"]
        .get("subject_template", "[Daily Briefing][{date}]")
        .format(
            date=date_str,
            papers_strong=len(papers_strong),
            papers_total=len(papers_new_items),
            jobs_total=len(jobs_new_items),
        )
    )

    html_body, html_full, text_body = render_daily_brief(context)
    attach_html = bool(cfg["email"].get("attach_html", False))
    attachments = []
    if attach_html:
        filename_tpl = cfg["email"].get("attach_filename", "daily_brief_{run_id}.html")
        filename = str(filename_tpl).format(run_id=run_id, date=date_str)
        attachments.append(
            {
                "filename": filename,
                "content": html_full,
                "maintype": "text",
                "subtype": "html",
            }
        )

    # dry-run: write html preview
    if args.dry_run or env == "dev":
        out_path = Path("logs") / f"daily_brief_{run_id}.html"
        out_path.write_text(html_full, encoding="utf-8")
        out_email = Path("logs") / f"daily_brief_{run_id}_email.html"
        out_email.write_text(html_body, encoding="utf-8")
        log_event("dry_run_output", run_id=run_id, path=str(out_path))
        log_event("dry_run_output_email", run_id=run_id, path=str(out_email))
        print(f"[dry-run] rendered full html: {out_path}")
        print(f"[dry-run] rendered email html: {out_email}")
    else:
        # decide send
        should_send = bool(
            papers_new_items or jobs_new_items or cfg["email"].get("send_when_empty", True)
        )
        if should_send:
            try:
                from_user = os.getenv(cfg["email"]["from_env"]["user"], "").strip()
                from_pass = os.getenv(cfg["email"]["from_env"]["pass"], "").strip()
                to_addr = os.getenv(cfg["email"]["to_env"], "").strip()
                smtp_host = cfg["email"].get("smtp_host", "smtp.gmail.com")
                smtp_port = int(cfg["email"].get("smtp_port", 465))
                if not from_user or not from_pass or not to_addr:
                    raise RuntimeError("Missing EMAIL_USER/EMAIL_PASS/TARGET_EMAIL env vars")

                send_email(
                    smtp_host=smtp_host,
                    smtp_port=smtp_port,
                    user=from_user,
                    password=from_pass,
                    to_addr=to_addr,
                    subject=subject,
                    html_body=html_body,
                    text_body=text_body,
                    attachments=attachments,
                )
                log_event("email_sent", run_id=run_id, to=to_addr, subject=subject)
            except Exception as ex:
                stats["error_count"] += 1
                tb = traceback.format_exc()
                db.record_error(
                    run_id,
                    module="email",
                    source="smtp",
                    error_type=type(ex).__name__,
                    message=tb,
                )
                log_error("email_send_failed", run_id=run_id, error=repr(ex))

    ok = stats["error_count"] == 0
    db.finish_run(run_id, ok=ok, stats=stats)
    db.close()

    # Maintenance (after closing the main DB connection to avoid locks)
    try:
        maintenance_db(db_path, keep_error_days=int(runtime.get("keep_error_days", 30)))
        log_event("db_maintenance_done", run_id=run_id)
    except Exception as ex:
        stats["error_count"] += 1
        log_error("db_maintenance_failed", run_id=run_id, error=repr(ex))

    # Alert if failed (prod only, and not dry-run)
    if env == "prod" and not args.dry_run and not ok:
        try:
            from_user = os.getenv(cfg["email"]["from_env"]["user"], "").strip()
            from_pass = os.getenv(cfg["email"]["from_env"]["pass"], "").strip()
            alert_to = os.getenv(
                cfg["email"].get("alert_to_env", cfg["email"]["to_env"]), ""
            ).strip()
            smtp_host = cfg["email"].get("smtp_host", "smtp.gmail.com")
            smtp_port = int(cfg["email"].get("smtp_port", 465))

            text = f"""[ALERT] foryourseek run failed

date: {date_str}
run_id: {run_id}
errors: {stats['error_count']}
llm_calls: {stats['llm_calls']}
papers_new: {stats['papers_new']}
jobs_new: {stats['jobs_new']}

Please check Actions logs and logs/{log_path.name}.
"""
            if from_user and from_pass and alert_to:
                send_alert_email(
                    smtp_host=smtp_host,
                    smtp_port=smtp_port,
                    user=from_user,
                    password=from_pass,
                    to_addr=alert_to,
                    subject=f"[ALERT][{date_str}] foryourseek failed (run {run_id})",
                    text_body=text,
                )
                log_event("alert_sent", run_id=run_id, to=alert_to)
        except Exception as ex:
            log_error("alert_send_failed", run_id=run_id, error=repr(ex))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
