from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(RuntimeError):
    pass


def _require(d: Dict[str, Any], key: str, path: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required config: {path}.{key}")
    return d[key]


def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(
            f"Config file not found: {p.resolve()}\n\nTip: copy config.example.yaml -> config.yaml"
        )
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    validate_config(data)
    return data


def validate_config(cfg: Dict[str, Any]) -> None:
    # Minimal validation (keep it forgiving; you can harden later with pydantic)
    _require(cfg, "profile", "")
    _require(cfg["profile"], "field", "profile")
    _require(cfg["profile"], "focus", "profile")
    cfg["profile"].setdefault("include_keywords", [])
    cfg["profile"].setdefault("exclude_keywords", [])

    _require(cfg, "papers", "")
    cfg["papers"].setdefault("max_per_day", 30)
    cfg["papers"].setdefault("strong_threshold", 8)
    cfg["papers"].setdefault("normal_threshold", 5)
    cfg["papers"].setdefault("journals", [])
    cfg["papers"].setdefault("max_candidates_per_run", cfg["papers"]["max_per_day"])
    cfg["papers"].setdefault("max_llm_per_run", cfg["papers"]["max_per_day"])
    cfg["papers"].setdefault("min_candidates_per_journal", 10)
    cfg["papers"].setdefault("prefilter", {})
    cfg["papers"]["prefilter"].setdefault("title_weight", 3)
    cfg["papers"]["prefilter"].setdefault("abstract_weight", 1)
    cfg["papers"]["prefilter"].setdefault("min_score_for_llm", 2)
    cfg["papers"]["prefilter"].setdefault("allow_if_no_keywords", True)

    _require(cfg, "jobs", "")
    cfg["jobs"].setdefault("max_per_day", 40)
    cfg["jobs"].setdefault("max_candidates_per_run", cfg["jobs"]["max_per_day"])
    cfg["jobs"].setdefault("max_llm_per_run", cfg["jobs"]["max_per_day"])
    cfg["jobs"].setdefault("degree_keywords", [])
    cfg["jobs"].setdefault("exclude_keywords", [])
    cfg["jobs"].setdefault("domain_keywords", [])
    cfg["jobs"].setdefault("prefilter", {})
    cfg["jobs"]["prefilter"].setdefault("title_weight", 3)
    cfg["jobs"]["prefilter"].setdefault("desc_weight", 1)
    cfg["jobs"]["prefilter"].setdefault("min_score_for_llm", 2)
    cfg["jobs"]["prefilter"].setdefault("allow_if_no_keywords", True)
    cfg["jobs"]["prefilter"].setdefault("keywords", [])
    cfg["jobs"].setdefault("sites", [])
    cfg["jobs"].setdefault(
        "dedup",
        {
            "method": "url",
            "semantic_key_fields": ["title", "org", "location", "deadline"],
        },
    )

    cfg.setdefault("extract", {"min_len": 120, "fallback": "trafilatura"})
    cfg.setdefault("llm", {})
    cfg["llm"].setdefault("provider", "openai")
    cfg["llm"].setdefault("model", "gpt-4o-mini")
    cfg["llm"].setdefault("temperature", 0)
    cfg["llm"].setdefault("max_calls_per_run", 50)
    cfg["llm"].setdefault("timeout_sec", 45)
    cfg["llm"].setdefault("max_input_chars_per_item", 8000)
    cfg["llm"].setdefault("on_budget_exceeded", "degrade")

    _require(cfg, "email", "")
    cfg["email"].setdefault("send_when_empty", True)
    cfg["email"].setdefault("random_review_when_empty", True)
    cfg["email"].setdefault(
        "subject_template",
        "[Daily Briefing][{date}] Papers: strong {papers_strong}/{papers_total} | PhD: {jobs_total}",
    )
    cfg["email"].setdefault("smtp_host", "smtp.gmail.com")
    cfg["email"].setdefault("smtp_port", 465)
    cfg["email"].setdefault("dev_override_to", "")
    _require(cfg["email"], "from_env", "email")
    _require(cfg["email"]["from_env"], "user", "email.from_env")
    _require(cfg["email"]["from_env"], "pass", "email.from_env")
    _require(cfg["email"], "to_env", "email")
    cfg["email"].setdefault("alert_to_env", cfg["email"]["to_env"])

    cfg.setdefault("runtime", {})
    cfg["runtime"].setdefault("env", "dev")
    cfg["runtime"].setdefault("timezone", "Asia/Tokyo")
    cfg["runtime"].setdefault("http_timeout_sec", 20)
    cfg["runtime"].setdefault("http_retries", 3)
    cfg["runtime"].setdefault("user_agent", "foryourseek/0.1")
    cfg["runtime"].setdefault("state_db_path", "state/history.db")
    cfg["runtime"].setdefault("keep_error_days", 30)
