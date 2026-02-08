from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def setup_logging(run_id: str, log_dir: str = "logs", level: str = "INFO") -> Path:
    """Log to stdout + a jsonl file for later upload as artifact."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"run_{run_id}.jsonl"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(root.level)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    # file handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(root.level)
    fh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(fh)

    # Attach for convenience
    logging.getLogger(__name__).info(
        json.dumps(
            {
                "ts": _ts(),
                "level": "INFO",
                "event": "logging_initialized",
                "run_id": run_id,
                "log_path": str(log_path),
            },
            ensure_ascii=False,
        )
    )

    return log_path


def _ts() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def log_event(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"ts": _ts(), "event": event, **fields}
    logging.getLogger("foryourseek").info(json.dumps(payload, ensure_ascii=False))


def log_error(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"ts": _ts(), "level": "ERROR", "event": event, **fields}
    logging.getLogger("foryourseek").error(json.dumps(payload, ensure_ascii=False))
