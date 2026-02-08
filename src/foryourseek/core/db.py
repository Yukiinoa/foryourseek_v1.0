from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .utils import now_utc_iso

SCHEMA_SQL = """

CREATE TABLE IF NOT EXISTS seen_items (
  id TEXT PRIMARY KEY,
  item_type TEXT NOT NULL,
  first_seen_at TEXT NOT NULL,
  source TEXT
);

CREATE TABLE IF NOT EXISTS papers (
  id TEXT PRIMARY KEY,
  title TEXT,
  url TEXT,
  doi TEXT,
  published_at TEXT,
  journal TEXT,
  authors TEXT,
  abstract TEXT,
  abstract_source TEXT,
  extract_method TEXT,
  extract_warnings TEXT,
  summary_bullets TEXT,
  relevance_score INTEGER,
  recommendation TEXT,
  reasons TEXT,
  llm_status TEXT,
  llm_input_chars INTEGER,
  llm_input_tokens INTEGER,
  llm_output_tokens INTEGER,
  llm_total_tokens INTEGER,
  budget_hit INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  title TEXT,
  org TEXT,
  location TEXT,
  url TEXT,
  posted_at TEXT,
  deadline TEXT,
  site TEXT,
  department TEXT,
  research_field TEXT,
  researcher_profile TEXT,
  positions TEXT,
  education_level TEXT,
  contract_type TEXT,
  job_status TEXT,
  funding_programme TEXT,
  salary TEXT,
  reference_number TEXT,
  languages TEXT,
  description TEXT,
  extract_method TEXT,
  extract_warnings TEXT,
  summary_bullets TEXT,
  relevance_score INTEGER,
  recommendation TEXT,
  reasons TEXT,
  llm_status TEXT,
  llm_input_chars INTEGER,
  llm_input_tokens INTEGER,
  llm_output_tokens INTEGER,
  llm_total_tokens INTEGER,
  budget_hit INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  ok INTEGER NOT NULL,
  duration_sec REAL,
  papers_found INTEGER,
  papers_new INTEGER,
  papers_strong INTEGER,
  jobs_found INTEGER,
  jobs_new INTEGER,
  llm_calls INTEGER,
  llm_input_tokens INTEGER,
  llm_output_tokens INTEGER,
  llm_total_tokens INTEGER,
  budget_hit INTEGER,
  error_count INTEGER
);

CREATE TABLE IF NOT EXISTS run_errors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  module TEXT,
  source TEXT,
  error_type TEXT,
  message TEXT,
  created_at TEXT NOT NULL
);
"""


@dataclass
class PaperRow:
    id: str
    title: str
    url: str
    doi: str
    published_at: str
    journal: str
    authors: str
    abstract: str
    abstract_source: str
    extract_method: str
    extract_warnings: List[str]
    summary_bullets: List[str]
    relevance_score: int
    recommendation: str
    reasons: List[str]
    llm_status: str
    llm_input_chars: int
    llm_input_tokens: int
    llm_output_tokens: int
    llm_total_tokens: int
    budget_hit: int


@dataclass
class JobRow:
    id: str
    title: str
    org: str
    location: str
    url: str
    posted_at: str
    deadline: str
    site: str
    department: str
    research_field: str
    researcher_profile: str
    positions: str
    education_level: str
    contract_type: str
    job_status: str
    funding_programme: str
    salary: str
    reference_number: str
    languages: str
    description: str
    extract_method: str
    extract_warnings: List[str]
    summary_bullets: List[str]
    relevance_score: int
    recommendation: str
    reasons: List[str]
    llm_status: str
    llm_input_chars: int
    llm_input_tokens: int
    llm_output_tokens: int
    llm_total_tokens: int
    budget_hit: int


class Database:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=DELETE;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        self._rename_column_if_exists("papers", "summary_zh", "summary_bullets")
        self._rename_column_if_exists("papers", "reasons_zh", "reasons")
        self._rename_column_if_exists("jobs", "summary_zh", "summary_bullets")
        self._rename_column_if_exists("jobs", "reasons_zh", "reasons")
        self._ensure_column("papers", "abstract_source", "TEXT")
        self._ensure_column("papers", "extract_method", "TEXT")
        self._ensure_column("papers", "extract_warnings", "TEXT")
        self._ensure_column("papers", "summary_bullets", "TEXT")
        self._ensure_column("papers", "reasons", "TEXT")
        self._ensure_column("papers", "llm_input_chars", "INTEGER")
        self._ensure_column("papers", "llm_input_tokens", "INTEGER")
        self._ensure_column("papers", "llm_output_tokens", "INTEGER")
        self._ensure_column("papers", "llm_total_tokens", "INTEGER")
        self._ensure_column("papers", "budget_hit", "INTEGER")
        self._ensure_column("jobs", "description", "TEXT")
        self._ensure_column("jobs", "department", "TEXT")
        self._ensure_column("jobs", "research_field", "TEXT")
        self._ensure_column("jobs", "researcher_profile", "TEXT")
        self._ensure_column("jobs", "positions", "TEXT")
        self._ensure_column("jobs", "education_level", "TEXT")
        self._ensure_column("jobs", "contract_type", "TEXT")
        self._ensure_column("jobs", "job_status", "TEXT")
        self._ensure_column("jobs", "funding_programme", "TEXT")
        self._ensure_column("jobs", "salary", "TEXT")
        self._ensure_column("jobs", "reference_number", "TEXT")
        self._ensure_column("jobs", "languages", "TEXT")
        self._ensure_column("jobs", "extract_method", "TEXT")
        self._ensure_column("jobs", "extract_warnings", "TEXT")
        self._ensure_column("jobs", "summary_bullets", "TEXT")
        self._ensure_column("jobs", "reasons", "TEXT")
        self._ensure_column("jobs", "llm_input_chars", "INTEGER")
        self._ensure_column("jobs", "llm_input_tokens", "INTEGER")
        self._ensure_column("jobs", "llm_output_tokens", "INTEGER")
        self._ensure_column("jobs", "llm_total_tokens", "INTEGER")
        self._ensure_column("jobs", "budget_hit", "INTEGER")
        self._ensure_column("runs", "duration_sec", "REAL")
        self._ensure_column("runs", "llm_input_tokens", "INTEGER")
        self._ensure_column("runs", "llm_output_tokens", "INTEGER")
        self._ensure_column("runs", "llm_total_tokens", "INTEGER")
        self._ensure_column("runs", "budget_hit", "INTEGER")

    def close(self) -> None:
        self.conn.close()

    # --- run log ---
    def create_run(self, run_id: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO runs(run_id, started_at, ok) VALUES (?, ?, 0)",
            (run_id, now_utc_iso()),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, *, ok: bool, stats: Dict[str, Any]) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, ok=?, duration_sec=?, papers_found=?, papers_new=?, papers_strong=?, jobs_found=?, jobs_new=?, llm_calls=?, llm_input_tokens=?, llm_output_tokens=?, llm_total_tokens=?, budget_hit=?, error_count=? WHERE run_id=?",
            (
                now_utc_iso(),
                1 if ok else 0,
                float(stats.get("duration_sec", 0.0)),
                int(stats.get("papers_found", 0)),
                int(stats.get("papers_new", 0)),
                int(stats.get("papers_strong", 0)),
                int(stats.get("jobs_found", 0)),
                int(stats.get("jobs_new", 0)),
                int(stats.get("llm_calls", 0)),
                int(stats.get("llm_input_tokens", 0)),
                int(stats.get("llm_output_tokens", 0)),
                int(stats.get("llm_total_tokens", 0)),
                int(stats.get("budget_hit", 0)),
                int(stats.get("error_count", 0)),
                run_id,
            ),
        )
        self.conn.commit()

    def record_error(
        self, run_id: str, *, module: str, source: str, error_type: str, message: str
    ) -> None:
        self.conn.execute(
            "INSERT INTO run_errors(run_id, module, source, error_type, message, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, module, source, error_type, message[:2000], now_utc_iso()),
        )
        self.conn.commit()

    # --- seen / dedup ---
    def is_seen(self, item_id: str) -> bool:
        cur = self.conn.execute("SELECT 1 FROM seen_items WHERE id=? LIMIT 1", (item_id,))
        return cur.fetchone() is not None

    def is_seen_any(self, item_ids: Iterable[str]) -> bool:
        ids = [i for i in item_ids if i]
        if not ids:
            return False
        placeholders = ",".join(["?"] * len(ids))
        cur = self.conn.execute(
            f"SELECT 1 FROM seen_items WHERE id IN ({placeholders}) LIMIT 1", ids
        )
        return cur.fetchone() is not None

    def find_seen_ids(self, item_ids: Iterable[str]) -> List[str]:
        ids = [i for i in item_ids if i]
        if not ids:
            return []
        placeholders = ",".join(["?"] * len(ids))
        cur = self.conn.execute(f"SELECT id FROM seen_items WHERE id IN ({placeholders})", ids)
        return [r[0] for r in cur.fetchall()]

    def mark_seen(self, item_id: str, *, item_type: str, source: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO seen_items(id, item_type, first_seen_at, source) VALUES (?, ?, ?, ?)",
            (item_id, item_type, now_utc_iso(), source),
        )
        self.conn.commit()

    def mark_seen_many(self, item_ids: Iterable[str], *, item_type: str, source: str) -> None:
        ids = [i for i in item_ids if i]
        if not ids:
            return
        now = now_utc_iso()
        self.conn.executemany(
            "INSERT OR IGNORE INTO seen_items(id, item_type, first_seen_at, source) VALUES (?, ?, ?, ?)",
            [(i, item_type, now, source) for i in ids],
        )
        self.conn.commit()

    # --- insert items ---
    def upsert_paper(self, row: PaperRow) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO papers(
                id,title,url,doi,published_at,journal,authors,abstract,abstract_source,extract_method,extract_warnings,
                summary_bullets,relevance_score,recommendation,reasons,llm_status,llm_input_chars,llm_input_tokens,llm_output_tokens,llm_total_tokens,budget_hit,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row.id,
                row.title,
                row.url,
                row.doi,
                row.published_at,
                row.journal,
                row.authors,
                row.abstract,
                row.abstract_source,
                row.extract_method,
                json.dumps(row.extract_warnings, ensure_ascii=False),
                json.dumps(row.summary_bullets, ensure_ascii=False),
                int(row.relevance_score),
                row.recommendation,
                json.dumps(row.reasons, ensure_ascii=False),
                row.llm_status,
                int(row.llm_input_chars),
                int(row.llm_input_tokens),
                int(row.llm_output_tokens),
                int(row.llm_total_tokens),
                int(row.budget_hit),
                now_utc_iso(),
            ),
        )
        self.conn.commit()

    def upsert_job(self, row: JobRow) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO jobs(
                id,title,org,location,url,posted_at,deadline,site,
                department,research_field,researcher_profile,positions,education_level,contract_type,job_status,funding_programme,salary,reference_number,languages,
                description,extract_method,extract_warnings,
                summary_bullets,relevance_score,recommendation,reasons,llm_status,llm_input_chars,llm_input_tokens,llm_output_tokens,llm_total_tokens,budget_hit,created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                row.id,
                row.title,
                row.org,
                row.location,
                row.url,
                row.posted_at,
                row.deadline,
                row.site,
                row.department,
                row.research_field,
                row.researcher_profile,
                row.positions,
                row.education_level,
                row.contract_type,
                row.job_status,
                row.funding_programme,
                row.salary,
                row.reference_number,
                row.languages,
                row.description,
                row.extract_method,
                json.dumps(row.extract_warnings, ensure_ascii=False),
                json.dumps(row.summary_bullets, ensure_ascii=False),
                int(row.relevance_score),
                row.recommendation,
                json.dumps(row.reasons, ensure_ascii=False),
                row.llm_status,
                int(row.llm_input_chars),
                int(row.llm_input_tokens),
                int(row.llm_output_tokens),
                int(row.llm_total_tokens),
                int(row.budget_hit),
                now_utc_iso(),
            ),
        )
        self.conn.commit()

    # --- queries for email ---
    def fetch_recent_papers(self, limit: int = 50) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT id,title,url,doi,published_at,journal,abstract,abstract_source,extract_method,extract_warnings,summary_bullets,relevance_score,recommendation,reasons,llm_status,llm_input_chars,llm_input_tokens,llm_output_tokens,llm_total_tokens,budget_hit FROM papers ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "id": r[0],
                    "title": r[1],
                    "url": r[2],
                    "doi": r[3],
                    "published_at": r[4],
                    "journal": r[5],
                    "abstract": r[6],
                    "abstract_source": r[7],
                    "extract_method": r[8],
                    "extract_warnings": _loads_json_list(r[9]),
                    "summary_bullets": _loads_json_list(r[10]),
                    "relevance_score": r[11],
                    "recommendation": r[12],
                    "reasons": _loads_json_list(r[13]),
                    "llm_status": r[14],
                    "llm_input_chars": r[15],
                    "llm_input_tokens": r[16],
                    "llm_output_tokens": r[17],
                    "llm_total_tokens": r[18],
                    "budget_hit": r[19],
                }
            )
        return rows

    def _ensure_column(self, table: str, column: str, col_type: str) -> None:
        cols = self._columns(table)
        if column not in cols:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            self.conn.commit()

    def _columns(self, table: str) -> set[str]:
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        return {r[1] for r in cur.fetchall()}

    def _rename_column_if_exists(self, table: str, old: str, new: str) -> None:
        cols = self._columns(table)
        if old not in cols or new in cols:
            return
        try:
            self.conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
            self.conn.commit()
        except Exception:
            # Fallback: add new column and copy values
            self._ensure_column(table, new, "TEXT")
            self.conn.execute(f"UPDATE {table} SET {new}={old} WHERE {new} IS NULL OR {new}=''")
            self.conn.commit()

    def fetch_recent_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT id,title,org,location,url,posted_at,deadline,site,department,research_field,researcher_profile,positions,education_level,contract_type,job_status,funding_programme,salary,reference_number,languages,description,extract_method,extract_warnings,summary_bullets,relevance_score,recommendation,reasons,llm_status,llm_input_chars,llm_input_tokens,llm_output_tokens,llm_total_tokens,budget_hit FROM jobs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "id": r[0],
                    "title": r[1],
                    "org": r[2],
                    "location": r[3],
                    "url": r[4],
                    "posted_at": r[5],
                    "deadline": r[6],
                    "site": r[7],
                    "department": r[8],
                    "research_field": r[9],
                    "researcher_profile": r[10],
                    "positions": r[11],
                    "education_level": r[12],
                    "contract_type": r[13],
                    "job_status": r[14],
                    "funding_programme": r[15],
                    "salary": r[16],
                    "reference_number": r[17],
                    "languages": r[18],
                    "description": r[19],
                    "extract_method": r[20],
                    "extract_warnings": _loads_json_list(r[21]),
                    "summary_bullets": _loads_json_list(r[22]),
                    "relevance_score": r[23],
                    "recommendation": r[24],
                    "reasons": _loads_json_list(r[25]),
                    "llm_status": r[26],
                    "llm_input_chars": r[27],
                    "llm_input_tokens": r[28],
                    "llm_output_tokens": r[29],
                    "llm_total_tokens": r[30],
                    "budget_hit": r[31],
                }
            )
        return rows

    def fetch_random_strong_paper_older_than_days(self, days: int = 30) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            """SELECT title,url,summary_bullets,reasons,published_at,journal
               FROM papers
               WHERE recommendation='strong'
                 AND created_at < date('now', ?)
               ORDER BY RANDOM()
               LIMIT 1""",
            (f"-{int(days)} days",),
        )
        r = cur.fetchone()
        if not r:
            return None
        return {
            "title": r[0],
            "url": r[1],
            "summary_bullets": _loads_json_list(r[2]),
            "reasons": _loads_json_list(r[3]),
            "published_at": r[4],
            "journal": r[5],
        }

    def fetch_random_or_oldest_strong_paper(self, days: int = 30) -> Optional[Dict[str, Any]]:
        """Prefer random strong older than N days; fallback to the oldest strong."""
        r = self.fetch_random_strong_paper_older_than_days(days=days)
        if r:
            return r
        cur = self.conn.execute("""SELECT title,url,summary_bullets,reasons,published_at,journal
               FROM papers
               WHERE recommendation='strong'
               ORDER BY created_at ASC
               LIMIT 1""")
        r = cur.fetchone()
        if not r:
            return None
        return {
            "title": r[0],
            "url": r[1],
            "summary_bullets": _loads_json_list(r[2]),
            "reasons": _loads_json_list(r[3]),
            "published_at": r[4],
            "journal": r[5],
        }


def _loads_json_list(s: Any) -> List[str]:
    if not s:
        return []
    try:
        v = json.loads(s)
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return []
