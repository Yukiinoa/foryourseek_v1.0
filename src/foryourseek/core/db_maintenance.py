from __future__ import annotations

import sqlite3


def maintenance_db(db_path: str, keep_error_days: int = 30) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()

        cur.execute(
            "DELETE FROM run_errors WHERE created_at < date('now', ?)",
            (f"-{int(keep_error_days)} days",),
        )

        # Optional: prune old non-strong items to control DB size
        # cur.execute("DELETE FROM papers WHERE created_at < date('now', '-180 days') AND recommendation != 'strong'")
        # cur.execute("DELETE FROM jobs WHERE created_at < date('now', '-180 days') AND recommendation != 'strong'")

        # VACUUM must run outside any active transaction.
        conn.commit()
        cur.execute("VACUUM")
    finally:
        conn.close()
