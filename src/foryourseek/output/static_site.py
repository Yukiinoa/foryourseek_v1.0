from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def build_static_site(output_dir: str, context: Dict[str, Any]) -> None:
    """Optional: generate a tiny static archive (for GitHub Pages).

    Privacy warning: if you publish it, others can see your interests.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Minimal index page
    html = """<!doctype html><html><head><meta charset='utf-8'><title>foryourseek archive</title></head><body>
    <h2>foryourseek archive</h2>
    <p>This is an optional feature. Implement your own pages here.</p>
    </body></html>"""
    (out / "index.html").write_text(html, encoding="utf-8")
