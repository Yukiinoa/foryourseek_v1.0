from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass
class EmailConfig:
    subject_template: str
    send_when_empty: bool
    random_review_when_empty: bool
    dev_override_to: str
    from_user_env: str
    from_pass_env: str
    to_env: str
    alert_to_env: str


def _jinja_env() -> Environment:
    template_dir = Path(__file__).parent / "templates"
    return Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def render_daily_brief(context: Dict[str, Any]) -> tuple[str, str, str]:
    env = _jinja_env()
    html_email = env.get_template("daily_brief_email.html.j2").render(**context)
    html_full = env.get_template("daily_brief.html.j2").render(**context)
    text = env.get_template("daily_brief.txt.j2").render(**context)
    return html_email, html_full, text


def send_email(
    *,
    smtp_host: str,
    smtp_port: int,
    user: str,
    password: str,
    to_addr: str,
    subject: str,
    html_body: str,
    text_body: str,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> None:
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    if attachments:
        for att in attachments:
            if not isinstance(att, dict):
                continue
            content = att.get("content")
            if content is None:
                continue
            filename = att.get("filename") or "attachment"
            maintype = str(att.get("maintype") or "text")
            subtype = str(att.get("subtype") or "plain")
            if maintype == "text" and isinstance(content, str):
                msg.add_attachment(content, subtype=subtype, filename=filename)
            else:
                if isinstance(content, str):
                    content = content.encode("utf-8")
                msg.add_attachment(content, maintype=maintype, subtype=subtype, filename=filename)

    # Most providers support SSL 465; if you need STARTTLS, adjust here.
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(user, password)
        server.send_message(msg)


def send_alert_email(
    *,
    smtp_host: str,
    smtp_port: int,
    user: str,
    password: str,
    to_addr: str,
    subject: str,
    text_body: str,
) -> None:
    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(text_body)

    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
        server.login(user, password)
        server.send_message(msg)


def resolve_recipients(cfg: Dict[str, Any]) -> tuple[str, str, str]:
    """Returns (from_user, from_pass, to_addr) based on env vars and dev override."""
    from_user = os.getenv(cfg["from_env"]["user"], "").strip()
    from_pass = os.getenv(cfg["from_env"]["pass"], "").strip()
    to_addr = os.getenv(cfg["to_env"], "").strip()

    env = (cfg.get("runtime_env") or "dev").lower()
    dev_override = (cfg.get("dev_override_to") or "").strip()
    if env == "dev" and dev_override:
        to_addr = dev_override

    return from_user, from_pass, to_addr
