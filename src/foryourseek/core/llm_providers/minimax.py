from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

from .base import ProviderInterface


class MiniMaxProvider(ProviderInterface):
    name = "minimax"

    def ready(self, cfg: Dict[str, Any]) -> bool:
        env_key = cfg.get("api_key_env", "MINIMAX_API_KEY")
        return bool(os.getenv(env_key))

    def call_json(
        self,
        *,
        model: str,
        temperature: float,
        timeout_sec: int,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int]]:
        api_key = os.getenv(cfg.get("api_key_env", "MINIMAX_API_KEY"), "")
        api_base = (cfg.get("api_base") or "https://api.minimaxi.com/v1").rstrip("/")
        url = f"{api_base}/chat/completions"

        messages = [
            {
                "role": "system",
                "content": "You are a precise academic assistant. Output JSON only, with no extra text.",
            },
            {"role": "user", "content": user_prompt},
        ]

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
        }
        response_format = cfg.get("response_format", "json_object")
        use_response_format = bool(cfg.get("use_response_format", True))
        if use_response_format and response_format:
            payload["response_format"] = {"type": response_format}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
        if resp.status_code >= 400:
            raise RuntimeError(f"MiniMax HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("MiniMax response missing choices")
        content = (choices[0].get("message") or {}).get("content") or ""

        usage = data.get("usage") or {}
        return (
            content.strip(),
            {
                "input": int(usage.get("prompt_tokens") or 0),
                "output": int(usage.get("completion_tokens") or 0),
                "total": int(usage.get("total_tokens") or 0),
            },
        )
