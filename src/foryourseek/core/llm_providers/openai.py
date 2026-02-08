from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from .base import ProviderInterface


class OpenAIProvider(ProviderInterface):
    name = "openai"

    def ready(self, cfg: Dict[str, Any]) -> bool:
        env_key = cfg.get("api_key_env", "OPENAI_API_KEY")
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
        from openai import OpenAI

        api_key = os.getenv(cfg.get("api_key_env", "OPENAI_API_KEY"), "")
        api_base = cfg.get("api_base") or None
        client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)

        messages = [
            {
                "role": "system",
                "content": "You are a precise academic assistant. Output JSON only, with no extra text.",
            },
            {"role": "user", "content": user_prompt},
        ]

        response_format = cfg.get("response_format", "json_object")
        use_response_format = bool(cfg.get("use_response_format", True))
        try:
            kwargs: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout_sec,
            )
            if use_response_format and response_format:
                kwargs["response_format"] = {"type": response_format}
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            # Fallback: retry without response_format
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout_sec,
            )

        content = resp.choices[0].message.content or ""
        usage = {}
        u = getattr(resp, "usage", None)
        if u:
            usage = {
                "input": getattr(u, "prompt_tokens", None)
                or (u.get("prompt_tokens") if isinstance(u, dict) else None)
                or 0,
                "output": getattr(u, "completion_tokens", None)
                or (u.get("completion_tokens") if isinstance(u, dict) else None)
                or 0,
                "total": getattr(u, "total_tokens", None)
                or (u.get("total_tokens") if isinstance(u, dict) else None)
                or 0,
            }
        return content.strip(), usage
