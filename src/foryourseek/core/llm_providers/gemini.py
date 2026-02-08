from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from .base import ProviderInterface


class GeminiProvider(ProviderInterface):
    name = "gemini"

    def ready(self, cfg: Dict[str, Any]) -> bool:
        envs = cfg.get("api_key_envs") or ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        return any(os.getenv(env) for env in envs)

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
        from google import genai

        envs = cfg.get("api_key_envs") or ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        api_key = ""
        for env in envs:
            v = os.getenv(env)
            if v:
                api_key = v
                break
        client = genai.Client(api_key=api_key) if api_key else genai.Client()

        config: Dict[str, Any] = {"temperature": float(temperature)}
        use_schema = bool(cfg.get("use_schema", True))
        if use_schema and response_schema:
            config["response_mime_type"] = "application/json"
            config["response_json_schema"] = response_schema

        try:
            resp = client.models.generate_content(model=model, contents=user_prompt, config=config)
        except Exception:
            # Fallback: retry without schema in case of incompatibility
            if response_schema and use_schema:
                config.pop("response_json_schema", None)
                config.pop("response_mime_type", None)
                resp = client.models.generate_content(
                    model=model, contents=user_prompt, config=config
                )
            else:
                raise

        content = getattr(resp, "text", "") or ""
        usage = {}
        u = getattr(resp, "usage_metadata", None) or getattr(resp, "usage", None)
        if u:
            usage = {
                "input": getattr(u, "prompt_token_count", None)
                or (u.get("prompt_token_count") if isinstance(u, dict) else None)
                or 0,
                "output": getattr(u, "candidates_token_count", None)
                or (u.get("candidates_token_count") if isinstance(u, dict) else None)
                or 0,
                "total": getattr(u, "total_token_count", None)
                or (u.get("total_token_count") if isinstance(u, dict) else None)
                or 0,
            }
        return content.strip(), usage
