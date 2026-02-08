from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple


class ProviderInterface(Protocol):
    name: str

    def ready(self, cfg: Dict[str, Any]) -> bool: ...

    def call_json(
        self,
        *,
        model: str,
        temperature: float,
        timeout_sec: int,
        user_prompt: str,
        response_schema: Optional[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Tuple[str, Dict[str, int]]: ...
