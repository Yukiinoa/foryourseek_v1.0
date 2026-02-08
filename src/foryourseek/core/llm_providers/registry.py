from __future__ import annotations

from typing import Dict, Optional

from .base import ProviderInterface
from .gemini import GeminiProvider
from .minimax import MiniMaxProvider
from .openai import OpenAIProvider

_PROVIDERS: Dict[str, ProviderInterface] = {
    "openai": OpenAIProvider(),
    "gemini": GeminiProvider(),
    "minimax": MiniMaxProvider(),
}


def get_provider(name: str) -> Optional[ProviderInterface]:
    if not name:
        return None
    return _PROVIDERS.get(name.lower())


def list_providers() -> list[str]:
    return sorted(_PROVIDERS.keys())
