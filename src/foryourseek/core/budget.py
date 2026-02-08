from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    max_calls: int
    calls_used: int = 0

    def can_call(self) -> bool:
        return self.calls_used < self.max_calls

    def consume_call(self, n: int = 1) -> None:
        self.calls_used += n
