from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageError:
    stage: str
    error: str
    fallback_to: str | None = None

    def as_dict(self) -> dict[str, str]:
        payload = {"stage": self.stage, "error": self.error}
        if self.fallback_to:
            payload["fallback_to"] = self.fallback_to
        return payload

