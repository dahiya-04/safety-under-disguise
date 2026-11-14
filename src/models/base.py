"""Model client interface shared by all evaluation modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass
class GenerationResult:
    text: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    reasoning_text: str | None = None  # reasoning models only; tail kept for diagnosis
    latency_s: float | None = None
    error: str | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return asdict(self)


class ModelClient(ABC):
    """A chat model that turns a single user prompt into a response."""

    model: str

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """Send ``prompt`` as a single user turn and return the response."""
