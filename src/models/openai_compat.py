"""Client for any OpenAI-compatible chat endpoint (HF router, Goedel, vLLM, ...)."""

from __future__ import annotations

import os
import time
from typing import Any

from src.models.base import GenerationResult, ModelClient
from src.utils.cache import JSONLCache, stable_hash

# Errors worth retrying: rate limits, timeouts, transient server errors.
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_REASONING_KEEP_CHARS = 3000  # keep the TAIL: that's where loops / truncation show up


def _extract_reasoning(message: Any) -> str | None:
    """Reasoning text from providers that expose it (field name varies)."""
    extra = getattr(message, "model_extra", None) or {}
    for name in ("reasoning_content", "reasoning"):
        value = getattr(message, name, None) or extra.get(name)
        if isinstance(value, str) and value.strip():
            value = value.strip()
            return value if len(value) <= _REASONING_KEEP_CHARS else "..." + value[-_REASONING_KEEP_CHARS:]
    return None


class OpenAICompatibleClient(ModelClient):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: str | None = None,
        timeout_s: float = 120,
        max_retries: int = 4,
        extra_body: dict[str, Any] | None = None,
        cache: JSONLCache | None = None,
        client: Any = None,
        **_: Any,  # ignore unrelated config keys (e.g. pricing)
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.extra_body = extra_body or {}
        self.cache = cache

        if client is None:
            from openai import OpenAI  # lazy import

            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(f"Environment variable {api_key_env} is not set (check .env).")
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.client = client

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        return msgs + [{"role": "user", "content": prompt}]

    def cache_key(self, prompt: str) -> str:
        return stable_hash(self.base_url, self.model, self._messages(prompt),
                           self.temperature, self.max_tokens, self.extra_body)

    def generate(self, prompt: str) -> GenerationResult:
        key = self.cache_key(prompt)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return GenerationResult(**{**hit, "cached": True})

        last_error = ""
        for attempt in range(self.max_retries):
            t0 = time.monotonic()
            try:
                kwargs: dict[str, Any] = dict(model=self.model, messages=self._messages(prompt),
                                              temperature=self.temperature, max_tokens=self.max_tokens)
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                resp = self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                usage = getattr(resp, "usage", None)
                details = getattr(usage, "completion_tokens_details", None) if usage else None
                result = GenerationResult(
                    text=(choice.message.content or "").strip(),
                    model=self.model,
                    finish_reason=choice.finish_reason,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
                    reasoning_text=_extract_reasoning(choice.message),
                    latency_s=round(time.monotonic() - t0, 2),
                )
                if self.cache is not None:
                    self.cache.put(key, result.to_dict())
                return result
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                last_error = f"{type(exc).__name__}({status}): {str(exc)[:200]}"
                if status is not None and status not in _RETRYABLE_STATUS:
                    break  # e.g. 400/401/404: retrying will not help
                time.sleep(min(60, 2 ** (attempt + 1)))

        # Errors are NOT cached, so a re-run retries them.
        return GenerationResult(text="", model=self.model, error=last_error)


def build_target_client(model_cfg: dict[str, Any], defaults: dict[str, Any],
                        cache: JSONLCache | None = None) -> OpenAICompatibleClient:
    """Merge per-model config over defaults and build a client."""
    cfg = {**defaults, **model_cfg}
    model_id = cfg.pop("id")
    return OpenAICompatibleClient(model=model_id, cache=cache, **cfg)
