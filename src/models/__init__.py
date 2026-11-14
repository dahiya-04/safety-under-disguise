"""Model clients (targets and judges)."""

from src.models.base import GenerationResult, ModelClient
from src.models.openai_compat import OpenAICompatibleClient, build_target_client

__all__ = ["GenerationResult", "ModelClient", "OpenAICompatibleClient", "build_target_client"]
