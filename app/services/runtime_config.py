"""Helpers to merge runtime overrides with defaults."""

from dataclasses import dataclass

from app.core.config import get_settings
from app.schemas.runtime_config import RuntimeConfig

@dataclass(frozen=True, slots=True)
class ResolvedRuntimeConfig:
    """Effective model configuration after merge."""

    model_provider: str
    api_key: str | None
    base_url: str | None
    model: str | None
    ocr_model: str | None


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def resolve_runtime_config(runtime_config: RuntimeConfig | None = None) -> ResolvedRuntimeConfig:
    """Merge request overrides into env defaults.

    This is intentionally simple for learning:
    - No provider-specific validation
    - No secret masking
    - Keeps only LLM/OCR fields needed in current skeleton
    """
    settings = get_settings()
    runtime_config = runtime_config or RuntimeConfig()
    resolved = ResolvedRuntimeConfig(
        model_provider=_normalize_optional(runtime_config.modelProvider)
        or settings.model_provider,
        api_key=_normalize_optional(runtime_config.apiKey) or settings.openai_api_key,
        base_url=_normalize_optional(runtime_config.baseURL) or settings.openai_base_url,
        model=_normalize_optional(runtime_config.model) or settings.openai_model,
        ocr_model=_normalize_optional(runtime_config.ocrModel) or settings.dashscope_ocr_model,
    )
    return resolved
