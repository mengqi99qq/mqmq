"""Simple structured output helper with one retry."""

from __future__ import annotations

from typing import Any


def invoke_with_fallback(llm: Any, messages: list[Any], _schema_cls: type) -> Any:
    """Invoke structured output runnable and retry once on failure.

    This intentionally keeps logic small for learning:
    - First call: normal invoke
    - Retry once: same call (handles transient model/network instability)
    """
    try:
        return llm.invoke(messages)
    except Exception as first_error:
        try:
            return llm.invoke(messages)
        except Exception as second_error:
            raise RuntimeError(
                f"Structured extraction failed twice: {second_error}"
            ) from first_error
