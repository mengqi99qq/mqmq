"""Runtime model configuration schema."""

from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    """Optional request-level runtime overrides."""

    modelProvider: str | None = Field(default=None)
    apiKey: str | None = Field(default=None)
    baseURL: str | None = Field(default=None)
    model: str | None = Field(default=None)
    ocrModel: str | None = Field(default=None)
