"""LLM-based JD extractor for learning backend."""

from __future__ import annotations

import time
from typing import Callable, Literal, Union

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.prompts.jd_prompts import (
    JD_BASIC_INFO_PROMPT,
    JD_REQUIREMENTS_PROMPT,
    JD_SYSTEM_PROMPT,
    JD_VALIDITY_PROMPT,
)
from app.schemas.jd import JDData, JDBasicInfo, JDRequirements
from app.utils.structured_output import invoke_with_fallback


class InvalidJDContentError(RuntimeError):
    """Raised when input content is not a valid JD."""


class JDBasicInfoResponse(BaseModel):
    basicInfo: JDBasicInfo = Field(default_factory=JDBasicInfo)


class JDRequirementsResponse(BaseModel):
    requirements: JDRequirements = Field(default_factory=JDRequirements)


class JDValidityResponse(BaseModel):
    isJD: Literal["Yes", "No"] = Field(default="No")


class JDExtractor:
    """Request-scoped JD extractor with two-way parallel extraction."""

    def __init__(
        self,
        model_provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        settings = get_settings()
        active_config = settings.get_active_config()

        self.model_provider = model_provider or active_config["model_provider"]
        self.api_key = api_key or active_config["api_key"]
        self.base_url = base_url if base_url is not None else active_config["base_url"]
        self.model = model or active_config["model"]

        if self.model_provider != "openai":
            raise ValueError(
                "Learning skeleton 当前仅实现 openai provider。"
                "你可以在此基础上扩展 google_genai / anthropic。"
            )
        if not self.api_key:
            raise ValueError("缺少 API Key：请在 .env 中配置 OPENAI_API_KEY。")

        kwargs = {
            "model": self.model,
            "model_provider": "openai",
            "api_key": self.api_key,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url

        chat_model = init_chat_model(**kwargs)
        self.validity_llm = chat_model.with_structured_output(JDValidityResponse)
        self.basic_info_llm = chat_model.with_structured_output(JDBasicInfoResponse)
        self.requirements_llm = chat_model.with_structured_output(JDRequirementsResponse)

        self.parallel = RunnableParallel(
            basic_info=self._timed_chain(
                JD_BASIC_INFO_PROMPT,
                self.basic_info_llm,
                JDBasicInfoResponse,
                lambda raw: raw.basicInfo,
            ),
            requirements=self._timed_chain(
                JD_REQUIREMENTS_PROMPT,
                self.requirements_llm,
                JDRequirementsResponse,
                lambda raw: raw.requirements,
            ),
        )

    @staticmethod
    def _messages(user_content: str) -> list[Union[SystemMessage, HumanMessage]]:
        return [
            SystemMessage(content=JD_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

    def _timed_chain(
        self,
        prompt: str,
        llm,
        response_cls: type[BaseModel],
        extract_field: Callable[[BaseModel], object],
    ):
        def build_messages(jd_text: str):
            return self._messages(f"{prompt}\n\nJD 文本：\n{jd_text}")

        def run_with_time(messages):
            start = time.perf_counter()
            raw = invoke_with_fallback(llm, messages, response_cls)
            parsed = extract_field(raw)
            return parsed, time.perf_counter() - start

        return RunnableLambda(build_messages) | RunnableLambda(run_with_time)

    def _is_normal_jd(self, jd_text: str) -> bool:
        messages = self._messages(f"{JD_VALIDITY_PROMPT}\n\n待判断内容：\n{jd_text}")
        result = invoke_with_fallback(self.validity_llm, messages, JDValidityResponse)
        return result.isJD == "Yes"

    def extract_all(self, jd_text: str) -> tuple[JDData, float]:
        if not self._is_normal_jd(jd_text):
            raise InvalidJDContentError("上传内容不是一份正常的岗位 JD，请粘贴职位描述后重试。")

        start = time.perf_counter()
        parallel_result = self.parallel.invoke(jd_text)
        basic_info, _ = parallel_result["basic_info"]
        requirements, _ = parallel_result["requirements"]
        data = JDData(
            basicInfo=basic_info if basic_info else JDBasicInfo(),
            requirements=requirements if requirements else JDRequirements(),
        )
        elapsed = round(time.perf_counter() - start, 4)
        return data, elapsed

    def extract_to_dict(self, jd_text: str) -> tuple[dict, float]:
        data, elapsed = self.extract_all(jd_text)
        return data.model_dump(), elapsed
