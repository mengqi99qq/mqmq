"""LLM-based resume extractor for learning."""

from __future__ import annotations

import time
from typing import Callable, Union

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.prompts.resume_prompts import (
    ACADEMIC_ACHIEVEMENTS_PROMPT,
    BASIC_INFO_PROMPT,
    EDUCATION_PROMPT,
    PROJECT_PROMPT,
    SYSTEM_PROMPT,
    WORK_EXPERIENCE_PROMPT,
)
from app.schemas.resume import (
    AcademicAchievementItem,
    BasicInfo,
    EducationItem,
    ProjectItem,
    ResumeData,
    WorkExperienceItem,
)
from app.utils.structured_output import invoke_with_fallback


class BasicInfoResponse(BaseModel):
    basicInfo: BasicInfo = Field(default_factory=BasicInfo)


class WorkExperienceResponse(BaseModel):
    workExperience: list[WorkExperienceItem] = Field(default_factory=list)


class EducationResponse(BaseModel):
    education: list[EducationItem] = Field(default_factory=list)


class ProjectResponse(BaseModel):
    projects: list[ProjectItem] = Field(default_factory=list)


class AcademicAchievementsResponse(BaseModel):
    academicAchievements: list[AcademicAchievementItem] = Field(default_factory=list)


class ResumeExtractor:
    """Request-scoped resume extractor with structured LLM output."""

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
        # 为 5 个字段创建独立的 structured output runnable，后续并行执行。
        self.basic_info_llm = chat_model.with_structured_output(BasicInfoResponse)
        self.work_llm = chat_model.with_structured_output(WorkExperienceResponse)
        self.edu_llm = chat_model.with_structured_output(EducationResponse)
        self.project_llm = chat_model.with_structured_output(ProjectResponse)
        self.academic_llm = chat_model.with_structured_output(AcademicAchievementsResponse)

        self.parallel = RunnableParallel(
            basic_info=self._timed_chain(
                BASIC_INFO_PROMPT,
                self.basic_info_llm,
                BasicInfoResponse,
                lambda raw: raw.basicInfo,
            ),
            work_experience=self._timed_chain(
                WORK_EXPERIENCE_PROMPT,
                self.work_llm,
                WorkExperienceResponse,
                lambda raw: raw.workExperience,
            ),
            education=self._timed_chain(
                EDUCATION_PROMPT,
                self.edu_llm,
                EducationResponse,
                lambda raw: raw.education,
            ),
            projects=self._timed_chain(
                PROJECT_PROMPT,
                self.project_llm,
                ProjectResponse,
                lambda raw: raw.projects,
            ),
            academic_achievements=self._timed_chain(
                ACADEMIC_ACHIEVEMENTS_PROMPT,
                self.academic_llm,
                AcademicAchievementsResponse,
                lambda raw: raw.academicAchievements,
            ),
        )

    @staticmethod
    def _messages(user_content: str) -> list[Union[SystemMessage, HumanMessage]]:
        """Create a consistent system+user message pair."""
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

    def _timed_chain(
        self,
        prompt: str,
        llm,
        response_cls: type[BaseModel],
        extract_field: Callable[[BaseModel], object],
    ):
        """Build one field-level chain: prompt + invoke + timed result."""

        def build_messages(resume_text: str):
            return self._messages(f"{prompt}\n\n简历文本：\n{resume_text}")

        def run_with_time(messages):
            start = time.perf_counter()
            raw = invoke_with_fallback(llm, messages, response_cls)
            parsed = extract_field(raw)
            return parsed, time.perf_counter() - start

        return RunnableLambda(build_messages) | RunnableLambda(run_with_time)

    def extract_all(self, resume_text: str) -> tuple[ResumeData, float]:
        """Extract 5 field groups in parallel, then aggregate."""
        start = time.perf_counter()
        parallel_result = self.parallel.invoke(resume_text)

        basic_info, _t_basic = parallel_result["basic_info"]
        work_experience, _t_work = parallel_result["work_experience"]
        education, _t_edu = parallel_result["education"]
        projects, _t_proj = parallel_result["projects"]
        academic_achievements, _t_academic = parallel_result["academic_achievements"]

        # 兜底保护：确保前端依赖字段都存在，避免渲染阶段空字段崩溃。
        data = ResumeData(
            basicInfo=basic_info if basic_info else BasicInfo(),
            workExperience=work_experience or [],
            education=education or [],
            projects=projects or [],
            academicAchievements=academic_achievements or [],
        )
        elapsed = round(time.perf_counter() - start, 4)
        return data, elapsed

    def extract_to_dict(self, resume_text: str) -> tuple[dict, float]:
        """Extract resume and return dictionary payload."""
        data, elapsed = self.extract_all(resume_text)
        return data.model_dump(), elapsed
