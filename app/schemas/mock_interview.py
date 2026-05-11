"""Schemas for mock interview session and streaming APIs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.schemas.interview import Category, InterviewType
from app.schemas.jd import JDData
from app.schemas.resume import ResumeData

StreamMode = Literal["start", "reply"]
MessageRole = Literal["assistant", "user"]
MockInterviewCreateStage = Literal["retrieving_evidence", "generating_plan", "session_created"]

OPENING_ROUND_KEYWORDS = ("开场", "自我介绍", "背景", "介绍", "opening", "warm")
PROJECT_ROUND_KEYWORDS = ("项目", "项目概述", "经历", "经验", "overview", "experience")
CODING_ROUND_KEYWORDS = ("代码", "编码", "算法", "编程", "leetcode", "coding")


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """判断文本是否包含任一关键词（大小写不敏感）。

    设计说明：
    - 输入文本可能来自 LLM 生成结果，大小写、语言混用较常见，因此先统一小写；
    - 采用 `in` 子串判断而不是严格相等，容忍“主题 + 描述”混合文本；
    - 该函数是计划结构校验的基础工具，避免在 validator 中重复样板逻辑。
    """
    normalized = text.lower()
    return any(keyword.lower() in normalized for keyword in keywords)


class MockInterviewRound(BaseModel):
    """A single interview round with topic and description."""

    round: int = Field(ge=1, description="轮次编号")
    topic: str = Field(min_length=1, max_length=100, description="轮次主题")
    description: str = Field(min_length=1, max_length=300, description="轮次描述")


class MockInterviewPlan(BaseModel):
    """Canonical round/topic-driven interview plan."""

    plan: list[MockInterviewRound] = Field(min_length=3, max_length=12)
    total_rounds: int = Field(ge=3, le=12, description="总轮次数")
    estimated_duration: str = Field(min_length=1, max_length=40, description="预计时长")
    leetcode_problem: str = Field(min_length=1, max_length=200, description="代码题题目")

    @model_validator(mode="after")
    def validate_plan(self) -> "MockInterviewPlan":
        """对生成的面试计划做结构和业务约束双重校验。

        校验目标：
        1) 结构完整：`plan` 数量必须与 `total_rounds` 一致；
        2) 轮次连续：轮次编号必须从 1 开始连续递增，避免跳号/重复；
        3) 业务语义：首轮必须是开场，第二轮必须是项目梳理，末轮必须是编码题。

        说明：
        - 这是“后置校验”（mode="after"），会在字段级校验通过后执行；
        - 一旦不满足约束，抛出 ValueError，由 FastAPI/Pydantic 自动转为请求错误。
        """
        if len(self.plan) != self.total_rounds:
            raise ValueError("plan length must match total_rounds")

        expected_rounds = list(range(1, self.total_rounds + 1))
        actual_rounds = [item.round for item in self.plan]
        if actual_rounds != expected_rounds:
            raise ValueError("round numbers must be continuous from 1 to total_rounds")

        first_round_text = f"{self.plan[0].topic} {self.plan[0].description}"
        second_round_text = f"{self.plan[1].topic} {self.plan[1].description}"
        last_round_text = f"{self.plan[-1].topic} {self.plan[-1].description}"

        if not _contains_keyword(first_round_text, OPENING_ROUND_KEYWORDS):
            raise ValueError("first round must be an opening round")
        if not _contains_keyword(second_round_text, PROJECT_ROUND_KEYWORDS):
            raise ValueError("second round must be a project overview round")
        if not _contains_keyword(last_round_text, CODING_ROUND_KEYWORDS):
            raise ValueError("last round must be a coding round")

        return self


class MockInterviewRetrievalFilters(BaseModel):
    """Metadata filters applied to interview retrieval."""

    category: Category | None = None
    interviewType: InterviewType | None = None
    company: str | None = None


class MockInterviewRetrievalItem(BaseModel):
    """A single interview document used as planning evidence."""

    interviewId: int
    source: str = ""
    sourceId: str = ""
    title: str
    company: str | None = None
    category: Category
    interviewType: InterviewType | None = None
    stage: str | None = None
    publishTime: str = ""
    snippet: str = ""
    score: float = 0.0
    reason: str = ""


class MockInterviewRetrievalResult(BaseModel):
    """Interview evidence used during plan generation."""

    queryText: str = ""
    appliedFilters: MockInterviewRetrievalFilters = Field(default_factory=MockInterviewRetrievalFilters)
    items: list[MockInterviewRetrievalItem] = Field(default_factory=list)


class MockInterviewDeveloperContext(BaseModel):
    """Shared developer-facing context for local trace export."""

    sessionMode: Literal["frontend_local_only"] = "frontend_local_only"
    privacyMode: Literal["frontend_local_export_only"] = "frontend_local_export_only"
    ragEnabled: bool = False
    transcriptPersistence: Literal["frontend_local_only"] = "frontend_local_only"
    tracePersistence: Literal["frontend_local_only"] = "frontend_local_only"


class MockInterviewRetrievalTracePayload(BaseModel):
    """Developer trace payload for retrieval stage."""

    queryText: str = ""
    filterChain: list[MockInterviewRetrievalFilters] = Field(default_factory=list)
    appliedFilters: MockInterviewRetrievalFilters = Field(default_factory=MockInterviewRetrievalFilters)
    candidateTopk: int | None = None
    topk: int | None = None
    denseWeight: float | None = None
    sparseWeight: float | None = None
    ragEnabled: bool
    resultItems: list[MockInterviewRetrievalItem] = Field(default_factory=list)
    elapsedMs: int = Field(ge=0)


class MockInterviewPlanTracePayload(BaseModel):
    """Developer trace payload for plan generation stage."""

    promptKey: Literal["plan"] = "plan"
    jdDataIncluded: bool = False
    resumeProjectCount: int = Field(ge=0)
    retrievalItemCount: int = Field(ge=0)
    retrievalQueryText: str = ""
    outputPlan: MockInterviewPlan
    fallbackUsed: bool = False
    elapsedMs: int = Field(ge=0)


class ReflectionResult(BaseModel):
    """Reflection evaluation result after each candidate answer."""

    depth_score: int = Field(ge=1, le=5, description="深度分数 (1-5)")
    authenticity_score: int = Field(ge=1, le=5, description="真实性分数 (1-5)")
    completeness_score: int = Field(ge=1, le=5, description="完整性分数 (1-5)")
    logic_score: int = Field(ge=1, le=5, description="逻辑性分数 (1-5)")
    overall_assessment: str = Field(min_length=10, max_length=300, description="整体评价")
    should_continue: bool = Field(description="是否继续当前轮次")
    suggested_follow_up: str = Field(default="", max_length=200, description="建议的追问方向")
    reason: str = Field(min_length=10, max_length=200, description="决策理由")

    @model_validator(mode="after")
    def validate_follow_up(self) -> "ReflectionResult":
        """校验反思结果中的追问建议字段。

        业务规则：
        - 当 `should_continue=True` 时，表示应继续当前轮次深入提问；
        - 此时必须提供 `suggested_follow_up`，否则面试官下一问缺少方向依据。

        这样可以确保“继续追问”的决策总是附带可执行的追问意图。
        """
        if self.should_continue and not self.suggested_follow_up.strip():
            raise ValueError("suggested_follow_up is required when should_continue is true")
        return self


class MockInterviewReflectionTracePayload(BaseModel):
    """Developer trace payload for reflection stage."""

    promptKey: Literal["reflection"] = "reflection"
    candidateAnswer: str = Field(min_length=1)
    currentRoundHistory: str = ""
    questionCount: int = Field(ge=0)
    output: ReflectionResult
    fallbackUsed: bool = False
    elapsedMs: int = Field(ge=0)


class MockInterviewInterviewerTracePayload(BaseModel):
    """Developer trace payload for interviewer generation stage."""

    promptKey: Literal["interviewer"] = "interviewer"
    round: int = Field(ge=1)
    topic: str = ""
    suggestedFollowUp: str = ""
    closeInterview: bool = False
    recentConversation: list[dict[str, Any]] = Field(default_factory=list)
    finalMessage: str = ""
    elapsedMs: int = Field(ge=0)


class MockInterviewDeveloperTraceEvent(BaseModel):
    """Structured developer trace event streamed to the frontend."""

    type: Literal["retrieval", "plan_generation", "reflection", "interviewer_generation"]
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    payload: (
        MockInterviewRetrievalTracePayload
        | MockInterviewPlanTracePayload
        | MockInterviewReflectionTracePayload
        | MockInterviewInterviewerTracePayload
    )


class MockInterviewSessionLimits(BaseModel):
    """Runtime limits exposed to frontend."""

    durationMinutes: int = 60
    softInputChars: int = 1200
    maxInputChars: int = 1500
    contextWindowMessages: int = 8
    sessionTtlMinutes: int = 90


class MockInterviewMessage(BaseModel):
    """Stored transcript message."""

    id: str
    role: MessageRole
    content: str
    createdAt: datetime = Field(default_factory=datetime.utcnow)


class MockInterviewState(BaseModel):
    """Canonical runtime state for mock interview streaming."""

    currentRound: int = Field(default=1, ge=1)
    questionsPerRound: dict = Field(default_factory=dict)
    assistantQuestionCount: int = Field(default=0, ge=0)
    turnCount: int = Field(default=0, ge=0)
    reflectionHistory: list[ReflectionResult] = Field(default_factory=list)
    closed: bool = False

    @model_validator(mode="after")
    def normalize_questions_per_round(self) -> "MockInterviewState":
        """标准化并校验 `questionsPerRound` 的键值结构。

        背景：
        - 前端快照中的 `questionsPerRound` 可能出现键为 int/str 混用；
        - 该状态会在每次 stream 请求中回传，必须在后端统一成稳定格式。

        处理步骤：
        1) 将所有 key 归一化为字符串形式的正整数（如 "1", "2"）；
        2) 校验 value 必须为非负数（问题计数不能为负）；
        3) 若当前轮次不存在计数，自动补 0，保证下游读取安全。
        """
        normalized: dict[str, int] = {}
        for key, value in self.questionsPerRound.items():
            round_number = int(str(key))
            if round_number < 1:
                raise ValueError("questionsPerRound keys must be positive integers")
            if value < 0:
                raise ValueError("questionsPerRound values must be non-negative")
            normalized[str(round_number)] = value

        if str(self.currentRound) not in normalized:
            normalized[str(self.currentRound)] = 0

        self.questionsPerRound = normalized
        return self


class MockInterviewSessionCreateRequest(BaseModel):
    """Create a new mock interview session."""

    interviewType: InterviewType
    category: Category
    jdText: str = ""
    jdData: JDData | None = None
    resumeData: ResumeData

    @model_validator(mode="after")
    def validate_required_jd(self) -> "MockInterviewSessionCreateRequest":
        """创建会话时校验 JD 输入完整性。

        约束理由：
        - `jdText` 是检索查询与计划生成的重要语义来源，不能为空；
        - `jdData` 是结构化 JD 信息，后续过滤与提示词拼装都会依赖；
        - learning 版在创建会话阶段将两者都视为必填，以减少流程分叉复杂度。
        """
        if not self.jdText.strip():
            raise ValueError("jdText is required for mock interview sessions")
        if self.jdData is None:
            raise ValueError("jdData is required for mock interview sessions")
        return self


class MockInterviewSessionCreateResponse(BaseModel):
    """Metadata returned after a session is created."""

    sessionId: str
    interviewType: InterviewType
    category: Category
    status: Literal["ready"] = "ready"
    limits: MockInterviewSessionLimits
    interviewPlan: MockInterviewPlan
    interviewState: MockInterviewState
    jdData: JDData | None = None
    retrieval: MockInterviewRetrievalResult = Field(default_factory=MockInterviewRetrievalResult)
    resumeFingerprint: str
    expiresAt: datetime
    developerContext: MockInterviewDeveloperContext | None = None


class MockInterviewCreateProgressEvent(BaseModel):
    """Progress payload for session creation stream."""

    stage: MockInterviewCreateStage
    message: str = Field(min_length=1, max_length=60)


class MockInterviewAnswerAnalysisStartedEvent(BaseModel):
    """Reply-phase status payload emitted before reflection begins."""

    stage: Literal["analyzing_answer"] = "analyzing_answer"
    message: str = Field(default="正在分析你的回答", min_length=1, max_length=60)


class MockInterviewStreamRequest(BaseModel):
    """Request body for streaming the next interviewer turn."""

    mode: StreamMode
    message: Optional[str] = None
    interviewType: InterviewType
    category: Category
    jdText: str = ""
    jdData: JDData | None = None
    resumeSnapshot: ResumeData
    retrieval: MockInterviewRetrievalResult = Field(default_factory=MockInterviewRetrievalResult)
    interviewPlan: MockInterviewPlan
    interviewState: MockInterviewState
    messages: list[MockInterviewMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_request(self) -> "MockInterviewStreamRequest":
        """校验续流请求（start/reply）的关键前置条件。

        校验点：
        1) 当 mode=reply 时，message 必须非空，避免“空回答”触发反思链路；
        2) 当前轮次 `currentRound` 不能超过计划总轮次，防止状态漂移越界。

        该校验确保前端回传快照在逻辑上可继续推进，不会破坏会话状态机。
        """
        if self.mode == "reply" and not (self.message or "").strip():
            raise ValueError("message is required when mode is 'reply'")
        if self.interviewState.currentRound > self.interviewPlan.total_rounds:
            raise ValueError("interviewState.currentRound cannot exceed interviewPlan.total_rounds")
        return self
