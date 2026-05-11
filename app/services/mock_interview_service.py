"""模拟面试核心服务（learning 版）。

设计目标：
1. 对齐主项目的接口形状（方便前端联调），但保持实现足够轻量。
2. 会话不在后端长期持久化，采用“前端回传快照 -> 后端重建临时态”的方式。
3. 采用 SSE 流式事件输出，细分创建阶段和对话阶段，便于前端逐步渲染。

总体流程：
- 创建会话：检索面经证据 -> 生成面试计划 -> 返回 session_created。
- 对话续流：根据 mode/start|reply 重建并推进轮次 -> 生成面试官下一问。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

from fastapi import HTTPException
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from app.core.config import get_settings
from app.prompts.mock_interview_prompts import get_mock_interview_prompts
from app.schemas.interview import Category, InterviewData, InterviewType
from app.schemas.jd import JDData
from app.schemas.mock_interview import (
    MockInterviewAnswerAnalysisStartedEvent,
    MockInterviewCreateProgressEvent,
    MockInterviewDeveloperContext,
    MockInterviewDeveloperTraceEvent,
    MockInterviewInterviewerTracePayload,
    MockInterviewMessage,
    MockInterviewPlan,
    MockInterviewPlanTracePayload,
    MockInterviewReflectionTracePayload,
    MockInterviewRetrievalFilters,
    MockInterviewRetrievalItem,
    MockInterviewRetrievalResult,
    MockInterviewRetrievalTracePayload,
    MockInterviewSessionCreateRequest,
    MockInterviewSessionCreateResponse,
    MockInterviewSessionLimits,
    MockInterviewState,
    MockInterviewStreamRequest,
    ReflectionResult,
)
from app.schemas.resume import ResumeData
from app.services.interview_service import get_data_service
from app.services.mock_interview_rag_service import (
    MockInterviewChromaRagService,
    RetrievalQueryContext,
)
from app.utils.structured_output import invoke_with_fallback

# 控制发送给模型的 JD 最大长度，避免提示词过大影响时延和成本。
JD_CHAR_LIMIT = 5000
# 非开场轮次每个 topic 最多提问数。
MAX_QUESTIONS_PER_TOPIC = 5
# 开场轮仅允许 1 问，确保流程快速进入实质面试。
OPENING_ROUND_MAX_QUESTIONS = 1

@dataclass(slots=True)
class MockInterviewSession:
    """服务内部使用的临时会话对象（ephemeral）。

    注意：learning 版不依赖服务端持久化存储；每次 stream 请求都可重建该对象。
    """

    session_id: str
    created_at: datetime
    last_active_at: datetime
    expires_at: datetime
    resume_fingerprint: str
    interview_type: InterviewType
    category: Category
    jd_text: str
    jd_data: JDData | None
    resume_snapshot: ResumeData
    retrieval_result: MockInterviewRetrievalResult
    interview_plan: MockInterviewPlan
    interview_state: MockInterviewState
    messages: list[MockInterviewMessage]


class MockInterviewService:
    """模拟面试服务：负责会话创建、轮次推进、反思分析与流式生成。"""

    def __init__(
        self,
        model_provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        rate_limiter: InMemoryRateLimiter | None = None,
    ):
        """初始化模型、提示词与运行时限制。

        关键点：
        - 未显式传参时，回退到环境配置中的 active model。
        - 使用内存限流器，避免模型请求突发过高。
        """
        settings = get_settings()
        active_config = settings.get_active_config()
        model_provider = model_provider or active_config["model_provider"]
        api_key = api_key or active_config["api_key"]
        base_url = base_url or active_config["base_url"]
        model = model or active_config["model"]

        if rate_limiter is None:
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=settings.rate_limit_requests_per_second,
                check_every_n_seconds=settings.rate_limit_check_every_n_seconds,
                max_bucket_size=settings.rate_limit_max_bucket_size,
            )

        self.chat_model = self._create_chat_model(
            model_provider=model_provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter,
        )
        self.plan_llm = self.chat_model.with_structured_output(MockInterviewPlan)
        self.reflection_llm = self.chat_model.with_structured_output(ReflectionResult)
        self.prompts = get_mock_interview_prompts()
        self._session_ttl_minutes = max(settings.mock_interview_session_ttl_minutes, 10)
        self._plan_generation_timeout_seconds = max(settings.mock_interview_plan_timeout_seconds, 5)
        self._rag_topk = max(settings.mock_interview_rag_topk, 1)
        self._rag_candidate_topk = max(settings.mock_interview_rag_candidate_topk, self._rag_topk)
        self._rag_enabled = bool(settings.mock_interview_rag and model_provider == "openai" and api_key)
        self._last_retrieval_used_rag = False
        self._rag_service = (
            MockInterviewChromaRagService(
                settings=settings,
                api_key=api_key,
                base_url=base_url,
                embedding_model_name=settings.mock_interview_embedding_model,
                topk=self._rag_topk,
                candidate_topk=self._rag_candidate_topk,
            )
            if self._rag_enabled
            else None
        )
        self._limits = MockInterviewSessionLimits(sessionTtlMinutes=self._session_ttl_minutes)

    def _create_chat_model(
        self,
        model_provider: str,
        model: str,
        api_key: str | None,
        base_url: str | None,
        rate_limiter: InMemoryRateLimiter,
    ):
        """按 provider 构建 LangChain chat model。"""
        if model_provider == "openai":
            kwargs = {
                "model": model,
                "model_provider": "openai",
                "api_key": api_key,
                "rate_limiter": rate_limiter,
            }
            if base_url:
                kwargs["base_url"] = base_url
            return init_chat_model(**kwargs)
        if model_provider == "google_genai":
            return init_chat_model(f"google_genai:{model}", rate_limiter=rate_limiter)
        if model_provider == "anthropic":
            return init_chat_model(model, model_provider="anthropic", rate_limiter=rate_limiter)
        raise ValueError(f"Unsupported model provider: {model_provider}")

    def _utcnow(self) -> datetime:
        """统一使用 UTC 时间，避免时区导致的 expires 计算偏差。"""
        return datetime.now(timezone.utc)

    def _trim_jd(self, jd_text: str) -> str:
        """清洗并截断 JD，控制上下文长度。"""
        return jd_text.strip()[:JD_CHAR_LIMIT]

    def _build_resume_fingerprint(self, resume_data: ResumeData) -> str:
        """生成简历指纹，便于前后端识别当前快照是否匹配。"""
        serialized = json.dumps(resume_data.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]

    def _new_expires_at(self, now: datetime) -> datetime:
        """基于当前时间计算新的会话过期时间。"""
        return now + timedelta(minutes=self._session_ttl_minutes)

    def _serialize_json(self, value: Any) -> str:
        """将复杂对象序列化为 JSON 文本，供提示词拼装使用。"""
        return json.dumps(value, ensure_ascii=False, indent=2)

    def _build_developer_context(self) -> MockInterviewDeveloperContext:
        """构造开发调试上下文。

        注意：这里返回“本次请求最终是否走了 RAG”，而不是仅看 env 配置开关。
        """
        return MockInterviewDeveloperContext(ragEnabled=self._last_retrieval_used_rag)

    def _build_trace_event(self, trace_type: str, payload: Any) -> dict[str, Any]:
        """统一包装 developer_trace 事件。"""
        return {
            "event": "developer_trace",
            "data": MockInterviewDeveloperTraceEvent(type=trace_type, payload=payload).model_dump(mode="json"),
        }

    @staticmethod
    def _build_retrieval_query_text(context: RetrievalQueryContext) -> str:
        """优先使用 JD 片段作为检索查询文本；缺失时退化为类别+类型。"""
        query_text = (context.jd_text or "").strip()[:400]
        if query_text:
            return query_text
        return f"{context.category.value}\n{context.interview_type.value}"

    @staticmethod
    def _normalize_company(company: str | None) -> str:
        """归一化公司名称，提升 company 过滤命中率。"""
        if not company:
            return ""
        normalized = re.sub(r"[（(].*?[)）]", "", company)
        normalized = re.sub(r"\s+", "", normalized)
        return normalized.strip()

    @staticmethod
    def _build_snippet(content: str) -> str:
        """构造展示用片段：压缩空白并截断。"""
        compact = " ".join((content or "").split())
        return compact[:180]

    def _build_retrieval_reason(self, interview: InterviewData, snippet: str) -> str:
        """拼接检索命中理由，帮助前端展示“为什么选这条面经”。"""
        reasons: list[str] = []
        if interview.company:
            reasons.append(f"公司：{interview.company}")
        if interview.interview_type:
            reasons.append(f"类型：{interview.interview_type.value}")
        if interview.stage:
            reasons.append(f"阶段：{interview.stage}")
        if snippet:
            reasons.append(f"片段：{snippet[:60]}")
        return "；".join(reasons)

    def _to_retrieval_item(self, interview: InterviewData) -> MockInterviewRetrievalItem:
        """把数据层 InterviewData 转换成 mock interview 检索项。"""
        snippet = self._build_snippet(interview.content)
        return MockInterviewRetrievalItem(
            interviewId=interview.id,
            source=interview.source,
            sourceId=interview.source_id,
            title=interview.title,
            company=interview.company,
            category=interview.category,
            interviewType=interview.interview_type,
            stage=interview.stage,
            publishTime=interview.publish_time,
            snippet=snippet,
            score=0.0,
            reason=self._build_retrieval_reason(interview, snippet),
        )

    def _retrieve_interview_evidence_non_rag(
        self, context: RetrievalQueryContext
    ) -> tuple[MockInterviewRetrievalResult, list[MockInterviewRetrievalFilters]]:
        """非 RAG 检索策略（learning 默认路径）。

        分层策略：
        1) category + interview_type + company
        2) category + interview_type
        3) category

        目标是在缺少向量索引时，仍能尽可能拿到相关证据。
        """
        topk = 5
        remaining = topk
        seen_ids: set[int] = set()
        selected: list[InterviewData] = []

        company = self._normalize_company(context.jd_data.basicInfo.company if context.jd_data else "")
        filter_chain: list[MockInterviewRetrievalFilters] = []
        applied_filters: MockInterviewRetrievalFilters | None = None

        tiers = [
            {"category": context.category, "interview_type": context.interview_type, "company": company or None},
            {"category": context.category, "interview_type": context.interview_type, "company": None},
            {"category": context.category, "interview_type": None, "company": None},
        ]

        data_service = get_data_service()
        # 逐层放宽过滤条件，直到拿满 topk 或 tiers 用尽。
        for tier in tiers:
            if remaining <= 0:
                break
            interviews, _ = data_service.filter_interviews(
                categories=[tier["category"]],
                interview_types=[tier["interview_type"].value] if tier["interview_type"] else None,
                company=tier["company"],
                page=1,
                page_size=remaining,
            )

            added_count = 0
            for interview in interviews:
                if interview.id in seen_ids:
                    continue
                seen_ids.add(interview.id)
                selected.append(interview)
                added_count += 1
                remaining -= 1
                if remaining <= 0:
                    break

            tier_filter = MockInterviewRetrievalFilters(
                category=tier["category"],
                interviewType=tier["interview_type"],
                company=tier["company"],
            )
            filter_chain.append(tier_filter)
            if added_count > 0:
                applied_filters = tier_filter

        result = MockInterviewRetrievalResult(
            queryText=self._build_retrieval_query_text(context),
            appliedFilters=applied_filters
            or MockInterviewRetrievalFilters(
                category=context.category,
                interviewType=context.interview_type,
                company=company or None,
            ),
            items=[self._to_retrieval_item(interview) for interview in selected[:topk]],
        )
        return result, filter_chain

    def _build_plan_messages(
        self,
        request: MockInterviewSessionCreateRequest,
        jd_data: JDData | None,
        retrieval_result: MockInterviewRetrievalResult,
    ) -> list[SystemMessage]:
        """构造“生成面试计划”阶段的系统提示词。"""
        prompt = self.prompts["plan"].format(
            domain=request.category.value,
            jd_info=self._serialize_json(jd_data.model_dump(mode="json") if jd_data else {}),
            resume_info=self._serialize_json(request.resumeData.model_dump(mode="json")),
            retrieved_interviews=self._serialize_json(retrieval_result.model_dump(mode="json")),
        )
        return [SystemMessage(content=prompt)]

    def _generate_plan(
        self,
        request: MockInterviewSessionCreateRequest,
        jd_data: JDData | None,
        retrieval_result: MockInterviewRetrievalResult,
    ) -> MockInterviewPlan:
        """调用结构化输出模型生成 MockInterviewPlan。"""
        messages = self._build_plan_messages(request, jd_data, retrieval_result)
        return invoke_with_fallback(self.plan_llm, messages, MockInterviewPlan)

    def _build_retrieval_context(self, request: MockInterviewSessionCreateRequest) -> RetrievalQueryContext:
        """从创建请求中提取检索所需上下文。"""
        return RetrievalQueryContext(
            category=request.category,
            interview_type=request.interviewType,
            resume_data=request.resumeData,
            jd_text=self._trim_jd(request.jdText),
            jd_data=request.jdData,
        )

    async def _run_retrieval(
        self, request: MockInterviewSessionCreateRequest
    ) -> tuple[MockInterviewRetrievalResult, dict[str, Any]]:
        """执行检索并返回 trace 元信息。

        检索属于同步 I/O + 计算逻辑，放到线程池避免阻塞事件循环。
        """
        loop = asyncio.get_running_loop()
        context = self._build_retrieval_context(request)
        start = time.perf_counter()
        rag_trace_extras = {
            "candidateTopk": None,
            "topk": None,
            "denseWeight": None,
            "sparseWeight": None,
        }
        rag_used = False
        if self._rag_enabled and self._rag_service is not None:
            try:
                result, filter_chain, rag_trace_extras = await loop.run_in_executor(
                    None, lambda: self._rag_service.retrieve(context)
                )
                rag_used = True
            except Exception:
                # learning 版策略：RAG 异常自动回退 non-RAG，确保链路可用。
                result, filter_chain = await loop.run_in_executor(
                    None, lambda: self._retrieve_interview_evidence_non_rag(context)
                )
        else:
            result, filter_chain = await loop.run_in_executor(
                None, lambda: self._retrieve_interview_evidence_non_rag(context)
            )

        self._last_retrieval_used_rag = rag_used
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        trace_meta = {
            "queryText": result.queryText,
            "filterChain": [item.model_dump(mode="json") for item in filter_chain],
            "appliedFilters": result.appliedFilters.model_dump(mode="json"),
            "candidateTopk": rag_trace_extras["candidateTopk"],
            "topk": rag_trace_extras["topk"],
            "denseWeight": rag_trace_extras["denseWeight"],
            "sparseWeight": rag_trace_extras["sparseWeight"],
            "resultItems": [item.model_dump(mode="json") for item in result.items],
            "elapsedMs": elapsed_ms,
            "ragEnabled": rag_used,
        }
        return result, trace_meta

    async def _run_plan_generation(
        self,
        request: MockInterviewSessionCreateRequest,
        retrieval_result: MockInterviewRetrievalResult,
    ) -> tuple[MockInterviewPlan, dict[str, Any]]:
        """执行计划生成并附带超时控制与 trace 信息。"""
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        timeout_seconds = self._plan_generation_timeout_seconds
        try:
            plan = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._generate_plan(request, request.jdData, retrieval_result)),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=f"生成面试计划超时（{timeout_seconds}s），请检查模型服务或稍后重试。",
            ) from exc

        elapsed_ms = round((time.perf_counter() - start) * 1000)
        trace_meta = {
            "jdDataIncluded": request.jdData is not None,
            "resumeProjectCount": len(request.resumeData.projects),
            "retrievalItemCount": len(retrieval_result.items),
            "retrievalQueryText": retrieval_result.queryText,
            "outputPlan": plan.model_dump(mode="json"),
            "fallbackUsed": False,
            "elapsedMs": elapsed_ms,
        }
        return plan, trace_meta

    def _build_initial_state(self) -> MockInterviewState:
        """创建初始面试状态。"""
        return MockInterviewState(
            currentRound=1,
            questionsPerRound={"1": 0},
            assistantQuestionCount=0,
            turnCount=0,
            reflectionHistory=[],
            closed=False,
        )

    def _build_session_create_response(
        self,
        *,
        session_id: str,
        interview_type: InterviewType,
        category: Category,
        jd_data: JDData | None,
        retrieval_result: MockInterviewRetrievalResult,
        plan: MockInterviewPlan,
        interview_state: MockInterviewState,
        resume_fingerprint: str,
        expires_at: datetime,
    ) -> MockInterviewSessionCreateResponse:
        """组装创建会话响应（session_created 事件主体）。"""
        return MockInterviewSessionCreateResponse(
            sessionId=session_id,
            interviewType=interview_type,
            category=category,
            limits=self._limits,
            interviewPlan=plan,
            interviewState=interview_state,
            jdData=jd_data,
            retrieval=retrieval_result,
            resumeFingerprint=resume_fingerprint,
            expiresAt=expires_at,
            developerContext=self._build_developer_context(),
        )

    async def stream_create_session(self, request: MockInterviewSessionCreateRequest) -> AsyncIterator[dict[str, Any]]:
        """创建会话的 SSE 事件流。

        事件顺序：
        progress(retrieval) -> developer_trace(retrieval) ->
        progress(plan) -> developer_trace(plan) ->
        session_created -> done
        """
        yield {
            "event": "progress",
            "data": MockInterviewCreateProgressEvent(stage="retrieving_evidence", message="正在检索相关面经").model_dump(
                mode="json"
            ),
        }
        retrieval_result, retrieval_trace = await self._run_retrieval(request)
        yield self._build_trace_event("retrieval", MockInterviewRetrievalTracePayload(**retrieval_trace))

        yield {
            "event": "progress",
            "data": MockInterviewCreateProgressEvent(stage="generating_plan", message="正在生成面试计划").model_dump(
                mode="json"
            ),
        }
        plan, plan_trace = await self._run_plan_generation(request, retrieval_result)
        yield self._build_trace_event("plan_generation", MockInterviewPlanTracePayload(**plan_trace))

        now = self._utcnow()
        session_id = str(uuid4())
        response = self._build_session_create_response(
            session_id=session_id,
            interview_type=request.interviewType,
            category=request.category,
            jd_data=request.jdData,
            retrieval_result=retrieval_result,
            plan=plan,
            interview_state=self._build_initial_state(),
            resume_fingerprint=self._build_resume_fingerprint(request.resumeData),
            expires_at=self._new_expires_at(now),
        )
        session_payload = response.model_dump(mode="json")
        session_payload["developerContext"] = self._build_developer_context().model_dump(mode="json")
        yield {"event": "session_created", "data": session_payload}
        yield {"event": "done", "data": {"sessionId": session_id, "status": "ready"}}

    def _ensure_stream_allowed(self, session: MockInterviewSession, request: MockInterviewStreamRequest) -> None:
        """校验当前请求模式是否合法，避免状态机错位。"""
        if session.interview_state.closed:
            raise HTTPException(status_code=409, detail="Mock interview session already completed")
        if request.mode == "start" and session.interview_state.assistantQuestionCount > 0:
            raise HTTPException(status_code=409, detail="Interview session already started")
        if request.mode == "reply" and session.interview_state.assistantQuestionCount == 0:
            raise HTTPException(status_code=409, detail="Interview session has not started yet")

    def _build_ephemeral_session(self, session_id: str, request: MockInterviewStreamRequest) -> MockInterviewSession:
        """根据前端传入快照重建临时会话态。

        这是 learning 版最关键的策略：服务端无状态（或弱状态）也能继续对话。
        """
        now = self._utcnow()
        return MockInterviewSession(
            session_id=session_id,
            created_at=now,
            last_active_at=now,
            expires_at=self._new_expires_at(now),
            resume_fingerprint="frontend-only",
            interview_type=request.interviewType,
            category=request.category,
            jd_text=self._trim_jd(request.jdText),
            jd_data=request.jdData,
            resume_snapshot=request.resumeSnapshot,
            retrieval_result=request.retrieval,
            interview_plan=request.interviewPlan,
            interview_state=request.interviewState.model_copy(deep=True),
            messages=list(request.messages),
        )

    def _get_current_round_key(self, session: MockInterviewSession) -> str:
        """将 currentRound 统一转为字典键格式。"""
        return str(session.interview_state.currentRound)

    def _ensure_current_round_bucket(self, session: MockInterviewSession) -> None:
        """确保 questionsPerRound 中存在当前轮次键。"""
        key = self._get_current_round_key(session)
        if key not in session.interview_state.questionsPerRound:
            session.interview_state.questionsPerRound[key] = 0

    def _get_current_round(self, session: MockInterviewSession):
        """读取当前轮次的计划配置（topic/description）。"""
        return session.interview_plan.plan[session.interview_state.currentRound - 1]

    def _is_last_round(self, session: MockInterviewSession) -> bool:
        """判断是否已经到达最后一轮。"""
        return session.interview_state.currentRound >= session.interview_plan.total_rounds

    def _is_coding_round(self, session: MockInterviewSession) -> bool:
        """约定：最后一轮即 coding round。"""
        return self._is_last_round(session)

    def _get_current_round_question_count(self, session: MockInterviewSession) -> int:
        """读取当前轮次已经提问数量。"""
        return session.interview_state.questionsPerRound.get(self._get_current_round_key(session), 0)

    def _get_round_question_limit(self, session: MockInterviewSession) -> int:
        """按轮次返回提问上限（开场轮更严格）。"""
        if session.interview_state.currentRound == 1:
            return OPENING_ROUND_MAX_QUESTIONS
        return MAX_QUESTIONS_PER_TOPIC

    def _normalize_reflection_for_round_limit(
        self,
        session: MockInterviewSession,
        reflection_result: ReflectionResult,
        current_round_question_count: int,
    ) -> ReflectionResult:
        """将 reflection 结果与硬性轮次上限对齐。

        即便模型建议继续，若已达到上限，也强制结束当前轮。
        """
        question_limit = self._get_round_question_limit(session)
        if current_round_question_count < question_limit:
            return reflection_result
        if self._is_last_round(session):
            reason = f"当前 topic 已达到 {question_limit} 问上限，结束面试。"
        elif session.interview_state.currentRound == 1:
            reason = "开场轮已完成自我介绍热身，进入下一轮。"
        else:
            reason = f"当前 topic 已达到 {question_limit} 问上限，进入下一轮。"
        return reflection_result.model_copy(
            update={"should_continue": False, "suggested_follow_up": "", "reason": reason}
        )

    def _advance_to_next_round(self, session: MockInterviewSession) -> tuple[int, int]:
        """推进到下一轮并返回 (from_round, to_round)。"""
        from_round = session.interview_state.currentRound
        session.interview_state.currentRound += 1
        self._ensure_current_round_bucket(session)
        return from_round, session.interview_state.currentRound

    def _append_reflection(self, session: MockInterviewSession, reflection: ReflectionResult) -> None:
        """记录本次回答对应的反思结果。"""
        session.interview_state.reflectionHistory.append(reflection)

    def _record_assistant_question(self, session: MockInterviewSession) -> None:
        """记录面试官提问计数（按轮次 + 全局）。"""
        self._ensure_current_round_bucket(session)
        key = self._get_current_round_key(session)
        question_limit = self._get_round_question_limit(session)
        next_count = min(session.interview_state.questionsPerRound[key] + 1, question_limit)
        session.interview_state.questionsPerRound[key] = next_count
        session.interview_state.assistantQuestionCount += 1

    def _get_current_round_messages(self, session: MockInterviewSession) -> str:
        """提取“当前轮次”相关历史对话文本，供 reflection 使用。

        思路：从尾部反向定位本轮第一条 assistant 消息，再截取到末尾。
        """
        current_round_count = session.interview_state.questionsPerRound.get(self._get_current_round_key(session), 0)
        if current_round_count <= 0 or not session.messages:
            return ""
        assistant_seen = 0
        start_index = 0
        for index in range(len(session.messages) - 1, -1, -1):
            if session.messages[index].role == "assistant":
                assistant_seen += 1
                if assistant_seen == current_round_count:
                    start_index = index
                    break
        round_messages = session.messages[start_index:]
        formatted: list[str] = []
        for message in round_messages:
            role = "面试官" if message.role == "assistant" else "候选人"
            formatted.append(f"{role}: {message.content}")
        return "\n\n".join(formatted)

    def _build_reflection_messages(self, session: MockInterviewSession, candidate_answer: str) -> list[SystemMessage]:
        """构建 reflection 提示词消息。"""
        current_round = self._get_current_round(session)
        prompt = self.prompts["reflection"].format(
            current_round=session.interview_state.currentRound,
            total_rounds=session.interview_plan.total_rounds,
            round_topic=current_round.topic,
            current_description=current_round.description,
            candidate_last_answer=candidate_answer,
            current_round_history=self._get_current_round_messages(session),
            current_round_question_count=self._get_current_round_question_count(session),
        )
        return [SystemMessage(content=prompt)]

    def _fallback_reflection(self) -> ReflectionResult:
        """反思失败时的保底结果，避免面试中断。"""
        return ReflectionResult(
            depth_score=3,
            authenticity_score=3,
            completeness_score=3,
            logic_score=3,
            overall_assessment="系统反思暂时不可用，建议补充更具体的项目案例和技术细节。",
            should_continue=True,
            suggested_follow_up="请结合一个具体项目案例继续展开，重点说明决策依据与结果。",
            reason="反思调用失败，默认继续当前轮次以避免流程中断。",
        )

    async def _call_reflection(
        self, session: MockInterviewSession, candidate_answer: str
    ) -> tuple[ReflectionResult, bool, int, str]:
        """调用 reflection 模型，并返回结果 + fallback标记 + 耗时 + 当前轮历史。"""
        loop = asyncio.get_running_loop()
        messages = self._build_reflection_messages(session, candidate_answer)
        history = self._get_current_round_messages(session)
        started_at = time.perf_counter()
        try:
            result = await loop.run_in_executor(None, lambda: invoke_with_fallback(self.reflection_llm, messages, ReflectionResult))
            return result, False, round((time.perf_counter() - started_at) * 1000), history
        except Exception:
            return self._fallback_reflection(), True, round((time.perf_counter() - started_at) * 1000), history

    def _recent_conversation(self, session: MockInterviewSession) -> list[dict[str, Any]]:
        """截取最近 N 条对话，控制提示词长度。"""
        recent_messages = session.messages[-self._limits.contextWindowMessages :]
        return [message.model_dump(mode="json") for message in recent_messages]

    def _build_interviewer_messages(
        self,
        session: MockInterviewSession,
        close_interview: bool,
        reflection_result: ReflectionResult | None = None,
    ) -> list[SystemMessage | HumanMessage]:
        """构建“面试官发问”提示词消息。"""
        current_round = self._get_current_round(session)
        suggested_follow_up = reflection_result.suggested_follow_up if reflection_result else ""
        is_coding_round = self._is_coding_round(session)
        leetcode_problem = session.interview_plan.leetcode_problem if is_coding_round else ""
        prompt = self.prompts["interviewer"].format(
            domain=session.category.value,
            current_round=session.interview_state.currentRound,
            total_rounds=session.interview_plan.total_rounds,
            current_topic=current_round.topic,
            current_description=current_round.description,
            interview_plan=self._serialize_json(session.interview_plan.model_dump(mode="json")),
            jd_info=self._serialize_json(session.jd_data.model_dump(mode="json") if session.jd_data else {}),
            resume_info=self._serialize_json(session.resume_snapshot.model_dump(mode="json")),
            retrieved_interviews=self._serialize_json(session.retrieval_result.model_dump(mode="json")),
            conversation_history=self._serialize_json(self._recent_conversation(session)),
            suggested_follow_up=suggested_follow_up,
            leetcode_problem=leetcode_problem,
        )
        if close_interview:
            # 用显式控制 token 指示模型输出收尾话术，而非继续追问。
            prompt += "\n\nCLOSE_INTERVIEW"
        payload = {"mode": "closing" if close_interview else "question", "isCodingRound": is_coding_round}
        return [SystemMessage(content=prompt), HumanMessage(content=f"input_json:\n{self._serialize_json(payload)}")]

    def _touch_session(self, session: MockInterviewSession) -> None:
        """更新会话活跃时间与过期时间。"""
        now = self._utcnow()
        session.last_active_at = now
        session.expires_at = self._new_expires_at(now)

    async def stream_turn(
        self,
        session_id: str,
        request: MockInterviewStreamRequest,
        recovery_token: str = "",
    ) -> AsyncIterator[dict[str, Any]]:
        """对话阶段 SSE 主流程（start/reply 共用入口）。"""
        del recovery_token
        session = self._build_ephemeral_session(session_id, request)
        self._ensure_stream_allowed(session, request)
        self._ensure_current_round_bucket(session)

        reflection_result: ReflectionResult | None = None
        close_interview = False

        if request.mode == "reply":
            # reply 模式：先处理候选人回答，再决定是否转轮，最后生成面试官输出。
            message = (request.message or "").strip()
            if len(message) > self._limits.maxInputChars:
                raise HTTPException(status_code=400, detail="回答长度不能超过 1500 字")

            user_message = MockInterviewMessage(id=f"user-{uuid4()}", role="user", content=message)
            session.messages.append(user_message)
            session.interview_state.turnCount += 1
            self._touch_session(session)
            yield {"event": "user_message", "data": user_message.model_dump(mode="json")}
            yield {"event": "answer_analysis_started", "data": MockInterviewAnswerAnalysisStartedEvent().model_dump(mode="json")}

            reflection_result, reflection_fallback_used, reflection_elapsed_ms, round_history = await self._call_reflection(
                session, message
            )
            current_round_question_count = self._get_current_round_question_count(session)
            reflection_result = self._normalize_reflection_for_round_limit(
                session, reflection_result, current_round_question_count
            )
            self._append_reflection(session, reflection_result)
            yield {"event": "reflection_result", "data": reflection_result.model_dump(mode="json")}
            yield self._build_trace_event(
                "reflection",
                MockInterviewReflectionTracePayload(
                    candidateAnswer=message,
                    currentRoundHistory=round_history,
                    questionCount=current_round_question_count,
                    output=reflection_result,
                    fallbackUsed=reflection_fallback_used,
                    elapsedMs=reflection_elapsed_ms,
                ),
            )

            if reflection_result.should_continue:
                close_interview = False
            elif self._is_last_round(session):
                session.interview_state.closed = True
                close_interview = True
            else:
                from_round, to_round = self._advance_to_next_round(session)
                yield {
                    "event": "round_transition",
                    "data": {"from_round": from_round, "to_round": to_round, "topic": self._get_current_round(session).topic},
                }

        assistant_message_id = f"assistant-{uuid4()}"
        # 无论 start/reply，最终都进入面试官输出阶段。
        yield {"event": "message_start", "data": {"messageId": assistant_message_id, "role": "assistant"}}
        aggregated_text = ""
        started_at = time.perf_counter()
        messages = self._build_interviewer_messages(
            session, close_interview=close_interview, reflection_result=reflection_result
        )
        async for chunk in self.chat_model.astream(messages):
            text = self._extract_chunk_text(chunk)
            if not text:
                continue
            aggregated_text += text
            yield {"event": "message_delta", "data": {"messageId": assistant_message_id, "delta": text}}

        final_text = aggregated_text.strip()
        assistant_message = MockInterviewMessage(id=assistant_message_id, role="assistant", content=final_text)
        session.messages.append(assistant_message)
        self._touch_session(session)
        interviewer_elapsed_ms = round((time.perf_counter() - started_at) * 1000)

        if close_interview:
            session.interview_state.closed = True
        else:
            self._record_assistant_question(session)

        interview_state_payload = session.interview_state.model_dump(mode="json")
        yield {
            "event": "message_end",
            "data": {
                "messageId": assistant_message_id,
                "content": final_text,
                "interviewState": interview_state_payload,
                "elapsedMs": interviewer_elapsed_ms,
            },
        }
        yield self._build_trace_event(
            "interviewer_generation",
            MockInterviewInterviewerTracePayload(
                round=session.interview_state.currentRound,
                topic=self._get_current_round(session).topic,
                suggestedFollowUp=reflection_result.suggested_follow_up if reflection_result else "",
                closeInterview=close_interview,
                recentConversation=self._recent_conversation(session),
                finalMessage=final_text,
                elapsedMs=interviewer_elapsed_ms,
            ),
        )
        yield {
            "event": "done",
            "data": {
                "sessionId": session.session_id,
                "status": "completed" if session.interview_state.closed else "ready",
                "interviewState": interview_state_payload,
            },
        }

    def _extract_chunk_text(self, chunk: Any) -> str:
        """兼容多种 provider 的流式 chunk 结构，统一提取文本增量。"""
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif hasattr(item, "get") and isinstance(item.get("text"), str):
                    parts.append(item.get("text"))
                elif hasattr(item, "text") and isinstance(item.text, str):
                    parts.append(item.text)
            return "".join(parts)
        return ""

