"""Chroma-based RAG retrieval service for mock interview.

这个文件专门承载 RAG 检索逻辑，目标是把业务编排层（mock_interview_service）
和检索实现层（向量索引/召回/过滤）解耦，便于后续替换检索实现。
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.schemas.interview import Category, InterviewData, InterviewType
from app.schemas.jd import JDData
from app.schemas.mock_interview import (
    MockInterviewRetrievalFilters,
    MockInterviewRetrievalItem,
    MockInterviewRetrievalResult,
)
from app.schemas.resume import ResumeData
from app.services.interview_service import get_data_service

# 进程级索引锁和签名缓存：避免并发请求同时重建索引导致重复写入。
_CHROMA_INDEX_LOCK = threading.Lock()
_CHROMA_INDEX_SIGNATURE: str | None = None


@dataclass(slots=True)
class RetrievalQueryContext:
    """检索阶段的统一输入上下文。

    字段说明：
    - category/interview_type: 用于构建过滤链和兜底查询词；
    - resume_data: 预留给后续“简历特征参与检索”的扩展位；
    - jd_text/jd_data: 当前主要查询语义来源（优先 jd_text）。
    """

    category: Category
    interview_type: InterviewType
    resume_data: ResumeData
    jd_text: str = ""
    jd_data: JDData | None = None


class MockInterviewChromaRagService:
    """最基础 Chroma RAG 服务（learning 版）。

    实现范围：
    - 本地持久化 Chroma 索引；
    - 基于 OpenAI embedding 的向量召回；
    - 按 category/type/company 分层过滤与去重；
    - 输出统一 RetrievalResult 供上层服务消费。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        api_key: str,
        base_url: str | None,
        embedding_model_name: str,
        topk: int,
        candidate_topk: int,
    ) -> None:
        """初始化 RAG 服务依赖与运行参数。

        设计意图：
        - `topk` 是最终返回条数；
        - `candidate_topk` 是向量粗召回条数，必须 >= topk；
        - embedding 客户端在构造阶段一次性创建，后续复用，减少请求期开销。

        兼容性参数：
        - `check_embedding_ctx_length=False`：避免将文本预处理成 token id 数组；
        - `chunk_size=10`：适配百炼 embedding 接口单批上限约束。
        """
        self._settings = settings
        self._topk = max(topk, 1)
        self._candidate_topk = max(candidate_topk, self._topk)
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model_name,
            api_key=api_key,
            base_url=base_url,
            # 兼容 DashScope OpenAI embedding 接口：
            # 1) 关闭长度预处理，避免把文本转换成 token id 数组；
            # 2) 限制单批大小，规避百炼 embedding 的 batch 上限。
            check_embedding_ctx_length=False,
            chunk_size=10,
        )

    @staticmethod
    def _build_snippet(content: str) -> str:
        """生成展示用摘要片段。

        处理策略：
        1) 压缩多空白（换行/制表符）为单空格；
        2) 截断到 180 字符，控制响应体大小与前端展示长度。
        """
        compact = " ".join((content or "").split())
        return compact[:180]

    @staticmethod
    def _normalize_company(company: str | None) -> str:
        """归一化公司名称，提升 company 过滤命中率。

        示例：
        - "阿里巴巴（杭州）" -> "阿里巴巴"
        - " 字节 跳动 " -> "字节跳动"
        """
        if not company:
            return ""
        import re

        normalized = re.sub(r"[（(].*?[)）]", "", company)
        normalized = re.sub(r"\s+", "", normalized)
        return normalized.strip()

    @staticmethod
    def _build_query_text(context: RetrievalQueryContext) -> str:
        """构建向量检索 query 文本。

        优先级：
        - 优先使用 JD 文本（语义更丰富），并截断到 400 字符；
        - 若 JD 缺失，则退化为 `category + interview_type`，保证始终可检索。
        """
        query_text = (context.jd_text or "").strip()[:400]
        if query_text:
            return query_text
        return f"{context.category.value}\n{context.interview_type.value}"

    def _resolve_chroma_index_path(self) -> Path:
        """解析 Chroma 索引路径（支持相对/绝对路径）。

        规则：
        - 绝对路径直接使用；
        - 相对路径默认相对 `backend-learning` 根目录解析。
        """
        raw_path = Path(self._settings.mock_interview_chroma_path)
        if raw_path.is_absolute():
            return raw_path
        base_dir = Path(__file__).resolve().parents[2]
        return (base_dir / raw_path).resolve()

    def _build_document_text(self, interview: InterviewData) -> str:
        """构造用于 embedding 的文档文本。

        将标题、公司、类别、类型、正文拼接为单文本块，兼顾结构标签与正文语义，
        便于向量模型同时捕获“主题意图 + 内容细节”。
        """
        parts = [
            interview.title or "",
            interview.company or "",
            interview.category.value if interview.category else "",
            interview.interview_type.value if interview.interview_type else "",
            interview.content or "",
        ]
        return "\n".join(part for part in parts if part).strip()

    def _build_index_signature(self, interviews: list[InterviewData]) -> str:
        """基于面经核心字段生成签名，用于判断是否需要重建索引。

        当前采用“全量签名”策略：
        - 任一面经核心字段变化（id/title/content/publish_time）都会导致签名变化；
        - 变化后触发全量重建，优先保证正确性，暂不做增量更新优化。
        """
        hasher = hashlib.sha256()
        for interview in interviews:
            hasher.update(str(interview.id).encode("utf-8"))
            hasher.update((interview.publish_time or "").encode("utf-8"))
            hasher.update((interview.title or "").encode("utf-8"))
            hasher.update((interview.content or "").encode("utf-8"))
        return hasher.hexdigest()

    def _ensure_chroma_index(self, interviews: list[InterviewData]) -> chromadb.Collection:
        """确保索引可用；签名变化时全量重建。

        执行流程：
        1) 打开/创建集合；
        2) 计算最新数据签名；
        3) 在进程锁内比较签名，若变化则清空旧数据并全量写入；
        4) 返回可查询的 collection。

        并发说明：
        - `_CHROMA_INDEX_LOCK` 仅保证单进程内串行重建；
        - 多进程场景下仍建议引入外部锁或离线构建流程。
        """
        chroma_path = self._resolve_chroma_index_path()
        chroma_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(
            name="mock_interview_evidence",
            metadata={"hnsw:space": "cosine"},
        )
        signature = self._build_index_signature(interviews)

        global _CHROMA_INDEX_SIGNATURE
        with _CHROMA_INDEX_LOCK:
            if _CHROMA_INDEX_SIGNATURE != signature:
                # 重建：清空旧数据并按最新面经全量写入。
                existing = collection.get(include=[], limit=1_000_000)
                ids = existing.get("ids", []) if isinstance(existing, dict) else []
                if ids:
                    collection.delete(ids=ids)

                docs: list[str] = []
                metadatas: list[dict[str, Any]] = []
                doc_ids: list[str] = []
                for interview in interviews:
                    docs.append(self._build_document_text(interview))
                    metadatas.append(
                        {
                            "interviewId": interview.id,
                            "source": interview.source or "",
                            "sourceId": interview.source_id or "",
                            "title": interview.title or "",
                            "company": interview.company or "",
                            "category": interview.category.value if interview.category else "",
                            "interviewType": interview.interview_type.value if interview.interview_type else "",
                            "stage": interview.stage or "",
                            "publishTime": interview.publish_time or "",
                            "content": interview.content or "",
                            "companyNorm": self._normalize_company(interview.company),
                        }
                    )
                    doc_ids.append(f"interview-{interview.id}")

                if docs:
                    vectors = self._embeddings.embed_documents(docs)
                    collection.add(ids=doc_ids, documents=docs, metadatas=metadatas, embeddings=vectors)

                _CHROMA_INDEX_SIGNATURE = signature

        return collection

    def _build_filter_chain(self, context: RetrievalQueryContext) -> list[MockInterviewRetrievalFilters]:
        """构建分层过滤链：先严格，后放宽。

        层级顺序：
        1) category + interviewType + company
        2) category + interviewType
        3) category

        目标是在保持相关性的前提下尽可能提高召回稳定性。
        """
        company = self._normalize_company(context.jd_data.basicInfo.company if context.jd_data else "")
        return [
            MockInterviewRetrievalFilters(
                category=context.category,
                interviewType=context.interview_type,
                company=company or None,
            ),
            MockInterviewRetrievalFilters(
                category=context.category,
                interviewType=context.interview_type,
                company=None,
            ),
            MockInterviewRetrievalFilters(
                category=context.category,
                interviewType=None,
                company=None,
            ),
        ]

    @staticmethod
    def _metadata_matches_filters(metadata: dict[str, Any], filters: MockInterviewRetrievalFilters) -> bool:
        """判断单条候选 metadata 是否满足当前过滤条件。

        语义约定：
        - 过滤字段为 None 表示“不限制”；
        - company 使用归一化字段 `companyNorm` 比较，避免格式差异影响命中。
        """
        if filters.category and metadata.get("category") != filters.category.value:
            return False
        if filters.interviewType and metadata.get("interviewType") != filters.interviewType.value:
            return False
        if filters.company and metadata.get("companyNorm") != filters.company:
            return False
        return True

    def _to_retrieval_item(self, metadata: dict[str, Any], distance: float | None) -> MockInterviewRetrievalItem | None:
        """将 Chroma 命中结果映射为业务层统一检索项。

        容错策略：
        - interviewId 非法时直接丢弃（返回 None）；
        - category/interviewType 枚举解析失败时做安全兜底；
        - 距离转分数采用 `1 - distance` 的简化映射，仅用于展示与排序参考。
        """
        interview_id = int(metadata.get("interviewId") or 0)
        if interview_id <= 0:
            return None

        raw_category = metadata.get("category") or Category.BACKEND.value
        try:
            category = Category(raw_category)
        except Exception:
            category = Category.BACKEND

        interview_type = None
        raw_type = metadata.get("interviewType") or ""
        if raw_type:
            try:
                interview_type = InterviewType(raw_type)
            except Exception:
                interview_type = None

        score = 0.0
        if distance is not None:
            score = round(1 - float(distance), 4)

        content = metadata.get("content") or ""
        snippet = self._build_snippet(content)
        return MockInterviewRetrievalItem(
            interviewId=interview_id,
            source=metadata.get("source") or "",
            sourceId=metadata.get("sourceId") or "",
            title=metadata.get("title") or "",
            company=metadata.get("company") or None,
            category=category,
            interviewType=interview_type,
            stage=metadata.get("stage") or None,
            publishTime=metadata.get("publishTime") or "",
            snippet=snippet,
            score=score,
            reason=f"RAG召回；相似度={score:.4f}",
        )

    def retrieve(
        self, context: RetrievalQueryContext
    ) -> tuple[MockInterviewRetrievalResult, list[MockInterviewRetrievalFilters], dict[str, Any]]:
        """执行一次基础 RAG 检索并返回 trace 附加字段。

        主流程：
        1) 生成 query 文本与过滤链；
        2) 确保索引可用；
        3) 执行向量粗召回（candidate_topk）；
        4) 在应用层做分层过滤 + 去重 + topk 截断；
        5) 输出标准检索结果与调试附加信息（trace_extras）。
        """
        query_text = self._build_query_text(context)
        filter_chain = self._build_filter_chain(context)
        interviews = get_data_service().list_all_interviews()
        collection = self._ensure_chroma_index(interviews)

        # 基础向量召回：先取 candidate_topk，再在应用层按过滤链筛选与去重。
        query_vec = self._embeddings.embed_query(query_text)
        base_result = collection.query(
            query_embeddings=[query_vec],
            n_results=self._candidate_topk,
            include=["metadatas", "distances"],
        )
        raw_metadatas = (base_result.get("metadatas") or [[]])[0]
        raw_distances = (base_result.get("distances") or [[]])[0]

        selected: list[MockInterviewRetrievalItem] = []
        seen_ids: set[int] = set()
        applied_filters = filter_chain[0]
        for filters in filter_chain:
            added_this_tier = False
            for metadata, distance in zip(raw_metadatas, raw_distances):
                if not metadata:
                    continue
                if not self._metadata_matches_filters(metadata, filters):
                    continue
                item = self._to_retrieval_item(metadata, distance)
                if item is None or item.interviewId in seen_ids:
                    continue
                selected.append(item)
                seen_ids.add(item.interviewId)
                added_this_tier = True
                if len(selected) >= self._topk:
                    break
            if added_this_tier:
                applied_filters = filters
            if len(selected) >= self._topk:
                break

        result = MockInterviewRetrievalResult(
            queryText=query_text,
            appliedFilters=applied_filters,
            items=selected[: self._topk],
        )
        trace_extras = {
            "candidateTopk": self._candidate_topk,
            "topk": self._topk,
            "denseWeight": 1.0,
            "sparseWeight": None,
        }
        return result, filter_chain, trace_extras

