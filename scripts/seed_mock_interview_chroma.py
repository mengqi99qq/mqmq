r"""Seed mock interview records into Chroma for RAG testing.

用法示例：
1) 使用默认配置写入测试集合（不会影响线上集合）
   .\.venv\Scripts\python scripts\seed_mock_interview_chroma.py

2) 指定集合名和重建数量
   .\.venv\Scripts\python scripts\seed_mock_interview_chroma.py --collection mock_interview_evidence_test --count 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from langchain_openai import OpenAIEmbeddings


@dataclass(slots=True)
class MockInterviewRow:
    interview_id: int
    title: str
    company: str
    category: str
    interview_type: str
    stage: str
    content: str
    source: str = "seed-script"
    source_id: str = ""
    publish_time: str = "2026-05-08"


COMPANIES = ["字节跳动", "腾讯", "阿里巴巴", "美团", "百度", "快手", "小米", "华为"]
STAGES = ["一面", "二面", "三面", "HR面"]
TYPES = ["campus", "social"]


def _normalize_company(company: str) -> str:
    return "".join(company.replace("（", "(").replace("）", ")").split()).split("(")[0]


def _build_doc_text(row: MockInterviewRow) -> str:
    return "\n".join(
        [
            row.title,
            row.company,
            row.category,
            row.interview_type,
            row.content,
        ]
    ).strip()


def _build_mock_rows(count: int) -> list[MockInterviewRow]:
    rows: list[MockInterviewRow] = []
    for i in range(count):
        company = COMPANIES[i % len(COMPANIES)]
        stage = STAGES[i % len(STAGES)]
        interview_type = TYPES[i % len(TYPES)]
        category = "backend"
        topic = ["Redis", "MySQL", "Kafka", "系统设计", "并发编程", "缓存一致性"][i % 6]
        row = MockInterviewRow(
            interview_id=10_000 + i,
            title=f"{company}{stage}{topic}面经",
            company=company,
            category=category,
            interview_type=interview_type,
            stage=stage,
            source_id=f"seed-{i}",
            content=(
                f"候选人回答了 {topic} 的原理、场景和边界条件；"
                f"面试官追问在高并发下如何做稳定性设计，并要求给出可落地方案。"
            ),
        )
        rows.append(row)
    return rows


def _resolve_chroma_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _collection_signature(rows: list[MockInterviewRow]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(str(row.interview_id).encode("utf-8"))
        hasher.update(row.title.encode("utf-8"))
        hasher.update(row.content.encode("utf-8"))
        hasher.update(row.publish_time.encode("utf-8"))
    return hasher.hexdigest()


def _candidate_embedding_models(
    configured_model: str, base_url: str | None, override_model: str | None
) -> list[str]:
    if override_model:
        return [override_model]
    candidates: list[str] = [configured_model]
    # DashScope 的 OpenAI 兼容接口常见 embedding 模型名与 OpenAI 官方不同。
    if base_url and "dashscope" in base_url.lower():
        candidates.extend(["text-embedding-v3", "text-embedding-v2", "text-embedding-v1"])
    # 兜底保留 OpenAI 默认名，避免误伤官方端点。
    candidates.append("text-embedding-3-small")
    # 去重并保持顺序
    unique: list[str] = []
    for item in candidates:
        if item not in unique:
            unique.append(item)
    return unique


def _fake_embedding_vector(text: str, dim: int = 256) -> list[float]:
    """构造可重复的伪向量，便于离线测试 Chroma 行为。"""
    base = text.encode("utf-8")
    seed = hashlib.sha256(base).digest()
    values: list[float] = []
    while len(values) < dim:
        seed = hashlib.sha256(seed + base).digest()
        values.extend((byte / 127.5) - 1.0 for byte in seed)
    return values[:dim]


def seed_collection(
    collection_name: str,
    count: int,
    verify_query: str,
    embedding_model_override: str | None = None,
    embedding_mode: str = "fake",
) -> None:
    settings = get_settings()
    active = settings.get_active_config()
    api_key = active.get("api_key")
    base_url = active.get("base_url")
    provider = active.get("model_provider")
    if embedding_mode == "openai" and (provider != "openai" or not api_key):
        raise RuntimeError("当前配置无法生成 OpenAI embedding，请检查 MODEL_PROVIDER=openai 且 OPENAI_API_KEY 已配置。")

    chroma_path = _resolve_chroma_path(settings.mock_interview_chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    # 为避免历史测试集合的向量维度冲突（如 fake 256d -> v4 1024d），每次 seed 先删后建。
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    rows = _build_mock_rows(count)
    docs = [_build_doc_text(row) for row in rows]
    ids = [f"seed-{row.interview_id}" for row in rows]
    metadatas: list[dict[str, Any]] = [
        {
            "interviewId": row.interview_id,
            "source": row.source,
            "sourceId": row.source_id,
            "title": row.title,
            "company": row.company,
            "category": row.category,
            "interviewType": row.interview_type,
            "stage": row.stage,
            "publishTime": row.publish_time,
            "content": row.content,
            "companyNorm": _normalize_company(row.company),
        }
        for row in rows
    ]

    vectors: list[list[float]]
    query_vec: list[float]
    chosen_model = ""
    if embedding_mode == "openai":
        model_candidates = _candidate_embedding_models(
            configured_model=settings.mock_interview_embedding_model,
            base_url=base_url,
            override_model=embedding_model_override,
        )
        embeddings = None
        vectors = None
        last_error: Exception | None = None
        for model_name in model_candidates:
            try:
                current = OpenAIEmbeddings(
                    model=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    # DashScope OpenAI 兼容 embedding 端点要求 input 为字符串列表；
                    # 关闭 LangChain 的长度预处理，避免把输入转成 token id 列表。
                    check_embedding_ctx_length=False,
                    # 百炼 embedding 接口单次 batch 最大 10，显式限制分批大小。
                    chunk_size=10,
                )
                vectors = current.embed_documents(docs)
                query_vec = current.embed_query(verify_query)
                embeddings = current
                chosen_model = model_name
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if embeddings is None or vectors is None:
            raise RuntimeError(f"embedding 生成失败，尝试模型={model_candidates}") from last_error
    else:
        # 默认使用本地假向量，不依赖第三方 embedding API，更适合联调验证。
        vectors = [_fake_embedding_vector(doc) for doc in docs]
        query_vec = _fake_embedding_vector(verify_query)
        chosen_model = "fake-sha256-256d"

    # 新建后的集合直接写入。
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=vectors)

    signature = _collection_signature(rows)
    query_res = collection.query(
        query_embeddings=[query_vec],
        n_results=min(5, count),
        include=["metadatas", "distances"],
    )
    top_meta = (query_res.get("metadatas") or [[]])[0]
    top_dist = (query_res.get("distances") or [[]])[0]
    top_preview = []
    for meta, dist in zip(top_meta, top_dist):
        if not meta:
            continue
        top_preview.append(
            {
                "interviewId": meta.get("interviewId"),
                "title": meta.get("title"),
                "company": meta.get("company"),
                "score": round(1 - float(dist), 4) if dist is not None else None,
            }
        )

    print("Seed completed.")
    print(f"chroma_path={chroma_path}")
    print(f"collection={collection_name}")
    print(f"rows={count}")
    print(f"embedding_model={chosen_model}")
    print(f"signature={signature}")
    print("top_hits=" + json.dumps(top_preview, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed mock interview data into Chroma.")
    parser.add_argument("--collection", default="mock_interview_evidence_test", help="Target Chroma collection name.")
    parser.add_argument("--count", type=int, default=16, help="Number of mock interview rows to generate.")
    parser.add_argument("--verify-query", default="高并发下如何设计 Redis 缓存一致性", help="Verification query text.")
    parser.add_argument("--embedding-model", default=None, help="Optional embedding model override.")
    parser.add_argument(
        "--embedding-mode",
        default="fake",
        choices=["fake", "openai"],
        help="Embedding mode: fake(offline) or openai(real API).",
    )
    args = parser.parse_args()

    count = max(args.count, 1)
    seed_collection(
        collection_name=args.collection,
        count=count,
        verify_query=args.verify_query,
        embedding_model_override=args.embedding_model,
        embedding_mode=args.embedding_mode,
    )


if __name__ == "__main__":
    main()

