"""JD extraction API routes for learning backend."""

import asyncio

from fastapi import APIRouter, HTTPException

from app.schemas.jd import JDData, JDExtractRequest, JDExtractResponse
from app.services.jd_extractor import JDExtractor, InvalidJDContentError

router = APIRouter(prefix="/jd", tags=["jd"])

MAX_TEXT_LENGTH = 30000


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


@router.post("/extract", response_model=JDExtractResponse)
async def extract_jd(request: JDExtractRequest):
    """Extract structured data from JD plain text."""
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail=_error("EMPTY_TEXT", "JD 文本不能为空"),
        )

    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=_error("TEXT_TOO_LARGE", f"文本超过 {MAX_TEXT_LENGTH} 字符限制"),
        )

    try:
        # learning 版统一使用 .env 默认模型参数，不再读取请求级 runtimeConfig。
        extractor = JDExtractor()
        # 与 resume 路由保持一致：将同步抽取逻辑放到线程池执行，避免阻塞事件循环。
        result, elapsed = await asyncio.to_thread(extractor.extract_to_dict, request.text)
        return JDExtractResponse(
            data=JDData(**result),
            elapsed_seconds=elapsed,
        )
    except InvalidJDContentError as exc:
        raise HTTPException(
            status_code=400,
            detail=_error("INVALID_JD_CONTENT", str(exc)),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=_error("JD_EXTRACT_FAILED", f"JD 提取失败: {exc}"),
        ) from exc
