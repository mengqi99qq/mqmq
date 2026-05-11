"""简历解析路由（learning 版）。

本文件是“前端上传 -> 后端解析 -> 结构化返回”的 HTTP 入口层，职责包括：
1. 接收并校验请求参数（文本或文件、runtime 覆盖参数）。
2. 根据文件类型选择解析策略（纯文本直读 / OCR 识别）。
3. 调用服务层进行 LLM 结构化抽取。
4. 将结果统一包装成前端约定的 `ResumeParseResponse`。

注意：
- 这里不实现具体 OCR 或 LLM 细节，只做路由编排与错误语义收敛。
- learning 版本优先保证“可读 + 可调试 + 可复现”。
"""

import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.resume import ElapsedTime, ParseMeta, ResumeParseRequest, ResumeParseResponse
from app.services.document_parser import (
    DIRECT_TEXT_EXTENSIONS,
    OCR_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    call_dashscope_ocr,
    extract_text_content,
)
from app.services.resume_extractor import ResumeExtractor
from app.core.config import get_settings

router = APIRouter(prefix="/resume", tags=["resume"])


def _error(code: str, message: str) -> dict:
    """统一错误响应结构，便于前端稳定解析错误码与提示文案。"""
    return {"error": {"code": code, "message": message}}


def _get_extension(filename: str) -> str:
    """从文件名提取小写扩展名（无扩展名时返回空字符串）。"""
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


@router.post("/parse-text", response_model=ResumeParseResponse)
async def parse_resume_text(payload: ResumeParseRequest):
    """解析纯文本简历（不经过 OCR）。

    使用场景：
    - 你在学习阶段希望快速验证“LLM 结构化抽取链路”。
    - 前端直接传文本，绕过文件上传与 OCR。
    """
    # 去掉首尾空白，避免仅包含空格/换行的输入进入模型调用。
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail=_error("EMPTY_TEXT", "简历文本不能为空"))

    # learning 版统一使用 .env 默认模型配置，不再接受请求级覆盖。
    extractor = ResumeExtractor()

    try:
        # 文本链路仅包含 LLM 耗时，不涉及 OCR。
        data, elapsed = extractor.extract_to_dict(text)
    except Exception as exc:
        # learning 版将底层异常折叠为统一业务错误，减少前端耦合。
        raise HTTPException(
            status_code=502,
            detail=_error("EXTRACTION_FAILED", "简历抽取失败"),
        ) from exc

    # 返回结构与主项目前端约定保持一致（尤其是 meta.elapsed 形状）。
    return ResumeParseResponse(
        data=data,
        meta=ParseMeta(
            filename="resume.txt",
            extension="txt",
            elapsed=ElapsedTime(ocr_seconds=0.0, llm_seconds=elapsed),
            guidance=(
                f"learning 模式（provider={extractor.model_provider}, model={extractor.model}）。"
            ),
        ),
    )


@router.post("/parse", response_model=ResumeParseResponse)
async def parse_resume_file(
    file: UploadFile = File(..., description="Resume file to parse"),
):
    """解析上传文件：文本直读 or OCR + LLM。

    请求体为 multipart/form-data，仅需要：
    - `file`：必填，简历文件
    """
    # 1) 基础参数校验：必须有文件名
    if not file.filename:
        raise HTTPException(status_code=400, detail=_error("NO_FILE", "未检测到上传文件"))

    # 2) 后缀校验：快速拒绝不支持类型，避免无意义计算与模型调用。
    ext = _get_extension(file.filename)
    if ext not in SUPPORTED_EXTENSIONS:
        support_text = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=_error("UNSUPPORTED_FILE_TYPE", f"不支持的文件类型: {ext}，支持: {support_text}"),
        )

    # 3) 大小校验：按配置限制上传大小，防止超大文件拖垮服务。
    settings = get_settings()
    max_size = settings.max_upload_mb * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > max_size:
        raise HTTPException(
            status_code=413,
            detail=_error("FILE_TOO_LARGE", f"文件超过 {settings.max_upload_mb}MB 限制"),
        )

    # 4) 统一使用 .env 默认配置，不再允许请求级 runtime 覆盖。
    extractor = ResumeExtractor()

    try:
        # 5) 解析策略分流
        # - txt/md：直接解码文本
        # - 图片/pdf：先 OCR 再送入 LLM
        if ext in DIRECT_TEXT_EXTENSIONS:
            text, ocr_elapsed = await extract_text_content(file_bytes)
        elif ext in OCR_EXTENSIONS:
            text, ocr_elapsed = await call_dashscope_ocr(file_bytes, ext)
        else:
            # 理论上已在上方拦截，这里属于防御式兜底。
            raise HTTPException(
                status_code=400,
                detail=_error("UNSUPPORTED_FILE_TYPE", f"不支持的文件类型: {ext}"),
            )

        # 6) LLM 抽取放到线程池执行，避免阻塞事件循环。
        data, llm_elapsed = await asyncio.to_thread(extractor.extract_to_dict, text)
    except HTTPException:
        # 已是业务异常则透传，不再二次包装。
        raise
    except Exception as exc:
        # 其他异常统一包装，便于前端提示“解析失败”。
        raise HTTPException(
            status_code=502,
            detail=_error("PARSE_FAILED", f"解析失败: {exc}"),
        ) from exc

    # 7) 返回标准化响应，包含文件信息与分阶段耗时。
    return ResumeParseResponse(
        data=data,
        meta=ParseMeta(
            filename=file.filename,
            extension=ext,
            elapsed=ElapsedTime(ocr_seconds=ocr_elapsed, llm_seconds=llm_elapsed),
            guidance=(
                f"learning 模式（provider={extractor.model_provider}, model={extractor.model}, "
                f"ocr_model={settings.dashscope_ocr_model}）。"
            ),
        ),
    )
