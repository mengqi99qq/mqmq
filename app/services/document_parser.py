"""Document parsing helpers: text extraction and OCR extraction."""

from __future__ import annotations

import asyncio
import base64
import time
from io import BytesIO

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from app.core.config import get_settings

DIRECT_TEXT_EXTENSIONS = {"txt", "md"}
OCR_EXTENSIONS = {"pdf", "jpg", "jpeg", "png"}
SUPPORTED_EXTENSIONS = DIRECT_TEXT_EXTENSIONS | OCR_EXTENSIONS
PDF_IMAGE_RENDER_SCALE = 1.35
PDF_IMAGE_JPEG_QUALITY = 68
PDF_IMAGE_RENDER_MAX_PAGES = 6


async def extract_text_content(file_bytes: bytes) -> tuple[str, float]:
    """Decode plain text bytes with utf-8/gbk fallback."""
    start_time = time.time()
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode("gbk")
        except UnicodeDecodeError:
            text = file_bytes.decode("utf-8", errors="ignore")
    return text, round(time.time() - start_time, 2)


async def render_pdf_to_image_parts(
    file_bytes: bytes, max_pages: int = PDF_IMAGE_RENDER_MAX_PAGES
) -> list[dict]:
    """Render a PDF to JPEG image parts for OCR model input."""

    def _render() -> list[dict]:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(file_bytes)
        page_count = min(len(pdf), max_pages)
        image_parts: list[dict] = []
        try:
            for page_index in range(page_count):
                page = pdf[page_index]
                bitmap = page.render(scale=PDF_IMAGE_RENDER_SCALE)
                image = bitmap.to_pil()
                buffer = BytesIO()
                image.save(
                    buffer,
                    format="JPEG",
                    quality=PDF_IMAGE_JPEG_QUALITY,
                    optimize=True,
                )
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    }
                )
        finally:
            if hasattr(pdf, "close"):
                pdf.close()
        return image_parts

    return await asyncio.to_thread(_render)


async def call_dashscope_ocr(
    file_bytes: bytes,
    file_extension: str,
) -> tuple[str, float]:
    """Use DashScope OpenAI-compatible OCR model to get plain text."""
    settings = get_settings()
    active_config = settings.get_active_config()
    model_provider = active_config["model_provider"]
    api_key = active_config["api_key"]
    base_url = active_config["base_url"]

    if model_provider != "openai":
        raise ValueError("Learning skeleton 的 OCR 当前仅支持 openai provider")
    if not api_key or not base_url:
        raise ValueError("OCR 需要 API key 与 base_url（OpenAI-compatible）")

    start_time = time.time()
    chat_model = init_chat_model(
        model=settings.dashscope_ocr_model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
    )

    ext = file_extension.lower()
    if ext == "pdf":
        image_parts = await render_pdf_to_image_parts(file_bytes)
        if not image_parts:
            raise ValueError("PDF 渲染失败，无法进行 OCR")
    else:
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        image_parts = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        ]

    prompt = (
        "请做OCR识别并输出全文纯文本。"
        "不要解释，不要总结，不要添加额外标记；保持原文段落与换行。"
    )
    messages = [HumanMessage(content=[{"type": "text", "text": prompt}, *image_parts])]
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: chat_model.invoke(messages))
    content = getattr(response, "content", "")
    if isinstance(content, str):
        text = content.strip()
    else:
        text = str(content).strip()
    return text, round(time.time() - start_time, 2)
