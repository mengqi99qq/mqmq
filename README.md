# backend-learning

这是一个用于“手动复现 FaceTomato 后端”的最小骨架项目。

## 目标

- 先跑通一个可启动、可访问的 FastAPI 服务。
- 保留与主项目接近的目录分层（`api/routes`、`services`、`schemas`、`core`、`prompts`、`utils`）。
- 你可以在此基础上一步步替换成真实实现（LLM、OCR、文件解析、并发抽取等）。

## 快速启动

```bash
cd backend-learning
uv sync
cp .env.example .env
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 6523
```

打开：

- API 文档: `http://127.0.0.1:6523/docs`
- 健康检查: `http://127.0.0.1:6523/health`

## 已提供接口

- `GET /health`
- `GET /`
- `POST /api/resume/parse-text`
- `POST /api/mock-interview/session/stream-create`（SSE）
- `POST /api/mock-interview/session/{sessionId}/stream`（SSE）

`/api/resume/parse-text` 当前已升级为“LLM 结构化抽取（学习版）”：
请求 -> 路由 -> runtime 合并 -> extractor -> LangChain structured output -> 返回结果。

## 推荐复现路线（从易到难）

1. 看懂 `app/main.py`（应用启动、路由挂载、CORS）。
2. 看懂 `app/core/config.py`（环境变量与配置对象）。
3. 看懂 `app/api/routes/resume.py`（请求校验与错误处理）。
4. 看懂 `app/services/resume_extractor.py`（业务逻辑入口）。
5. 阅读并改造 `app/services/resume_extractor.py` 的 LLM 调用（可先换 prompt 再换模型）。
6. 新增文件上传解析接口（PDF/DOCX/图片）。
7. 再引入结构化输出兜底、并行抽取、限流器。

## 使用前准备

- 在 `.env` 中填好 `OPENAI_API_KEY`。
- 如果你走兼容 OpenAI 协议网关，再配置 `OPENAI_BASE_URL`。

## 你可以动手的练习

- 练习 1：把单次抽取拆成并行抽取（basicInfo/skills/highlights 分开调用）。
- 练习 2：在 `invoke_with_fallback` 增加“JSON 修复重试”逻辑。
- 练习 3：扩展 `RuntimeConfig`，支持按请求覆盖 OCR/Speech 参数。
