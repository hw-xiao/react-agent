"""
本地 Qwen 多模态 API 服务主入口。

启动方式:
    python -m code.app
    或
    .venv\\Scripts\\python.exe -m code.app

功能:
    - 加载 Qwen 多模态模型
    - 提供 OpenAI 兼容的 /v1/chat/completions 端点
    - 默认启用 ReAct 智能体，支持天气查询等工具调用
"""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from code.api.routes import router, init_service, register_exception_handlers
from code.config.settings import HOST, PORT, MODEL_ID
from code.tools import weather  # noqa: F401 — 触发天气工具注册
from code.tools import rag_tool  # noqa: F401 — 触发RAG知识库工具注册
from code.utils.logger import get_logger

log = get_logger("app")


def create_app() -> FastAPI:
    """创建 FastAPI 应用，配置中间件和路由。"""
    app = FastAPI(title="Local Qwen OpenAI-compatible API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    register_exception_handlers(app)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        body = await request.body()
        log.warning(f"请求校验失败: {request.method} {request.url}")
        log.debug(f"请求体: {body.decode('utf-8', errors='replace')}")
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return app


def main():
    """启动服务：初始化模型 → 创建应用 → 运行。"""
    log.info("=" * 60)
    log.info("本地 Qwen 多模态 API 服务启动")
    log.info(f"模型路径: {MODEL_ID}")
    log.info(f"监听地址: {HOST}:{PORT}")
    log.info("=" * 60)

    # 初始化模型（加载到 GPU/CPU）
    init_service(MODEL_ID)

    # 创建并运行 FastAPI 应用
    app = create_app()
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
