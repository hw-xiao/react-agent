"""
FastAPI 路由：OpenAI 兼容的 chat completions 端点。

支持三种模式:
  1. ReAct 智能体（默认）— 模型可调用工具获取实时数据和知识库检索
  2. 直接生成 — 普通多模态推理（工具关闭时）
  3. RAG 检索增强 — 通过 ReAct 工具 search_knowledge_base 实现
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from code.agent.react import run_react_agent
from code.config.settings import (
    ENABLE_TOOLS, GEN_TIMEOUT, MODEL_ID,
    SERVER_MAX_TOKENS, MAX_NEW_TOKENS,
)
from code.models.qwen_adapter import QwenAdapter
from code.utils.logger import get_logger, log_api_request, log_api_response
from code.utils.market_report import run_market_report_cycle

log = get_logger("api")

router = APIRouter()

# 全局模型实例（在 app.py 中初始化）
_service: QwenAdapter = None


def init_service(model_id: str) -> None:
    """初始化模型实例，应在应用启动时调用。"""
    global _service
    _service = QwenAdapter(model_id)


def get_service() -> QwenAdapter:
    """获取已初始化的模型实例。"""
    if _service is None:
        raise RuntimeError("模型未初始化，请先调用 init_service()")
    return _service


def normalize_chat_payload(payload: dict) -> dict:
    """
    将客户端请求规范化为统一格式。

    处理 content 为字符串/列表/字典的情况，统一 images 字段。
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    normalized_messages = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("Each message must be an object")

        role = str(msg.get("role") or "user")
        content = msg.get("content")
        images = msg.get("images") or []

        if isinstance(content, str):
            normalized_content = content
        elif isinstance(content, list):
            normalized_content = []
            for part in content:
                if isinstance(part, str):
                    normalized_content.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    normalized_content.append(part)
                else:
                    normalized_content.append({"type": "text", "text": str(part)})
        elif isinstance(content, dict):
            normalized_content = [{"type": "text", "text": str(content.get("text", ""))}]
            for key in ("image", "image_url"):
                if key in content and content[key] is not None:
                    value = content[key]
                    if isinstance(value, dict):
                        value = value.get("url")
                    normalized_content.append({"type": "image_url", "image_url": {"url": value}})
        else:
            normalized_content = str(content) if content is not None else ""

        if images:
            if isinstance(normalized_content, str):
                normalized_content = [{"type": "text", "text": normalized_content}]
            if not isinstance(normalized_content, list):
                normalized_content = [{"type": "text", "text": str(normalized_content)}]
            for image in images:
                if isinstance(image, dict):
                    image_value = image.get("url") or image.get("image_url")
                else:
                    image_value = image
                normalized_content.append({"type": "image_url", "image_url": {"url": image_value}})

        normalized_messages.append({"role": role, "content": normalized_content})

    return {
        "model": payload.get("model") or MODEL_ID,
        "messages": normalized_messages,
        "max_tokens": int(payload.get("max_tokens") or MAX_NEW_TOKENS),
        "temperature": float(payload.get("temperature") or 0.2),
        "stream": bool(payload.get("stream") or False),
    }


@router.get("/health")
def health():
    """健康检查端点。"""
    return {"status": "ok", "model": MODEL_ID}


@router.get("/v1/models")
def list_models():
    """列出可用模型。"""
    return {
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "local",
        }]
    }


@router.get("/v1/market/report")
def trigger_market_report():
    """手动触发财经热点汇总与邮件推送。"""
    report = run_market_report_cycle()
    return {
        "status": "ok",
        "message": "财经热点汇总已生成并尝试推送邮件",
        "email_sent": report.get("email_sent", False),
        "receiver": report.get("email_receiver", ""),
        "generated_at": report.get("generated_at", ""),
    }


@router.post("/v1/chat/completions")
async def chat_completion(request: Request):
    """
    OpenAI 兼容的 chat completion 端点。

    根据配置自动选择推理模式:
      - ENABLE_TOOLS=1 (默认): ReAct 智能体，支持天气查询和知识库检索
      - ENABLE_TOOLS=0: 直接模型生成
    """
    try:
        raw_body = await request.body()
        payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}

        log_api_request("/v1/chat/completions", f"messages={len(payload.get('messages', []))}")

        normalized = normalize_chat_payload(payload)
        model_name = normalized["model"]

        # 服务端限制最大生成 token 数
        requested = int(normalized.get("max_tokens", MAX_NEW_TOKENS))
        max_to_generate = min(requested, SERVER_MAX_TOKENS)
        if requested != max_to_generate:
            log.info(f"限制 max_tokens: {requested} -> {max_to_generate}")

        # 选择推理模式
        use_agent = ENABLE_TOOLS
        service = get_service()

        try:
            if use_agent:
                log.info("使用 ReAct 智能体模式")
                answer = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_react_agent, service, normalized["messages"],
                        max_to_generate, normalized["temperature"],
                    ),
                    timeout=GEN_TIMEOUT * 2,
                )
            else:
                log.info("使用直接生成模式")
                answer = await asyncio.wait_for(
                    asyncio.to_thread(
                        service.generate, normalized["messages"],
                        max_to_generate, normalized["temperature"],
                    ),
                    timeout=GEN_TIMEOUT,
                )
        except asyncio.TimeoutError:
            log.error("模型生成超时")
            answer = "[模型生成超时，已中止；请重试或减小 max_tokens/图片大小]"
        except Exception as exc:
            log.error(f"模型生成失败: {exc}", exc_info=True)
            answer = "[模型生成失败，已返回占位回复；请重试或改用纯文本输入]"

        # 空响应兜底
        if not isinstance(answer, str) or not answer.strip():
            log.warning("模型返回空响应")
            answer = "[模型未生成任何内容，可能已 OOM 或生成失败，请查看服务器日志]"

        log_api_response("/v1/chat/completions", answer)

        resp_obj = {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        # 流式响应兼容
        if normalized.get("stream"):
            def event_stream():
                try:
                    yield "data: " + json.dumps(resp_obj, ensure_ascii=False) + "\n\n"
                except Exception as e:
                    log.error(f"SSE 生成异常: {e}", exc_info=True)
                    return

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return resp_obj

    except json.JSONDecodeError as exc:
        log.error(f"JSON 解析失败: {exc.msg}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc.msg}")
    except Exception as exc:
        log.error(f"请求处理异常: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


def register_exception_handlers(app):
    """注册全局异常处理器。"""

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        log.error(f"未处理异常: {exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(exc)})
