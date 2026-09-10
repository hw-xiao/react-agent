"""
日志系统：同时输出到控制台和文件，按日期轮转。
ReAct 各阶段使用独立日志函数，便于追踪智能体推理过程。
"""

import logging
import os
from datetime import datetime
from typing import Optional

from code.config.settings import LOG_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _ensure_log_dir() -> str:
    """确保日志目录存在，返回路径。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


def _init_logger() -> logging.Logger:
    """初始化根日志器，添加控制台和文件两个 handler。"""
    global _initialized
    logger = logging.getLogger("qwen_agent")
    if _initialized:
        return logger

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 控制台输出（INFO 及以上）
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # 文件输出（DEBUG 及以上，按日期命名）
    log_dir = _ensure_log_dir()
    log_file = os.path.join(log_dir, f"agent_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _initialized = True
    logger.info(f"日志系统初始化完成，日志文件: {log_file}")
    return logger


def get_logger(name: str = "qwen_agent") -> logging.Logger:
    """获取命名日志器，所有子模块统一调用此函数。"""
    _init_logger()
    return logging.getLogger("qwen_agent").getChild(name)


# ── ReAct 各阶段专用日志函数 ──

def log_react_start(user_query: str) -> None:
    """记录 ReAct 循环开始。"""
    log = get_logger("react")
    log.info("=" * 60)
    log.info("[ReAct] 循环启动")
    log.info(f"[ReAct] 用户输入: {user_query}")
    log.info("=" * 60)


def log_react_thought(round_num: int, thought: str) -> None:
    """记录模型思考阶段。"""
    log = get_logger("react")
    log.info(f"[ReAct] 第{round_num}轮 Thought (思考)")
    log.debug(f"[ReAct] 思考内容:\n{thought}")


def log_react_action(round_num: int, tool_name: str, arguments: dict) -> None:
    """记录工具调用决策。"""
    log = get_logger("react")
    log.info(f"[ReAct] 第{round_num}轮 Action (行动)")
    log.info(f"[ReAct] 调用工具: {tool_name}")
    log.info(f"[ReAct] 工具参数: {arguments}")


def log_react_observation(round_num: int, tool_result: str) -> None:
    """记录工具返回结果。"""
    log = get_logger("react")
    log.info(f"[ReAct] 第{round_num}轮 Observation (观察)")
    log.debug(f"[ReAct] 工具结果:\n{tool_result}")


def log_react_final(answer: str) -> None:
    """记录最终回答。"""
    log = get_logger("react")
    log.info("[ReAct] 循环完成，生成最终回答")
    log.info(f"[ReAct] 最终回答: {answer[:200]}{'...' if len(answer) > 200 else ''}")


def log_react_no_tool(response: str) -> None:
    """记录模型未触发工具调用的情况。"""
    log = get_logger("react")
    log.info("[ReAct] 模型未请求工具调用，直接返回回复")
    log.debug(f"[ReAct] 原始回复: {response}")


def log_react_error(stage: str, error: Exception) -> None:
    """记录 ReAct 各阶段异常。"""
    log = get_logger("react")
    log.error(f"[ReAct] {stage} 阶段异常: {error}", exc_info=True)


def log_model_generate(method: str, messages_count: int, max_tokens: int) -> None:
    """记录模型推理调用。"""
    log = get_logger("model")
    log.info(f"[Model] {method} 调用 | 消息数={messages_count} | max_tokens={max_tokens}")


def log_model_response(method: str, response: str) -> None:
    """记录模型生成结果。"""
    log = get_logger("model")
    log.info(f"[Model] {method} 生成完成")
    log.debug(f"[Model] 响应内容:\n{response}")


def log_api_request(endpoint: str, payload_summary: str) -> None:
    """记录 API 请求。"""
    log = get_logger("api")
    log.info(f"[API] {endpoint} | {payload_summary}")


def log_api_response(endpoint: str, answer: str) -> None:
    """记录 API 响应。"""
    log = get_logger("api")
    log.info(f"[API] {endpoint} 响应完成")
    log.debug(f"[API] 响应内容: {answer[:200]}{'...' if len(answer) > 200 else ''}")
