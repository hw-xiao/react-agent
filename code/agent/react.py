"""
ReAct 智能体循环：思考(Thought) → 行动(Action) → 观察(Observation) → 回答。

核心流程:
  1. 用户输入 + 系统提示词 → 模型第一轮推理
  2. 解析模型输出是否包含工具调用（Action + Action Input）
  3. 如果有 → 执行工具 → 将结果作为 Observation 回传模型
  4. 模型第二轮推理 → 生成自然语言最终回答
  5. 如果没有工具调用 → 直接返回模型回复
"""

import json
import re
from typing import Any, Dict, List, Optional

from code.agent.prompts import build_agent_system_prompt
from code.models.qwen_adapter import QwenAdapter
from code.tools.base import execute_tool
from code.utils.logger import (
    get_logger, log_react_start, log_react_thought,
    log_react_action, log_react_observation,
    log_react_final, log_react_no_tool, log_react_error,
)

log = get_logger("react")


def parse_tool_call(response: str) -> Optional[Dict[str, Any]]:
    """
    从模型输出中解析工具调用。

    解析格式:
        Thought: ...
        Action: <tool_name>
        Action Input: {"city": "..."}

    Args:
        response: 模型生成的文本

    Returns:
        {"name": tool_name, "arguments": {...}} 或 None
    """
    action_match = re.search(r"Action:\s*(\w+)", response)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)

    if action_match and input_match:
        tool_name = action_match.group(1).strip()
        try:
            arguments = json.loads(input_match.group(1))
            return {"name": tool_name, "arguments": arguments}
        except json.JSONDecodeError as e:
            log.warning(f"工具参数 JSON 解析失败: {e}")
            return None
    return None


def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    """从 messages 中提取最后一条 user 消息的文本内容。"""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                return "\n".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return str(content)
    return ""


def run_react_agent(
    service: QwenAdapter,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
) -> str:
    """
    执行完整的 ReAct 智能体循环。

    流程:
      [用户输入] → [模型思考: 是否需要工具?]
           ├─ 是 → [Action: 调用工具] → [Observation: 工具结果] → [模型: 生成最终回答]
           └─ 否 → 直接返回模型回复

    Args:
        service: QwenAdapter 模型实例
        messages: OpenAI 格式的消息列表
        max_tokens: 生成最大 token 数
        temperature: 采样温度

    Returns:
        智能体最终回答字符串
    """
    # ── 阶段0: 准备 ──
    user_text = _extract_user_text(messages)
    agent_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_agent_system_prompt()},
        {"role": "user", "content": user_text},
    ]
    log_react_start(user_text)

    # ── 阶段1: 模型思考 — 判断是否需要调用工具 ──
    try:
        first_response = service.generate_multiturn(agent_messages, max_tokens, temperature)
    except Exception as e:
        log_react_error("第一轮推理", e)
        return f"[智能体第一轮推理失败: {e}]"

    log_react_thought(1, first_response)

    # ── 阶段2: 解析工具调用 ──
    tool_call = parse_tool_call(first_response)

    if not tool_call:
        # 无工具调用，直接返回模型回复
        log_react_no_tool(first_response)
        return first_response

    log_react_action(1, tool_call["name"], tool_call["arguments"])

    # ── 阶段3: 执行工具，获取结果 ──
    try:
        tool_result = execute_tool(tool_call["name"], tool_call["arguments"])
        log_react_observation(1, tool_result)
    except Exception as e:
        tool_result = f"工具执行出错: {e}"
        log_react_error("工具执行", e)

    # ── 阶段4: 将工具结果回传模型，生成最终回答 ──
    agent_messages.append({"role": "assistant", "content": first_response})
    agent_messages.append({
        "role": "user",
        "content": f"Observation: {tool_result}\n\n请根据以上工具返回的结果，用自然语言回答用户的问题。如果结果中没有用户需要的数据，请如实说明。",
    })

    try:
        final_response = service.generate_multiturn(agent_messages, max_tokens, temperature)
        log_react_final(final_response)
        return final_response
    except Exception as e:
        log_react_error("第二轮推理", e)
        return f"工具查询结果如下：\n{tool_result}"
