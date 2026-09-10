"""
API 请求/响应数据模型（Pydantic）。
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """单条聊天消息。"""
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容的 chat completion 请求。"""
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = 0.2
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    """OpenAI 兼容的 chat completion 响应。"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
