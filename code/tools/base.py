"""
工具基类：所有外部工具继承此基类，统一接口规范。
新增工具只需继承 BaseTool 并实现 execute 方法，自动注册到工具表。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """
    工具抽象基类。

    子类需实现:
      - name: 工具名称（用于 ReAct 的 Action 字段）
      - description: 工具描述（写入系统提示词）
      - parameters: 参数 schema（JSON Schema 格式）
      - execute(): 执行逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，如 'get_weather'。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，供模型理解何时使用此工具。"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """参数 JSON Schema。"""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具逻辑，返回字符串结果。"""
        ...

    def to_prompt(self) -> str:
        """生成写入系统提示词的工具说明文本。"""
        params_desc = ", ".join(
            f"{k}: {v.get('type', 'string')}" for k, v in
            self.parameters.get("properties", {}).items()
        )
        return f"- {self.name}: {self.description}\n  参数: {params_desc}"


# ── 工具注册表 ──
_TOOL_REGISTRY: Dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    """注册工具到全局表，重复注册会覆盖。"""
    _TOOL_REGISTRY[tool.name] = tool


def get_tool(name: str) -> BaseTool:
    """按名称获取工具实例。"""
    return _TOOL_REGISTRY[name]


def list_tools() -> Dict[str, BaseTool]:
    """返回所有已注册工具。"""
    return dict(_TOOL_REGISTRY)


def execute_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    按名称执行工具，统一入口。

    Args:
        name: 工具名称
        arguments: 工具参数字典

    Returns:
        工具执行结果字符串
    """
    tool = _TOOL_REGISTRY.get(name)
    if not tool:
        return f"未知工具: {name}"
    try:
        return tool.execute(**arguments)
    except TypeError as e:
        return f"工具参数错误: {e}"
    except Exception as e:
        return f"工具执行出错: {e}"


def all_tools_prompt() -> str:
    """生成所有已注册工具的说明文本，供系统提示词使用。"""
    return "\n".join(tool.to_prompt() for tool in _TOOL_REGISTRY.values())
