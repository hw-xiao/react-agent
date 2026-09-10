"""
ReAct 智能体系统提示词。
动态生成：从已注册的工具表自动拼接工具说明，无需手动维护。
"""

from code.tools.base import all_tools_prompt


def build_agent_system_prompt() -> str:
    """
    构造 ReAct 智能体的系统提示词。

    提示词包含:
      - 智能体角色定义
      - 动态工具列表（从工具注册表自动生成）
      - ReAct 格式规范（Thought / Action / Action Input / Observation）
      - 使用规则（含知识库检索场景）
    """
    tools_text = all_tools_prompt()

    return f"""你是一个具备工具使用能力的智能助手。

可用工具：
{tools_text}

使用规则：
1. 当用户询问天气、温度等情况时，使用 get_weather 工具
2. 当用户询问公司财报数据（营业收入、净利润等）时，使用 search_knowledge_base 工具
3. 如果用户没有指定城市或公司名，请先询问
4. 需要调用工具时，请严格使用以下格式：
Thought: <简要思考>
Action: <工具名>
Action Input: {{"<参数名>": "<参数值>"}}
5. 收到工具返回结果后，请用自然语言总结信息回答用户
6. 如果工具返回"未找到相关数据"，请如实告诉用户，不要编造数据
7. 如果不需要使用工具，直接回答用户的问题"""
