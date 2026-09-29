"""财经热点工具：统一纳入 ReAct/工具层，后续可接入 MCP 标准接口。

本工具不依赖外部 MCP SDK，而是通过项目现有 BaseTool 注册机制实现统一调用。
这样后续如果接入更标准的 MCP Server，接口层无需大改。
"""

from typing import Any, Dict

from code.tools.base import BaseTool, register_tool
from code.utils.logger import get_logger
from code.utils.market_report import run_market_report_cycle

log = get_logger("market_tool")


class MarketNewsTool(BaseTool):
    """财经热点快报工具：每次调用可抓取资讯、归纳方向并尝试发送邮件。"""

    @property
    def name(self) -> str:
        return "get_market_news_briefing"

    @property
    def description(self) -> str:
        return (
            "获取当天财经热点简报，汇总宏观政策、A股行业、数字资产、商品和全球市场关注方向，"
            "并尝试发送邮件。适用于财经、宏观、市场观察等场景。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "抓取的资讯条数，默认 8，可选 5-12",
                    "default": 8,
                }
            },
            "required": [],
        }

    def execute(self, limit: int = 8) -> str:
        try:
            log.info("调用财经热点工具，limit=%s", limit)
            report = run_market_report_cycle()
            summary = report.get("summary", "")
            headlines = report.get("headlines", [])[: max(0, min(int(limit), 12))]
            lines = [summary]
            if headlines:
                lines.append("")
                lines.append("精选资讯：")
                for idx, item in enumerate(headlines, start=1):
                    title = item.get("title", "")
                    link = item.get("link", "")
                    lines.append(f"{idx}. {title}")
                    if link:
                        lines.append(f"   {link}")
            return "\n".join(lines)
        except Exception as exc:  # pragma: no cover
            log.error("财经热点工具执行失败: %s", exc, exc_info=True)
            return f"财经热点工具执行失败: {exc}"


register_tool(MarketNewsTool())
