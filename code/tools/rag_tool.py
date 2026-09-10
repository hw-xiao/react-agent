"""
RAG 知识库检索工具：让 ReAct 智能体能查询财报文档。

模型通过 ReAct 格式调用此工具:
  Action: search_knowledge_base
  Action Input: {"query": "大众交通2026年半年报营业收入"}

工具内部:
  1. 延迟初始化 RagPipeline（首次调用时加载 PDF）
  2. 调用 pipeline.query_with_citations() 获取带来源标注的结果
  3. 返回给模型，模型基于结果生成自然语言回答
"""

from typing import Any, Dict

from code.tools.base import BaseTool, register_tool
from code.utils.logger import get_logger

log = get_logger("rag_tool")

# 全局 pipeline 实例（延迟初始化，避免启动时解析 PDF）
_pipeline = None


def get_pipeline():
    """
    获取全局 RagPipeline 实例，首次调用时初始化。

    从 settings.py 读取数据目录和参数。
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from code.config.settings import RAG_DATA_DIR, RAG_TEMP_DIR, RAG_CACHE_DIR, RAG_CHUNK_STRATEGY

    if not RAG_DATA_DIR:
        log.error("RAG_DATA_DIR 未配置，无法初始化 RAG 管道")
        return None

    from code.rag.pipeline import RagPipeline

    _pipeline = RagPipeline(
        data_dir=RAG_DATA_DIR,
        temp_dir=RAG_TEMP_DIR,
        cache_dir=RAG_CACHE_DIR,
        chunk_strategy=RAG_CHUNK_STRATEGY,
    )
    return _pipeline


class KnowledgeBaseTool(BaseTool):
    """
    知识库检索工具：查询财报文档获取财务数据。

    当用户询问公司年报/半年报中的财务数据时使用此工具。
    返回结果包含数据出处（文件名、页码），找不到则说明。
    """

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return ("查询财报知识库，检索公司年报/半年报中的财务数据（营业收入、净利润、"
                "总资产等）。返回结果包含数据出处标记。当用户询问某公司的财务数据时使用此工具。")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "查询内容，如 '大众交通2026年半年报营业收入' 或 '金健米业净利润'",
                }
            },
            "required": ["query"],
        }

    def execute(self, query: str = "") -> str:
        """
        执行知识库检索。

        Args:
            query: 查询文本

        Returns:
            带来源标注的检索结果，找不到则返回"未找到相关数据"
        """
        if not query:
            return "错误：缺少查询参数"

        log.info(f"知识库检索: {query}")

        pipeline = get_pipeline()
        if pipeline is None:
            return "知识库未配置，请联系管理员设置 RAG_DATA_DIR 环境变量。"

        try:
            result = pipeline.query_with_citations(query, top_k=3, max_chars_per_chunk=300)
            if "未找到相关数据" in result:
                log.info(f"知识库检索: 未找到 '{query}' 的相关数据")
            else:
                citation_count = result.count("来源:")
                log.info(f"知识库检索完成，返回 {citation_count} 条引用")
                log.info(f"检索结果摘要:\n{result[:500]}")
            return result
        except Exception as e:
            log.error(f"知识库检索失败: {e}", exc_info=True)
            return f"知识库检索出错: {e}"


# 模块加载时自动注册
register_tool(KnowledgeBaseTool())
