"""
RAG 数据模型：Document / Chunk / SearchResult。

所有模型使用 dataclass，携带完整元数据链，确保检索结果可追溯到来源文件和页码。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """
    原始文档：一个 PDF/Word/Excel 文件解析后的完整内容。

    元数据字段:
        source_file: 源文件名（如 "大众交通2026年半年度报告全文.pdf"）
        source_path: 源文件完整路径
        file_type:   文件类型（pdf/docx/xlsx/...）
        company:    公司名（从文件名或内容提取）
        report_type: 报告类型（半年报/年报/季报）
        report_year: 报告年份
        total_pages: 总页数
    """
    source_file: str
    source_path: str
    file_type: str
    pages: List[Dict[str, Any]] = field(default_factory=list)
    company: str = ""
    report_type: str = ""
    report_year: str = ""
    total_pages: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "company": self.company,
            "report_type": self.report_type,
            "report_year": self.report_year,
            "total_pages": self.total_pages,
        }


@dataclass
class Chunk:
    """
    文档分块：文档切分后的最小检索单元。

    元数据字段:
        chunk_id:     全局唯一 ID
        text:         分块文本内容
        source_file:  来源文件名
        source_path:  来源文件路径
        page_num:     所在页码（从1开始）
        section_path: 章节路径（如 ["第三节", "管理层讨论与分析"]）
        chunk_type:   分块类型（text/table/title）
        char_count:   字符数
    """
    chunk_id: str
    text: str
    source_file: str
    source_path: str
    page_num: int = 0
    section_path: List[str] = field(default_factory=list)
    chunk_type: str = "text"
    char_count: int = 0
    company: str = ""

    def to_citation(self) -> str:
        """生成引用标注字符串，如 [大众交通2026年半年报.pdf, 第15页]。"""
        parts = [self.source_file]
        if self.page_num:
            parts.append(f"第{self.page_num}页")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        return f"[{', '.join(parts)}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "source_file": self.source_file,
            "page_num": self.page_num,
            "section_path": self.section_path,
            "chunk_type": self.chunk_type,
            "char_count": self.char_count,
            "company": self.company,
        }


@dataclass
class SearchResult:
    """
    检索结果：一次检索返回的单条命中。

    字段:
        chunk:       命中的 Chunk 对象
        score:       综合相似度分数（越高越相关）
        retrieval_method: 检索方法（vector/bm25/hybrid）
    """
    chunk: Chunk
    score: float
    retrieval_method: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "retrieval_method": self.retrieval_method,
            "citation": self.chunk.to_citation(),
            "text": self.chunk.text[:200] + "..." if len(self.chunk.text) > 200 else self.chunk.text,
            "source_file": self.chunk.source_file,
            "page_num": self.chunk.page_num,
        }
