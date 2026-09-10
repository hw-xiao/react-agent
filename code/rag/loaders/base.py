"""
文档加载器基类：所有文件类型加载器继承此基类。

新增文件类型只需:
  1. 继承 BaseLoader
  2. 实现 load() 方法
  3. 在 registry.py 中注册扩展名映射
"""

from abc import ABC, abstractmethod
from typing import List

from code.rag.models import Document
from code.utils.logger import get_logger

log = get_logger("loader")


class BaseLoader(ABC):
    """
    文档加载器抽象基类。

    子类需实现:
      - load() -> List[Document]: 解析文件，返回 Document 列表
      - supported_extensions: 支持的文件扩展名列表
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名，如 ['.pdf']。"""
        ...

    @abstractmethod
    def load(self, file_path: str) -> Document:
        """
        解析单个文件，返回 Document 对象。

        Args:
            file_path: 文件完整路径

        Returns:
            Document 对象，包含按页解析的文本和表格
        """
        ...

    @staticmethod
    def extract_metadata_from_filename(filename: str) -> dict:
        """
        从文件名提取元数据（公司名、报告类型、年份）。

        示例:
          "大众交通：大众交通(集团)股份有限公司2026年半年度报告全文.pdf"
          → {"company": "大众交通", "report_type": "半年报", "report_year": "2026"}

          "金健米业2026年半年度报告全文.pdf"
          → {"company": "金健米业", "report_type": "半年报", "report_year": "2026"}
        """
        meta = {"company": "", "report_type": "", "report_year": ""}

        # 提取年份
        import re
        year_match = re.search(r"(20\d{2})", filename)
        if year_match:
            meta["report_year"] = year_match.group(1)

        # 提取报告类型
        if "半年" in filename or "半年度" in filename:
            meta["report_type"] = "半年报"
        elif "年度" in filename or "年报" in filename:
            meta["report_type"] = "年报"
        elif "季" in filename:
            meta["report_type"] = "季报"

        # 提取公司名：取冒号前的部分，或文件名开头到年份前
        if "：" in filename:
            meta["company"] = filename.split("：")[0].strip()
        elif "：" in filename:
            meta["company"] = filename.split(":")[0].strip()
        elif year_match:
            meta["company"] = filename[:year_match.start()].strip()

        return meta
