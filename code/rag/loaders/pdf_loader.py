"""
PDF 文档加载器：使用 pdfplumber 解析文本和表格。

解析策略:
  - 逐页提取文本，保留段落结构
  - 识别表格数据，标记为 table 类型块
  - 记录页码和章节路径元数据
  - 跳过纯空白页
"""

import os
from typing import List

import pdfplumber

from code.rag.loaders.base import BaseLoader
from code.rag.models import Document
from code.utils.logger import get_logger

log = get_logger("pdf_loader")


class PdfLoader(BaseLoader):
    """PDF 文档加载器，基于 pdfplumber。"""

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def load(self, file_path: str) -> Document:
        """
        解析 PDF 文件，返回包含逐页文本和表格的 Document。

        Args:
            file_path: PDF 文件路径

        Returns:
            Document 对象，pages 列表中每页包含 text 和 tables
        """
        filename = os.path.basename(file_path)
        meta = self.extract_metadata_from_filename(filename)

        log.info(f"开始解析PDF: {filename}")
        log.info(f"  元数据: 公司={meta['company']}, 类型={meta['report_type']}, 年份={meta['report_year']}")

        pages = []
        current_section_path: List[str] = []

        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            log.info(f"  总页数: {total}")

            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text() or ""
                tables = page.extract_tables() or []

                # 跳过空白页
                if not page_text.strip() and not tables:
                    continue

                # 简单章节检测：行首为"第X节"或全大写标题
                for line in page_text.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("第") and "节" in stripped[:6]:
                        current_section_path = [stripped]
                        break

                # 清理文本：合并多余空行
                cleaned_lines = []
                for line in page_text.split("\n"):
                    if line.strip():
                        cleaned_lines.append(line.strip())
                cleaned_text = "\n".join(cleaned_lines)

                pages.append({
                    "page_num": page_num,
                    "text": cleaned_text,
                    "tables": tables,
                    "section_path": list(current_section_path),
                })

                if page_num % 50 == 0:
                    log.debug(f"  已解析 {page_num}/{total} 页")

        doc = Document(
            source_file=filename,
            source_path=file_path,
            file_type="pdf",
            pages=pages,
            company=meta["company"],
            report_type=meta["report_type"],
            report_year=meta["report_year"],
            total_pages=len(pages),
        )

        log.info(f"  解析完成: {len(pages)} 页, {sum(len(p['text']) for p in pages)} 字符, {sum(len(p['tables']) for p in pages)} 个表格")
        return doc
