"""
文档分块策略：将 Document 切分为 Chunk 列表。

三种策略:
  - fixed:      固定长度分块（简单快速，适合 POC）
  - semantic:   语义分块（按段落合并，目标 300-800 字符，表格附带上下文）
  - hierarchical: 层级分块（保留父子关系，适合长文档）

核心优化:
  1. 不再逐行切分，而是合并相邻段落直到达到目标块大小
  2. 表格前后附带页面上下文，确保数字和表头在同一块
  3. 控制最小块大小（100字符），避免碎片
"""

import re
import uuid
from typing import List

from code.rag.models import Chunk, Document
from code.utils.logger import get_logger

log = get_logger("chunker")

# 目标块大小范围
MIN_CHUNK_CHARS = 100
TARGET_CHUNK_CHARS = 500
MAX_CHUNK_CHARS = 1200


def chunk_documents(
    documents: List[Document],
    strategy: str = "semantic",
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """
    将 Document 列表切分为 Chunk 列表。

    Args:
        documents:    Document 列表
        strategy:     分块策略 (fixed/semantic/hierarchical)
        chunk_size:   固定分块时的块大小（字符数）
        overlap:      固定分块时的重叠量（字符数）

    Returns:
        Chunk 列表，包含完整元数据
    """
    all_chunks: List[Chunk] = []

    for doc in documents:
        if strategy == "fixed":
            chunks = _chunk_fixed(doc, chunk_size, overlap)
        elif strategy == "semantic":
            chunks = _chunk_semantic(doc)
        elif strategy == "hierarchical":
            chunks = _chunk_hierarchical(doc, chunk_size)
        else:
            log.warning(f"未知分块策略: {strategy}，回退到 semantic")
            chunks = _chunk_semantic(doc)

        all_chunks.extend(chunks)
        log.info(f"文档 {doc.source_file} 分块: {len(chunks)} 个 chunk (策略: {strategy})")

    log.info(f"总计分块: {len(all_chunks)} 个 chunk")
    return all_chunks


def _make_chunk(
    text: str,
    doc: Document,
    page_num: int,
    section_path: List[str],
    chunk_type: str = "text",
) -> Chunk:
    """创建单个 Chunk 对象，自动生成唯一 ID。在文本前预置公司名前缀。"""
    company = doc.company or ""
    # 在chunk文本前加上公司名前缀，确保检索时能通过公司名命中
    prefix = f"【{company}】" if company else ""
    full_text = f"{prefix}{text.strip()}" if prefix else text.strip()
    return Chunk(
        chunk_id=str(uuid.uuid4())[:12],
        text=full_text,
        source_file=doc.source_file,
        source_path=doc.source_path,
        page_num=page_num,
        section_path=list(section_path),
        chunk_type=chunk_type,
        char_count=len(full_text),
        company=company,
    )


def _chunk_fixed(doc: Document, chunk_size: int, overlap: int) -> List[Chunk]:
    """
    固定长度分块：按 chunk_size 字符切分，带 overlap 重叠。

    简单但可能切断语义，适合快速验证。
    """
    chunks: List[Chunk] = []
    for page in doc.pages:
        text = page["text"]
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(_make_chunk(
                    chunk_text, doc, page["page_num"],
                    page.get("section_path", []), "text"
                ))
            start += chunk_size - overlap
    return chunks


def _chunk_semantic(doc: Document) -> List[Chunk]:
    """
    语义分块（优化版）：合并相邻段落为目标块大小。

    规则:
      1. 将页面文本按行拆分为段落
      2. 累积段落直到达到 TARGET_CHUNK_CHARS（~500字符）
      3. 超过 MAX_CHUNK_CHARS 时强制切分
      4. 表格单独成块，但附带同页上下文前缀
      5. 确保每个块 >= MIN_CHUNK_CHARS（合并过短块）
    """
    chunks: List[Chunk] = []

    for page in doc.pages:
        page_num = page["page_num"]
        section_path = page.get("section_path", [])
        text = page.get("text", "")

        # ── 表格单独成块，附带页面上下文 ──
        page_title = ""
        if text:
            lines = text.strip().split("\n")
            if lines:
                page_title = lines[0].strip()

        for table in page.get("tables", []):
            table_text = _format_table(table)
            if not table_text.strip():
                continue
            # 表格前加上页面标题作为上下文
            if page_title:
                full_text = f"{page_title}\n{table_text}"
            else:
                full_text = table_text
            chunks.append(_make_chunk(
                full_text, doc, page_num, section_path, "table"
            ))

        # ── 文本段落合并分块 ──
        if not text:
            continue

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            continue

        current_text = ""
        for para in paragraphs:
            # 如果当前块 + 新段落超过上限，先保存当前块
            if current_text and len(current_text) + len(para) + 1 > MAX_CHUNK_CHARS:
                if len(current_text) >= MIN_CHUNK_CHARS:
                    chunks.append(_make_chunk(
                        current_text, doc, page_num, section_path, "text"
                    ))
                    current_text = para
                else:
                    # 当前块太短，继续追加
                    current_text += "\n" + para
            else:
                if current_text:
                    current_text += "\n" + para
                else:
                    current_text = para

            # 达到目标大小时切分
            if len(current_text) >= TARGET_CHUNK_CHARS:
                chunks.append(_make_chunk(
                    current_text, doc, page_num, section_path, "text"
                ))
                current_text = ""

        # 页末剩余文本
        if current_text.strip():
            if len(current_text) < MIN_CHUNK_CHARS and chunks:
                # 过短则合并到上一块（如果同页同章节）
                last = chunks[-1]
                if last.page_num == page_num and last.chunk_type == "text":
                    last.text = last.text + "\n" + current_text
                    last.char_count = len(last.text)
                else:
                    chunks.append(_make_chunk(
                        current_text, doc, page_num, section_path, "text"
                    ))
            else:
                chunks.append(_make_chunk(
                    current_text, doc, page_num, section_path, "text"
                ))

    return chunks


def _chunk_hierarchical(doc: Document, chunk_size: int) -> List[Chunk]:
    """
    层级分块：保留章节父子关系，适合长文档检索。

    每个块携带 section_path，可用于父子回溯。
    """
    chunks: List[Chunk] = []

    for page in doc.pages:
        page_num = page["page_num"]
        section_path = page.get("section_path", [])
        text = page.get("text", "")

        if not text:
            continue

        # 按章节标题切分
        section_chunks = re.split(r'\n(?=第[一二三四五六七八九十]+[节章])', text)

        for section_text in section_chunks:
            if not section_text.strip():
                continue
            # 长段落再按 chunk_size 切分
            if len(section_text) > chunk_size:
                for i in range(0, len(section_text), chunk_size):
                    chunk_text = section_text[i:i + chunk_size]
                    if chunk_text.strip():
                        chunks.append(_make_chunk(
                            chunk_text, doc, page_num, section_path, "text"
                        ))
            else:
                chunks.append(_make_chunk(
                    section_text, doc, page_num, section_path, "text"
                ))

    return chunks


def _format_table(table: List[List[str]]) -> str:
    """
    将 pdfplumber 提取的表格列表格式化为可读文本。

    Args:
        table: 二维列表，每个元素是一行单元格

    Returns:
        格式化的表格文本
    """
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(cell or "").strip() for cell in row]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)
