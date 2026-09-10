"""
文档加载器注册表：按文件扩展名自动选择加载器。

新增文件类型只需:
  1. 创建新的 Loader 类继承 BaseLoader
  2. 在 _LOADER_MAP 中注册扩展名

未来支持 Word/Excel/CSV 等，只需新增对应 Loader 并注册。
"""

import os
from typing import Dict, List, Optional

from code.rag.loaders.base import BaseLoader
from code.rag.loaders.pdf_loader import PdfLoader
from code.rag.models import Document
from code.utils.logger import get_logger

log = get_logger("registry")

# 扩展名 → 加载器类 的映射
_LOADER_MAP: Dict[str, BaseLoader] = {}

# 注册已实现的加载器
_pdf_loader = PdfLoader()
for ext in _pdf_loader.supported_extensions:
    _LOADER_MAP[ext] = _pdf_loader


def get_loader(file_path: str) -> Optional[BaseLoader]:
    """
    根据文件扩展名获取对应的加载器。

    Args:
        file_path: 文件路径

    Returns:
        匹配的 BaseLoader 实例，无匹配则返回 None
    """
    ext = os.path.splitext(file_path)[1].lower()
    return _LOADER_MAP.get(ext)


def load_file(file_path: str) -> Optional[Document]:
    """
    加载单个文件，自动选择加载器。

    Args:
        file_path: 文件完整路径

    Returns:
        Document 对象，不支持的类型返回 None
    """
    loader = get_loader(file_path)
    if not loader:
        log.warning(f"不支持的文件类型: {file_path}")
        return None
    return loader.load(file_path)


def load_directory(dir_path: str) -> List[Document]:
    """
    加载目录下所有支持的文件。

    Args:
        dir_path: 目录路径

    Returns:
        Document 列表，每个文件一个 Document
    """
    documents: List[Document] = []
    if not os.path.isdir(dir_path):
        log.error(f"目录不存在: {dir_path}")
        return documents

    for filename in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, filename)
        if not os.path.isfile(file_path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _LOADER_MAP:
            log.debug(f"跳过不支持的文件类型: {filename}")
            continue

        doc = load_file(file_path)
        if doc:
            documents.append(doc)

    log.info(f"目录加载完成: {dir_path} | {len(documents)} 个文档")
    return documents


def supported_extensions() -> List[str]:
    """返回所有已注册的文件扩展名。"""
    return sorted(set(_LOADER_MAP.keys()))
