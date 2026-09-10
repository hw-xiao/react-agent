"""
向量存储接口层：管理向量索引和相似度搜索。

当前实现: InMemoryStore（numpy 矩阵 + 余弦相似度）
未来扩展: FaissStore（FAISS IndexFlatIP）、MilvusStore（Milvus 向量数据库）

只需继承 BaseVectorStore 实现 add()/search()，即可替换存储后端。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from code.rag.models import Chunk
from code.utils.logger import get_logger

log = get_logger("vectorstore")


class BaseVectorStore(ABC):
    """向量存储抽象基类。"""

    @abstractmethod
    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """
        添加文档块和对应向量到索引。

        Args:
            chunks:  文档块列表
            vectors: 对应的向量矩阵, shape=(n, dim)
        """
        ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        向量相似度搜索。

        Args:
            query_vec: 查询向量, shape=(1, dim)
            top_k:     返回前 K 个结果

        Returns:
            [(chunk_index, score), ...] 列表
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """返回索引中的向量数量。"""
        ...


class InMemoryStore(BaseVectorStore):
    """
    内存向量存储：numpy 矩阵 + L2 归一化后余弦相似度。

    优点: 零依赖，适合文档量 <10万 的场景
    扩展: 文档量大时替换为 FaissStore
    """

    def __init__(self):
        self._vectors: np.ndarray = None
        self._chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """添加向量到内存索引。"""
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._chunks.extend(chunks)
        log.info(f"向量存储: 添加 {len(chunks)} 个向量, 总计 {len(self._chunks)} 个")

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        余弦相似度搜索。

        Returns:
            [(chunk_index, cosine_score), ...] 列表，按分数降序
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []

        # L2 归一化后内积 = 余弦相似度
        query_norm = _l2_normalize(query_vec)
        docs_norm = _l2_normalize(self._vectors)

        # 计算余弦相似度
        scores = docs_norm @ query_norm[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def count(self) -> int:
        return len(self._chunks)

    def get_chunk(self, index: int) -> Chunk:
        """按索引获取文档块。"""
        return self._chunks[index]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2 归一化：向量除以其模长，使余弦相似度=内积。"""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms
