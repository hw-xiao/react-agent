"""
Embedding 接口层：将文本转向量。

当前实现: TfidfEmbedder（TF-IDF + jieba 分词，纯本地，无模型依赖）
未来扩展: BgeEmbedder（BGE-small-zh 离线模型）、DashScopeEmbedder（API 调用）

只需继承 BaseEmbedder 实现 embed() 方法，即可替换 embedding 策略。
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from code.utils.logger import get_logger

log = get_logger("embedder")


class BaseEmbedder(ABC):
    """Embedding 抽象基类。"""

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度。"""
        ...

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转为向量矩阵。

        Args:
            texts: 文本列表

        Returns:
            numpy 数组, shape=(len(texts), dim), dtype=float32
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """将单条查询文本转为向量，shape=(1, dim)。"""
        ...


class TfidfEmbedder(BaseEmbedder):
    """
    TF-IDF Embedder：使用 sklearn TfidfVectorizer + jieba 中文分词。

    优点: 无模型依赖，纯本地运行，速度快
    缺点: 语义理解不如深度模型，但对财报关键数字检索效果不错
    """

    def __init__(self, max_features: int = 5000):
        """
        Args:
            max_features: TF-IDF 最大词表大小
        """
        self._max_features = max_features
        self._vectorizer: TfidfVectorizer = None
        self._dim_val = max_features

    @property
    def dim(self) -> int:
        return self._dim_val

    def fit(self, texts: List[str]) -> "TfidfEmbedder":
        """
        训练 TF-IDF 模型（在文档库上拟合词表）。

        Args:
            texts: 所有文档块文本

        Returns:
            self（链式调用）
        """
        log.info(f"训练 TF-IDF 模型, max_features={self._max_features}")
        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            tokenizer=self._tokenize,
            token_pattern=None,
        )
        self._vectorizer.fit(texts)
        # 实际维度可能小于 max_features
        self._dim_val = len(self._vectorizer.vocabulary_)
        log.info(f"TF-IDF 训练完成, 实际词表大小={self._dim_val}")
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本列表转为 TF-IDF 向量矩阵。"""
        if self._vectorizer is None:
            raise RuntimeError("TF-IDF 模型未训练，请先调用 fit()")
        vectors = self._vectorizer.transform(texts)
        return vectors.toarray().astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """将查询文本转为向量。"""
        return self.embed([query])

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """jieba 中文分词。"""
        import jieba
        return [w.strip() for w in jieba.cut(text) if w.strip()]
