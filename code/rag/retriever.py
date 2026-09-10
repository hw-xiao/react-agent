"""
混合检索器：向量检索（语义）+ BM25（精确匹配）→ RRF 融合 + 公司名感知加权。

流程:
  1. 向量检索: TF-IDF 余弦相似度，找语义相关块
  2. BM25 检索: jieba 分词 + BM25Okapi，精确匹配数字/专有名词
  3. RRF 融合: Reciprocal Rank Fusion 合并两路结果
  4. 公司名感知加权: 查询中提到的公司名，对应chunk分数 ×1.5
  5. 返回 top_k 个 SearchResult
"""

from typing import List, Optional, Set

import numpy as np
from rank_bm25 import BM25Okapi

from code.rag.embeddings import BaseEmbedder
from code.rag.models import Chunk, SearchResult
from code.rag.vectorstore import BaseVectorStore
from code.utils.logger import get_logger

log = get_logger("retriever")

# 公司名感知加权因子
COMPANY_BOOST = 1.5


class HybridRetriever:
    """
    混合检索器：向量 + BM25 双路检索 + RRF 融合 + 公司名感知加权。

    Args:
        embedder:     Embedding 模型实例
        vectorstore:  向量存储实例
        bm25_k:       RRF 融合参数 k（越大融合越平滑）
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vectorstore: BaseVectorStore,
        bm25_k: int = 60,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.bm25_k = bm25_k
        self._bm25: BM25Okapi = None
        self._bm25_texts: List[str] = []
        self._chunks: List[Chunk] = []
        self._companies: Set[str] = set()

    def build_bm25(self, chunks: List[Chunk]) -> None:
        """
        构建 BM25 索引。

        Args:
            chunks: 所有文档块
        """
        import jieba
        self._chunks = chunks
        self._bm25_texts = [c.text for c in chunks]
        tokenized = [list(jieba.cut(t)) for t in self._bm25_texts]
        self._bm25 = BM25Okapi(tokenized)
        # 收集所有公司名，用于查询时匹配
        self._companies = {c.company for c in chunks if c.company}
        log.info(f"BM25 索引构建完成: {len(tokenized)} 篇文档, 公司: {self._companies}")

    def _detect_company(self, query: str) -> Optional[str]:
        """从查询中检测公司名。"""
        for company in self._companies:
            if company and company in query:
                return company
        return None

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        混合检索：向量 + BM25 → RRF 融合 → 公司名加权。

        Args:
            query:  用户查询文本
            top_k:  返回前 K 个结果

        Returns:
            SearchResult 列表，按融合分数降序
        """
        if self.vectorstore.count() == 0:
            log.warning("向量存储为空，无法检索")
            return []

        # ── 检测查询中的公司名 ──
        detected_company = self._detect_company(query)
        if detected_company:
            log.info(f"检测到公司名: {detected_company}，将提升该公司结果权重")

        # ── 路径1: 向量检索 ──
        query_vec = self.embedder.embed_query(query)
        vec_results = self.vectorstore.search(query_vec, top_k=top_k * 4)
        vec_rank = {idx: rank for rank, (idx, _) in enumerate(vec_results)}

        # ── 路径2: BM25 检索 ──
        bm25_results = self._bm25_search(query, top_k=top_k * 4)
        bm25_rank = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}

        # ── 打印双路检索 top 结果 ──
        log.info(f"=== 向量检索 top5 (查询='{query[:50]}') ===")
        for idx, score in vec_results[:5]:
            chunk = self.vectorstore.get_chunk(idx)
            preview = chunk.text[:100].replace("\n", " ")
            log.info(f"  向量[{idx}] score={score:.4f} | {chunk.company} p{chunk.page_num} | {preview}...")
        log.info(f"=== BM25检索 top5 (查询='{query[:50]}') ===")
        for idx, score in bm25_results[:5]:
            chunk = self.vectorstore.get_chunk(idx)
            preview = chunk.text[:100].replace("\n", " ")
            log.info(f"  BM25[{idx}] score={score:.4f} | {chunk.company} p{chunk.page_num} | {preview}...")

        # ── RRF 融合 ──
        all_indices = set(vec_rank.keys()) | set(bm25_rank.keys())
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in vec_rank:
                score += 1.0 / (self.bm25_k + vec_rank[idx] + 1)
            if idx in bm25_rank:
                score += 1.0 / (self.bm25_k + bm25_rank[idx] + 1)
            rrf_scores[idx] = score

        # ── 公司名感知加权 ──
        if detected_company:
            for idx in rrf_scores:
                chunk = self.vectorstore.get_chunk(idx)
                if chunk.company == detected_company:
                    rrf_scores[idx] *= COMPANY_BOOST
            log.info(f"已对 {detected_company} 的结果加权 ×{COMPANY_BOOST}")

        # 排序取 top_k
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]

        results = []
        for idx, score in ranked:
            chunk = self.vectorstore.get_chunk(idx)
            results.append(SearchResult(chunk=chunk, score=score, retrieval_method="hybrid"))

        # ── 打印最终融合结果 ──
        log.info(f"=== RRF融合+公司加权 top{top_k} ===")
        for i, (idx, score) in enumerate(ranked, 1):
            chunk = self.vectorstore.get_chunk(idx)
            preview = chunk.text[:120].replace("\n", " ")
            log.info(f"  [{i}] score={score:.4f} | {chunk.company} p{chunk.page_num} | {chunk.chunk_type} | {preview}...")

        log.info(f"混合检索完成: 查询='{query[:50]}' | 向量候选={len(vec_results)} BM25候选={len(bm25_results)} | 返回 {len(results)} 个结果")
        return results

    def _bm25_search(self, query: str, top_k: int = 10) -> List[tuple]:
        """
        BM25 关键词检索。

        Returns:
            [(chunk_index, bm25_score), ...] 列表
        """
        if self._bm25 is None:
            return []

        import jieba
        query_tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
