"""
RAG 管道编排器：串联 加载→分块→Embedding→索引→检索→格式化 全链路。

缓存策略:
  1. 首次构建：解析PDF → 分块 → 输出 chunks.json 到 temp → 训练 TF-IDF → 保存索引到 cachedb
  2. 后续启动：直接从 cachedb 加载 chunks + 向量矩阵，跳过 PDF 解析

中间数据:
  - {temp}/chunks.json: 分块后的完整文本和元数据
  - {cachedb}/chunks.pkl: 序列化的 chunks 列表
  - {cachedb}/vectors.npy: 向量矩阵
  - {cachedb}/tfidf_vocab.pkl: TF-IDF 词表
"""

import json
import os
import pickle
from datetime import datetime
from typing import List, Optional

import numpy as np

from code.rag.chunkers import chunk_documents
from code.rag.embeddings import BaseEmbedder, TfidfEmbedder
from code.rag.loaders.registry import load_directory
from code.rag.models import Chunk, Document, SearchResult
from code.rag.retriever import HybridRetriever
from code.rag.vectorstore import BaseVectorStore, InMemoryStore
from code.utils.logger import get_logger

log = get_logger("pipeline")

# 缓存版本：模型结构变更时递增，强制重建索引
CACHE_VERSION = "v2_company"


class RagPipeline:
    """
    RAG 管道：封装从文档加载到检索的完整流程。

    用法:
        pipeline = RagPipeline(data_dir="/path/to/docs")
        results = pipeline.query("大众交通2026年营业收入", top_k=5)
    """

    def __init__(
        self,
        data_dir: str,
        temp_dir: str = "",
        cache_dir: str = "",
        embedder: Optional[BaseEmbedder] = None,
        vectorstore: Optional[BaseVectorStore] = None,
        chunk_strategy: str = "semantic",
    ):
        """
        Args:
            data_dir:       知识库文档目录（PDF 等）
            temp_dir:       中间数据输出目录（chunks.json 等）
            cache_dir:      缓存目录（序列化的索引数据）
            embedder:       Embedding 模型（默认 TfidfEmbedder）
            vectorstore:    向量存储（默认 InMemoryStore）
            chunk_strategy: 分块策略 (semantic/fixed/hierarchical)
        """
        self.data_dir = data_dir
        self.temp_dir = temp_dir
        self.cache_dir = cache_dir
        self.embedder = embedder or TfidfEmbedder()
        self.vectorstore = vectorstore or InMemoryStore()
        self.chunk_strategy = chunk_strategy

        self._documents: List[Document] = []
        self._retriever: Optional[HybridRetriever] = None
        self._indexed = False

        # 确保目录存在
        for d in [self.temp_dir, self.cache_dir]:
            if d:
                os.makedirs(d, exist_ok=True)

    def _cache_paths(self) -> dict:
        """返回缓存文件的完整路径。"""
        return {
            "chunks": os.path.join(self.cache_dir, "chunks.pkl") if self.cache_dir else "",
            "vectors": os.path.join(self.cache_dir, "vectors.npy") if self.cache_dir else "",
            "vocab": os.path.join(self.cache_dir, "tfidf_vocab.pkl") if self.cache_dir else "",
            "doc_meta": os.path.join(self.cache_dir, "doc_meta.json") if self.cache_dir else "",
            "version": os.path.join(self.cache_dir, "cache_version.txt") if self.cache_dir else "",
        }

    def _try_load_cache(self) -> bool:
        """
        尝试从缓存加载索引。

        Returns:
            True 如果缓存加载成功，False 如果需要重建
        """
        if not self.cache_dir:
            return False

        paths = self._cache_paths()
        required = ["chunks", "vectors", "vocab"]
        if not all(os.path.exists(paths[k]) for k in required):
            log.info("缓存文件不完整，需要重建索引")
            return False

        # 检查缓存版本
        version_path = paths["version"]
        if not os.path.exists(version_path):
            log.info("缓存版本文件不存在，需要重建索引")
            return False
        with open(version_path, "r", encoding="utf-8") as f:
            cached_version = f.read().strip()
        if cached_version != CACHE_VERSION:
            log.info(f"缓存版本不匹配 (缓存={cached_version}, 当前={CACHE_VERSION})，需要重建索引")
            return False

        try:
            # 加载 chunks
            with open(paths["chunks"], "rb") as f:
                chunks: List[Chunk] = pickle.load(f)
            log.info(f"从缓存加载 {len(chunks)} 个 chunks")

            # 加载向量
            vectors = np.load(paths["vectors"])
            log.info(f"从缓存加载向量矩阵: shape={vectors.shape}")

            # 加载 TF-IDF 词表并恢复 embedder
            with open(paths["vocab"], "rb") as f:
                vocab_data = pickle.load(f)
            if isinstance(self.embedder, TfidfEmbedder):
                self.embedder._vectorizer = vocab_data["vectorizer"]
                self.embedder._dim_val = vocab_data["dim"]

            # 加载文档元数据
            if os.path.exists(paths["doc_meta"]):
                with open(paths["doc_meta"], "r", encoding="utf-8") as f:
                    self._documents = [Document(**d) for d in json.load(f)]

            # 重建向量存储和检索器
            self.vectorstore.add(chunks, vectors)
            self._retriever = HybridRetriever(
                embedder=self.embedder,
                vectorstore=self.vectorstore,
            )
            self._retriever.build_bm25(chunks)

            log.info(f"缓存加载完成，跳过 PDF 解析")
            return True

        except Exception as e:
            log.warning(f"缓存加载失败: {e}，将重建索引")
            return False

    def _save_cache(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        """将索引数据保存到缓存目录。"""
        if not self.cache_dir:
            return

        paths = self._cache_paths()
        try:
            # 保存 chunks
            with open(paths["chunks"], "wb") as f:
                pickle.dump(chunks, f)

            # 保存向量
            np.save(paths["vectors"], vectors)

            # 保存 TF-IDF 词表
            if isinstance(self.embedder, TfidfEmbedder):
                with open(paths["vocab"], "wb") as f:
                    pickle.dump({
                        "vectorizer": self.embedder._vectorizer,
                        "dim": self.embedder._dim_val,
                    }, f)

            # 保存文档元数据（只存摘要信息，不存 pages 内容）
            doc_meta = [d.to_dict() for d in self._documents]
            for d, orig in zip(doc_meta, self._documents):
                d["source_path"] = orig.source_path
                d["file_type"] = orig.file_type
            with open(paths["doc_meta"], "w", encoding="utf-8") as f:
                json.dump(doc_meta, f, ensure_ascii=False, indent=2)

            log.info(f"索引缓存已保存到 {self.cache_dir}")

            # 写入缓存版本
            with open(paths["version"], "w", encoding="utf-8") as f:
                f.write(CACHE_VERSION)

        except Exception as e:
            log.warning(f"缓存保存失败: {e}")

    def _save_temp_json(self, chunks: List[Chunk]) -> None:
        """将分块结果输出为 JSON 到 temp 目录，便于调试查看。"""
        if not self.temp_dir:
            return

        output_path = os.path.join(
            self.temp_dir, f"chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            chunks_data = []
            for c in chunks:
                chunks_data.append({
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "source_file": c.source_file,
                    "page_num": c.page_num,
                    "section_path": c.section_path,
                    "chunk_type": c.chunk_type,
                    "char_count": c.char_count,
                    "company": c.company,
                })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, ensure_ascii=False, indent=2)

            log.info(f"分块 JSON 已输出到 {output_path} ({len(chunks)} 块)")

            # 同时输出一个统计摘要
            stats_path = os.path.join(self.temp_dir, "chunk_stats.json")
            char_counts = [c.char_count for c in chunks]
            stats = {
                "total_chunks": len(chunks),
                "total_chars": sum(char_counts),
                "avg_chars": round(sum(char_counts) / max(len(char_counts), 1), 1),
                "min_chars": min(char_counts) if char_counts else 0,
                "max_chars": max(char_counts) if char_counts else 0,
                "chunk_types": {
                    t: sum(1 for c in chunks if c.chunk_type == t)
                    for t in set(c.chunk_type for c in chunks)
                },
                "by_document": {},
            }
            for c in chunks:
                if c.source_file not in stats["by_document"]:
                    stats["by_document"][c.source_file] = 0
                stats["by_document"][c.source_file] += 1

            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            log.info(f"分块统计已输出到 {stats_path}")

        except Exception as e:
            log.warning(f"JSON 输出失败: {e}")

    def _ensure_index(self) -> None:
        """
        延迟构建索引：优先从缓存加载，缓存不存在时才解析 PDF。

        流程:
          1. 尝试从 cachedb 加载（秒级）
          2. 失败则完整构建：解析PDF → 分块 → 输出JSON → 训练 → 保存缓存
        """
        if self._indexed:
            return

        # 优先尝试缓存
        if self._try_load_cache():
            self._indexed = True
            return

        log.info("=" * 60)
        log.info("RAG 索引构建开始（从头解析）")
        log.info(f"数据目录: {self.data_dir}")
        if self.cache_dir:
            log.info(f"缓存目录: {self.cache_dir}")
        if self.temp_dir:
            log.info(f"临时目录: {self.temp_dir}")
        log.info("=" * 60)

        # 步骤1: 加载文档
        self._documents = load_directory(self.data_dir)
        if not self._documents:
            log.warning("未加载到任何文档，RAG 管道为空")
            self._indexed = True
            return

        # 步骤2: 分块
        chunks = chunk_documents(self._documents, strategy=self.chunk_strategy)
        if not chunks:
            log.warning("分块结果为空")
            self._indexed = True
            return

        # 步骤2.5: 输出中间 JSON
        self._save_temp_json(chunks)

        # 步骤3: 训练 Embedding
        texts = [c.text for c in chunks]
        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.fit(texts)

        # 步骤4: 生成向量并加入存储
        vectors = self.embedder.embed(texts)
        self.vectorstore.add(chunks, vectors)

        # 步骤5: 构建检索器（含 BM25）
        self._retriever = HybridRetriever(
            embedder=self.embedder,
            vectorstore=self.vectorstore,
        )
        self._retriever.build_bm25(chunks)

        # 步骤6: 保存缓存
        self._save_cache(chunks, vectors)

        self._indexed = True
        log.info(f"RAG 索引构建完成: {len(self._documents)} 文档, {len(chunks)} 块, {self.vectorstore.count()} 向量")
        log.info("=" * 60)

    def query(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        查询知识库，返回检索结果（含元数据引用）。

        当查询中检测到公司名时，过滤掉不匹配的结果，只返回该公司的chunk。

        Args:
            query:  用户查询
            top_k:  返回前 K 个结果

        Returns:
            SearchResult 列表，每个结果携带 chunk 文本、来源文件、页码等元数据
        """
        self._ensure_index()

        if self._retriever is None:
            return []

        # 先获取更多候选，再按公司过滤
        raw_results = self._retriever.search(query, top_k=top_k * 3)

        # 检测查询中的公司名，过滤不匹配的结果
        detected_company = self._retriever._detect_company(query)
        if detected_company:
            filtered = [r for r in raw_results if r.chunk.company == detected_company]
            if filtered:
                log.info(f"公司过滤: 只保留 {detected_company} 的结果 ({len(filtered)}/{len(raw_results)})")
                results = filtered[:top_k]
            else:
                results = raw_results[:top_k]
        else:
            results = raw_results[:top_k]

        # 打印 top_k 检索结果详细内容
        if results:
            log.info(f"=== 检索结果 top_{top_k} (查询: '{query}') ===")
            for i, r in enumerate(results, 1):
                preview = r.chunk.text[:200].replace("\n", " ")
                log.info(f"  [{i}] 分数={r.score:.4f} | 公司={r.chunk.company} | 第{r.chunk.page_num}页 | 类型={r.chunk.chunk_type} | 字符={r.chunk.char_count}")
                log.info(f"      内容: {preview}...")
        else:
            log.info(f"检索结果为空 (查询: '{query}')")

        return results

    def query_with_citations(self, query: str, top_k: int = 5, max_chars_per_chunk: int = 0) -> str:
        """
        查询并返回带引用标注的格式化文本。

        格式示例:
            以下是从知识库中检索到的相关内容（共 N 条）:

            [1] 来源: 大众交通2026年半年报.pdf, 第177页, 类型: table
            内容:
            【大众交通】营业收入和营业成本...
            营业收入 1,234,567,890 元

            [2] 来源: ..., 第X页, 类型: text
            内容:
            ...

        Args:
            query:  用户查询
            top_k:  返回前 K 个结果
            max_chars_per_chunk: 每条结果最大字符数，0表示不截断

        Returns:
            带引用标注的检索结果文本
        """
        results = self.query(query, top_k=top_k)

        if not results:
            return "未找到相关数据。"

        lines = [f"以下是从知识库中检索到的相关内容（共 {len(results)} 条）:", ""]
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            company_str = f", 公司: {chunk.company}" if chunk.company else ""
            lines.append(f"[{i}] 来源: {chunk.source_file}, 第{chunk.page_num}页{company_str}, 类型: {chunk.chunk_type}, 相关度: {result.score:.4f}")
            lines.append("内容:")
            text = chunk.text
            if max_chars_per_chunk > 0 and len(text) > max_chars_per_chunk:
                text = text[:max_chars_per_chunk] + "..."
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    @property
    def document_count(self) -> int:
        """已加载的文档数量。"""
        return len(self._documents)

    @property
    def is_indexed(self) -> bool:
        """索引是否已构建。"""
        return self._indexed

    def get_document_list(self) -> List[dict]:
        """返回已加载文档的元数据列表。"""
        return [doc.to_dict() for doc in self._documents]

    def clear_cache(self) -> None:
        """清除缓存文件，下次启动会重建索引。"""
        if not self.cache_dir:
            return
        paths = self._cache_paths()
        for path in paths.values():
            if path and os.path.exists(path):
                os.remove(path)
                log.info(f"已删除缓存: {path}")
