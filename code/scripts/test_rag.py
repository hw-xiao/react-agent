"""
RAG 知识库独立测试脚本：不需要启动模型服务，直接测试 RAG 管道。

测试内容:
  1. PDF 加载是否正确
  2. 分块质量和数量
  3. 检索准确性（查大众交通营业收入等）
  4. 元数据标注完整性

用法:
    .venv\\Scripts\\python.exe -m code.scripts.test_rag
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from code.config.settings import RAG_DATA_DIR, RAG_TEMP_DIR, RAG_CACHE_DIR
from code.rag.pipeline import RagPipeline
from code.utils.logger import get_logger

log = get_logger("test_rag")


def main():
    print("=" * 60)
    print("RAG 知识库独立测试")
    print("=" * 60)
    print(f"数据目录: {RAG_DATA_DIR}")
    print(f"临时目录: {RAG_TEMP_DIR}")
    print(f"缓存目录: {RAG_CACHE_DIR}")

    if not os.path.isdir(RAG_DATA_DIR):
        print(f"[错误] 数据目录不存在: {RAG_DATA_DIR}")
        sys.exit(1)

    # 初始化管道（带缓存和临时目录）
    pipeline = RagPipeline(
        data_dir=RAG_DATA_DIR,
        temp_dir=RAG_TEMP_DIR,
        cache_dir=RAG_CACHE_DIR,
    )

    # ── 测试1: 文档加载和分块 ──
    print("\n" + "=" * 60)
    print("[1] 文档加载和分块")
    print("=" * 60)
    pipeline._ensure_index()
    print(f"文档数: {pipeline.document_count}")
    for doc in pipeline.get_document_list():
        print(f"  - {doc['source_file']} | {doc['company']} | {doc['report_type']} | {doc['report_year']} | {doc['total_pages']}页")

    # ── 测试2: 检索准确性 ──
    test_queries = [
        "大众交通2026年半年报营业收入",
        "大众交通营业收入",
        "金健米业净利润",
        "远东股份2026年半年报",
        "大众交通总资产",
    ]

    print("\n" + "=" * 60)
    print("[2] 检索准确性测试")
    print("=" * 60)

    for query in test_queries:
        print(f"\n--- 查询: {query} ---")
        results = pipeline.query(query, top_k=3)

        if not results:
            print("  未找到结果")
            continue

        for i, result in enumerate(results, 1):
            print(f"  [{i}] 分数={result.score:.4f} | 公司={result.chunk.company} | 来源={result.chunk.source_file} | 第{result.chunk.page_num}页")
            print(f"      类型={result.chunk.chunk_type} | 字符数={result.chunk.char_count}")
            text_preview = result.chunk.text[:150].replace("\n", " ")
            print(f"      内容: {text_preview}...")

    # ── 测试3: 带引用标注的输出 ──
    print("\n" + "=" * 60)
    print("[3] 带引用标注的格式化输出")
    print("=" * 60)

    sample_query = "大众交通2026年半年报营业收入"
    print(f"\n查询: {sample_query}")
    print("-" * 40)
    output = pipeline.query_with_citations(sample_query, top_k=3)
    print(output)

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
