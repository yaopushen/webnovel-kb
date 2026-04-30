"""
批量导入小说的命令行工具。

使用方法:
    python -m webnovel_kb.run_ingest --novels-dir /path/to/novels [--catalog catalog.json]

环境变量:
    LLM_API_KEY: LLM API密钥
    LLM_BASE_URL: Embedding API地址
    LLM_EMBEDDING_MODEL: 嵌入模型名称
    LLM_EMBEDDING_DIMENSIONS: 向量维度
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webnovel_kb.config import (
    LLM_API_KEY, LLM_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_RERANK_MODEL, LLM_EMBEDDING_DIMENSIONS,
)

os.environ.setdefault("LLM_API_KEY", LLM_API_KEY or "")
os.environ.setdefault("LLM_BASE_URL", LLM_BASE_URL or "")
os.environ.setdefault("LLM_EMBEDDING_MODEL", LLM_EMBEDDING_MODEL or "")
os.environ.setdefault("LLM_RERANK_MODEL", LLM_RERANK_MODEL or "")
os.environ.setdefault("LLM_EMBEDDING_DIMENSIONS", str(LLM_EMBEDDING_DIMENSIONS))

from webnovel_kb.server import kb


def main():
    parser = argparse.ArgumentParser(description="批量导入小说到知识库")
    parser.add_argument("--novels-dir", required=True, help="小说txt文件所在目录")
    parser.add_argument("--catalog", default="", help="目录信息JSON文件")
    args = parser.parse_args()

    novels_path = Path(args.novels_dir)
    if not novels_path.exists():
        print(f"目录不存在: {args.novels_dir}")
        return 1

    catalog = {}
    if args.catalog and Path(args.catalog).exists():
        with open(args.catalog, "r", encoding="utf-8") as f:
            catalog = json.load(f)

    txt_files = sorted(novels_path.glob("*.txt"))
    if not txt_files:
        print(f"未找到txt文件: {args.novels_dir}")
        return 1

    print(f"找到 {len(txt_files)} 个小说文件")
    print(f"嵌入模型: {LLM_EMBEDDING_MODEL} (维度={LLM_EMBEDDING_DIMENSIONS})")
    print()

    total_start = time.time()
    success_count = 0
    fail_count = 0

    for i, f in enumerate(txt_files, 1):
        stem = f.stem
        info = catalog.get(stem, {})
        title = info.get("title", stem)
        author = info.get("author", "未知")
        genre = info.get("genre", "未分类")
        file_size_mb = f.stat().st_size / (1024 * 1024)
        print(f"[{i}/{len(txt_files)}] {title} ({author}) [{genre}] {file_size_mb:.1f}MB")
        start = time.time()
        try:
            result = kb.ingest_novel(str(f), title, author, genre)
            elapsed = time.time() - start
            if "error" in result:
                print(f"  ❌ 失败: {result['error']}")
                fail_count += 1
            else:
                print(f"  ✅ 成功: {result['chunk_count']} 个分块, {elapsed:.1f}s")
                success_count += 1
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ❌ 错误: {e} ({elapsed:.1f}s)")
            fail_count += 1

    total_elapsed = time.time() - total_start
    stats = kb.get_stats()
    print(f"\n{'='*60}")
    print(f"导入完成! 成功 {success_count} 本, 失败 {fail_count} 本")
    print(f"总耗时: {total_elapsed:.1f}s")
    print(f"总计小说: {stats['total_novels']} 本")
    print(f"总计分块: {stats['total_chunks']} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
