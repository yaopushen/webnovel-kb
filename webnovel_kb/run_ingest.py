import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webnovel_kb.config import (
    LLM_API_KEY, LLM_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_RERANK_MODEL, LLM_EMBEDDING_DIMENSIONS,
)

os.environ.setdefault("LLM_API_KEY", LLM_API_KEY)
os.environ.setdefault("LLM_BASE_URL", LLM_BASE_URL)
os.environ.setdefault("LLM_EMBEDDING_MODEL", LLM_EMBEDDING_MODEL)
os.environ.setdefault("LLM_RERANK_MODEL", LLM_RERANK_MODEL)
os.environ.setdefault("LLM_EMBEDDING_DIMENSIONS", str(LLM_EMBEDDING_DIMENSIONS))

from webnovel_kb.server import kb

NOVELS_DIR = r"D:\文档\小说库"
CATALOG_FILE = r"D:\文档\小说库\catalog.json"

catalog = {}
if Path(CATALOG_FILE).exists():
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        catalog = json.load(f)

novels_path = Path(NOVELS_DIR)
txt_files = sorted(novels_path.glob("*.txt"))
print(f"Found {len(txt_files)} novels to ingest")
print(f"Embedding: (model={os.environ['LLM_EMBEDDING_MODEL']}, dim={os.environ['LLM_EMBEDDING_DIMENSIONS']})")
print(f"Rerank: (model={os.environ['LLM_RERANK_MODEL']})")
print()

total_start = time.time()
success_count = 0
fail_count = 0

for i, f in enumerate(txt_files, 1):
    stem = f.stem
    info = catalog.get(stem, {})
    title = info.get("title", stem)
    author = info.get("author", "unknown")
    genre = info.get("genre", "unknown")
    file_size_mb = f.stat().st_size / (1024 * 1024)
    print(f"[{i}/{len(txt_files)}] {title} ({author}) [{genre}] {file_size_mb:.1f}MB")
    start = time.time()
    try:
        result = kb.ingest_novel(str(f), title, author, genre)
        elapsed = time.time() - start
        if "error" in result:
            print(f"  FAIL: {result['error']}")
            fail_count += 1
        else:
            print(f"  OK: {result['chunk_count']} chunks, {elapsed:.1f}s")
            success_count += 1
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e} ({elapsed:.1f}s)")
        fail_count += 1

total_elapsed = time.time() - total_start
stats = kb.get_stats()
print(f"\n{'='*60}")
print(f"Done! {success_count} success, {fail_count} failed")
print(f"Total time: {total_elapsed:.1f}s")
print(f"Total novels: {stats['total_novels']}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Tantivy ready: {stats.get('tantivy_ready', stats.get('bm25_ready', False))}")
print(f"Reranker: {'Enabled' if kb.reranker else 'None'}")