#!/usr/bin/env python3
"""
Build optimized search index (Tantivy BM25) from existing ChromaDB data.
Run this once after deploying the new search engines.
v1.9: FAISS removed (GPU incompatible), rank_bm25 removed (memory hog).
"""
import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webnovel_kb.search_engines import TANTIVY_AVAILABLE, TantivyBM25, tokenize

os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("LLM_BASE_URL", "")
os.environ.setdefault("LLM_CHAT_BASE_URL", "")


def build_tantivy_index(data_dir: Path, collection) -> int:
    if not TANTIVY_AVAILABLE:
        print("ERROR: Tantivy not available. Install it first:")
        print("  pip install tantivy")
        return 0
    
    tantivy_dir = data_dir / "tantivy_index"
    
    if tantivy_dir.exists() and any(tantivy_dir.iterdir()):
        idx = TantivyBM25(tantivy_dir)
        if idx.doc_count > 0:
            print(f"Tantivy index already exists with {idx.doc_count} documents")
            return idx.doc_count
    
    print("Building Tantivy index...")
    total = collection.count()
    if total == 0:
        print("No documents in collection")
        return 0
    
    idx = TantivyBM25(tantivy_dir)
    documents = []
    batch_size = 500
    
    for offset in range(0, total, batch_size):
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=batch_size,
            offset=offset
        )
        if batch and batch.get("ids"):
            for i, cid in enumerate(batch["ids"]):
                documents.append({
                    "chunk_id": cid,
                    "text": batch["documents"][i] if batch.get("documents") else "",
                    "metadata": batch["metadatas"][i] if batch.get("metadatas") else {}
                })
        
        if offset % 5000 == 0:
            print(f"  Tantivy progress: {len(documents)}/{total}")
    
    if documents:
        idx.build_index(documents)
        print(f"  Tantivy built: {idx.doc_count} documents")
    
    return idx.doc_count


def main():
    parser = argparse.ArgumentParser(description="Build Tantivy BM25 index")
    parser.add_argument("--data-dir", type=str, default="./webnovel_data",
                        help="Data directory path")
    args = parser.parse_args()
    
    print(f"Tantivy available: {TANTIVY_AVAILABLE}")
    
    if not TANTIVY_AVAILABLE:
        print("ERROR: Tantivy not available. Install it first:")
        print("  pip install tantivy")
        return 1
    
    data_dir = Path(args.data_dir)
    print(f"Data directory: {data_dir}")
    
    import chromadb
    from webnovel_kb.api.clients import create_embedding_function
    
    embedding_fn = create_embedding_function(cache_path=str(data_dir / "embeddings_cache.pkl"))
    client = chromadb.PersistentClient(path=str(data_dir / "chroma_db"))
    collection = client.get_or_create_collection(
        name="webnovel_chunks",
        embedding_function=embedding_fn
    )
    
    total = collection.count()
    print(f"Total chunks in ChromaDB: {total}")
    
    if total == 0:
        print("ERROR: No chunks found in ChromaDB")
        return 1
    
    t0 = time.time()
    tantivy_count = build_tantivy_index(data_dir, collection)
    elapsed = time.time() - t0
    
    print(f"\n{'='*50}")
    print(f"Build complete in {elapsed:.1f}s")
    print(f"{'='*50}")
    print(f"Tantivy documents: {tantivy_count}")
    
    tantivy_dir = data_dir / "tantivy_index"
    if tantivy_dir.exists():
        total_size = sum(f.stat().st_size for f in tantivy_dir.rglob("*") if f.is_file())
        print(f"Tantivy index files: {total_size / 1024 / 1024:.1f} MB")
    
    bm25_path = data_dir / "bm25_index.pkl"
    if bm25_path.exists():
        bm25_size = bm25_path.stat().st_size / 1024 / 1024
        print(f"\nOld rank_bm25 cache: {bm25_size:.1f} MB (can be deleted after verification)")
    
    faiss_path = data_dir / "faiss_index.faiss"
    if faiss_path.exists():
        faiss_size = faiss_path.stat().st_size / 1024 / 1024
        print(f"Old FAISS index: {faiss_size:.1f} MB (can be deleted after verification)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
