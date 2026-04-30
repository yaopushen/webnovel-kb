import json
import math
import hashlib
import re
import time
import logging
import shutil
import pickle
import threading
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import asdict
from datetime import datetime

import chromadb
import jieba
import networkx as nx
from fastmcp import FastMCP

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from webnovel_kb.data_models import (
    NovelMeta, StyleProfile, PlotPattern,
    Entity, Relationship, WritingTemplate,
)
from webnovel_kb.api_clients import (
    RemoteEmbeddingFunction, RemoteReranker, RemoteChatClient,
    LocalEmbeddingFunction, _create_embedding_function,
)
from webnovel_kb.config import (
    DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    LLM_API_KEY, LLM_BASE_URL, LLM_CHAT_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_RERANK_MODEL,
    LLM_EMBEDDING_DIMENSIONS, LLM_CHAT_MODEL
)
from webnovel_kb.search_engines import (
    TantivyBM25, FAISSVectorStore, HybridSearchEngine,
    TANTIVY_AVAILABLE, FAISS_AVAILABLE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webnovel-kb")


def _normalize_title(title: str) -> str:
    """标准化标题：去除常见标点和空格，转小写，用于模糊匹配"""
    return re.sub(r'[？?！!。，,.、""\'\'【】\[\]()（）《》<>·\s]', '', title).lower().strip()

mcp = FastMCP(
    "webnovel-kb",
    instructions="网文知识库MCP服务器 - 从顶级网文中提取写作风格、情节模式、人物关系、世界观等深层创作知识，支持语义检索、知识图谱查询和风格分析"
)


class WebNovelKnowledgeBase:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = _create_embedding_function(cache_path=str(data_dir / "embeddings_cache.pkl"))
        self.reranker = RemoteReranker() if LLM_API_KEY else None
        self.chat = RemoteChatClient() if LLM_API_KEY else None
        self.chroma_client = chromadb.PersistentClient(path=str(data_dir / "chroma_db"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="webnovel_chunks",
            embedding_function=self.embedding_fn,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 64,
                "hnsw:construction_ef": 256,
                "hnsw:search_ef": 128
            }
        )
        self.patterns_collection = self.chroma_client.get_or_create_collection(
            name="plot_patterns",
            embedding_function=self.embedding_fn,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 32,
                "hnsw:construction_ef": 128,
                "hnsw:search_ef": 64
            }
        )
        self.entities_collection = self.chroma_client.get_or_create_collection(
            name="entities",
            embedding_function=self.embedding_fn,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "hnsw:construction_ef": 64,
                "hnsw:search_ef": 32
            }
        )
        self.graph = nx.DiGraph()
        self.novels: dict[str, NovelMeta] = {}
        self.style_profiles: dict[str, StyleProfile] = {}
        self.plot_patterns: list[PlotPattern] = []
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.writing_templates: list[WritingTemplate] = []
        
        # Legacy BM25 (fallback)
        self.bm25_corpus: list[list[str]] = []
        self.bm25_metadata: list[dict] = []
        self.bm25: Optional[Any] = None
        self._bm25_ready = False
        
        # New optimized search engines
        self._tantivy_index: Optional[TantivyBM25] = None
        self._faiss_store: Optional[FAISSVectorStore] = None
        self._hybrid_engine: Optional[HybridSearchEngine] = None
        self._use_optimized_search = TANTIVY_AVAILABLE and FAISS_AVAILABLE
        
        self._background_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_lock = threading.Lock()
        
        self._load_state()
        self._ensure_default_templates()
        self._index_plot_patterns()
        
        # Initialize optimized search engines
        if self._use_optimized_search:
            self._init_optimized_search()
        else:
            logger.warning("Optimized search not available, using legacy BM25")
            self._preload_bm25_background()
    
    def _init_optimized_search(self):
        try:
            tantivy_dir = self.data_dir / "tantivy_index"
            self._tantivy_index = TantivyBM25(tantivy_dir)
            
            faiss_path = self.data_dir / "faiss_index.faiss"
            self._faiss_store = FAISSVectorStore(
                faiss_path, 
                dimensions=LLM_EMBEDDING_DIMENSIONS
            )
            
            faiss_loaded = False
            if faiss_path.exists():
                faiss_loaded = self._faiss_store.load_index()
                if faiss_loaded:
                    logger.info(f"FAISS index loaded: {self._faiss_store.count} vectors")
            
            tantivy_dir = self.data_dir / "tantivy_index"
            tantivy_exists = tantivy_dir.exists() and any(tantivy_dir.iterdir())
            
            self._hybrid_engine = HybridSearchEngine(
                self._tantivy_index, 
                self._faiss_store,
                cache_ttl=60
            )
            
            if not faiss_loaded or not tantivy_exists:
                logger.info("Building optimized indexes...")
                threading.Thread(target=self._build_optimized_indexes, daemon=True).start()
            
            logger.info("Optimized search engines initialized")
        except Exception as e:
            logger.error(f"Failed to init optimized search: {e}")
            self._use_optimized_search = False
            self._preload_bm25_background()
    
    def _build_optimized_indexes(self):
        try:
            self._build_tantivy_index()
            self._build_faiss_index()
        except Exception as e:
            logger.error(f"Failed to build optimized indexes: {e}")

    def _index_plot_patterns(self):
        if not self.plot_patterns:
            return
        try:
            existing_ids = set()
            offset = 0
            batch = 1000
            while True:
                result = self.patterns_collection.get(limit=batch, offset=offset, include=[])
                ids_chunk = result.get("ids", []) if result else []
                if not ids_chunk:
                    break
                existing_ids.update(ids_chunk)
                if len(ids_chunk) < batch:
                    break
                offset += batch

            ids = []
            documents = []
            metadatas = []
            for i, p in enumerate(self.plot_patterns):
                pattern_id = f"pattern_{i}"
                if pattern_id in existing_ids:
                    continue
                text = f"{p.pattern_type}: {p.description}"
                if p.pattern_text:
                    text += f"\n{p.pattern_text}"
                ids.append(pattern_id)
                documents.append(text)
                metadatas.append({
                    "pattern_type": p.pattern_type or "",
                    "source_novel": p.source_novel or "",
                    "effectiveness": p.effectiveness or "",
                    "index": i
                })
            if ids:
                batch_size = 100
                for j in range(0, len(ids), batch_size):
                    self.patterns_collection.upsert(
                        ids=ids[j:j+batch_size],
                        documents=documents[j:j+batch_size],
                        metadatas=metadatas[j:j+batch_size]
                    )
                logger.info(f"Indexed {len(ids)} plot patterns into vector store")
        except Exception as e:
            logger.warning(f"Failed to index plot patterns: {e}")
        self._index_entities()

    def _index_entities(self):
        if not self.entities:
            return
        try:
            existing_ids = set()
            offset = 0
            batch = 1000
            while True:
                result = self.entities_collection.get(limit=batch, offset=offset, include=[])
                ids_chunk = result.get("ids", []) if result else []
                if not ids_chunk:
                    break
                existing_ids.update(ids_chunk)
                if len(ids_chunk) < batch:
                    break
                offset += batch

            ids = []
            documents = []
            metadatas = []
            for eid, e in self.entities.items():
                entity_id = f"entity_{eid}"
                if entity_id in existing_ids:
                    continue
                text = f"{e.entity_type}: {e.name}"
                if e.description:
                    text += f" - {e.description}"
                ids.append(entity_id)
                documents.append(text)
                metadatas.append({
                    "entity_type": e.entity_type or "",
                    "name": e.name or "",
                    "source_novel": e.source_novel or "",
                    "entity_id": eid
                })
            if ids:
                batch_size = 100
                for j in range(0, len(ids), batch_size):
                    self.entities_collection.upsert(
                        ids=ids[j:j+batch_size],
                        documents=documents[j:j+batch_size],
                        metadatas=metadatas[j:j+batch_size]
                    )
                logger.info(f"Indexed {len(ids)} entities into vector store")
        except Exception as e:
            logger.warning(f"Failed to index entities: {e}")

    def _load_state(self):
        state_file = self.data_dir / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            for k, v in state.get("novels", {}).items():
                self.novels[k] = NovelMeta(**v)
            graph_file = self.data_dir / "knowledge_graph.json"
            if graph_file.exists():
                self._load_graph(graph_file)
            patterns_file = self.data_dir / "plot_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, "r", encoding="utf-8") as f:
                    for p in json.load(f):
                        self.plot_patterns.append(PlotPattern(**p))
            styles_file = self.data_dir / "style_profiles.json"
            if styles_file.exists():
                with open(styles_file, "r", encoding="utf-8") as f:
                    for k, v in json.load(f).items():
                        self.style_profiles[k] = StyleProfile(**v)
            entities_file = self.data_dir / "entities.json"
            if entities_file.exists():
                with open(entities_file, "r", encoding="utf-8") as f:
                    for k, v in json.load(f).items():
                        self.entities[k] = Entity(**v)
            rels_file = self.data_dir / "relationships.json"
            if rels_file.exists():
                with open(rels_file, "r", encoding="utf-8") as f:
                    for r in json.load(f):
                        self.relationships.append(Relationship(**r))
            templates_file = self.data_dir / "writing_templates.json"
            if templates_file.exists():
                with open(templates_file, "r", encoding="utf-8") as f:
                    for t in json.load(f):
                        self.writing_templates.append(WritingTemplate(**t))
            self._rebuild_bm25()

    def _ensure_default_templates(self):
        """Ensure basic writing templates exist for common web novel scenarios."""
        if len(self.writing_templates) >= 8:
            return  # already sufficient
        defaults = [
            ("场景模板", "金手指激活", "1.危机时刻→2.触发条件达成→3.金手指显现→4.初次体验→5.认知震撼",
             ["危机时刻", "触发条件", "金手指显现", "初次体验", "认知震撼"]),
            ("场景模板", "越级反杀", "1.遭遇强敌→2.实力差距展现→3.利用金手指/智谋→4.惊天逆转→5.战后收获",
             ["遭遇强敌", "实力差距", "巧用资源", "惊天逆转", "战后收获"]),
            ("场景模板", "扮猪吃虎打脸", "1.低调入场→2.众人轻视→3.冲突升级→4.展现实力→5.众人震惊→6.身份揭露",
             ["低调入场", "众人轻视", "冲突升级", "展现实力", "众人震惊", "身份揭露"]),
            ("场景模板", "秘境探索", "1.秘境入口→2.环境描写→3.第一波危险→4.发现宝物→5.强敌或陷阱→6.逃出生天",
             ["秘境入口", "环境描写", "遭遇危险", "发现宝物", "强敌陷阱", "逃出生天"]),
            ("场景模板", "情感线突破", "1.日常互动→2.触发事件→3.情感波动→4.试探或误会→5.真情流露→6.关系升级",
             ["日常互动", "触发事件", "情感波动", "试探误会", "真情流露", "关系升级"]),
            ("场景模板", "世界观揭示", "1.日常场景→2.异常现象→3.信息碎片→4.真相揭示→5.格局重塑",
             ["日常场景", "异常现象", "信息碎片", "真相揭示", "格局重塑"]),
            ("节奏模板", "章末悬念钩子", "本章收束→抛出新信息→制造紧迫感→留下疑问→切视角或断章",
             ["本章收束", "新信息", "紧迫感", "留下疑问", "切断点"]),
            ("场景模板", "修炼突破", "1.瓶颈感知→2.资源准备→3.突破过程→4.异象引发关注→5.实力大涨→6.新的瓶颈暗示",
             ["瓶颈感知", "资源准备", "突破过程", "异象关注", "实力大涨", "新瓶颈暗示"]),
        ]
        added = False
        for template_type, scene_type, structure, beats_str in defaults:
            # Avoid duplicates
            existing = [t for t in self.writing_templates
                        if t.template_type == template_type and t.scene_type == scene_type]
            if not existing:
                self.writing_templates.append(WritingTemplate(
                    template_type=template_type,
                    scene_type=scene_type,
                    structure=structure,
                    key_beats=beats_str,
                    source_novel="通用模板"
                ))
                added = True
        if added:
            self._save_state()

    def _save_state(self):
        backup_dir = self.data_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for fname in ["state.json", "knowledge_graph.json", "plot_patterns.json",
                       "style_profiles.json", "entities.json", "relationships.json",
                       "writing_templates.json"]:
            src = self.data_dir / fname
            if src.exists():
                shutil.copy2(src, backup_dir / f"{fname}.{timestamp}.bak")

        state = {
            "novels": {k: asdict(v) for k, v in self.novels.items()},
        }
        with open(self.data_dir / "state.json", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        self._save_graph()
        with open(self.data_dir / "plot_patterns.json", "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in self.plot_patterns], f, ensure_ascii=False, indent=2)
        with open(self.data_dir / "style_profiles.json", "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.style_profiles.items()}, f, ensure_ascii=False, indent=2)
        with open(self.data_dir / "entities.json", "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.entities.items()}, f, ensure_ascii=False, indent=2)
        with open(self.data_dir / "relationships.json", "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.relationships], f, ensure_ascii=False, indent=2)
        with open(self.data_dir / "writing_templates.json", "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self.writing_templates], f, ensure_ascii=False, indent=2)

    def _save_graph(self):
        data = nx.node_link_data(self.graph)
        with open(self.data_dir / "knowledge_graph.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_graph(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True)

    def _chunk_text(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[tuple[str, str]]:
        chapter_pattern = re.compile(
            r'(?:^|\n)(?:第[零一二三四五六七八九十百千万\d]+[章节回卷]'
            r'|Chapter\s+\d+'
            r'|chapter\s+\d+'
            r'|\d{1,5}[\.、\s])',
            re.IGNORECASE
        )
        chapter_positions = [(m.start(), m.group().strip()) for m in chapter_pattern.finditer(text)]

        if chapter_positions and len(chapter_positions) > 3:
            chunks = []
            current_chapter = ""
            for i, (pos, title) in enumerate(chapter_positions):
                current_chapter = title
                end_pos = chapter_positions[i + 1][0] if i + 1 < len(chapter_positions) else len(text)
                chapter_text = text[pos:end_pos]
                if len(chapter_text) <= chunk_size * 1.5:
                    chunk = chapter_text.strip()
                    if chunk:
                        chunks.append((chunk, current_chapter))
                else:
                    sub_chunks = self._chunk_text_simple(chapter_text, chunk_size, overlap)
                    for sc in sub_chunks:
                        chunks.append((sc, current_chapter))
            return chunks
        return [(c, "") for c in self._chunk_text_simple(text, chunk_size, overlap)]

    def _chunk_text_simple(self, text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                for sep in ["。", "！", "？", "\n\n", "\n", "；"]:
                    pos = text.rfind(sep, start + chunk_size // 2, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(text) else end
        return chunks

    def _tokenize(self, text: str) -> list[str]:
        tokens = list(jieba.cut(text))
        stopwords = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
                     "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
                     "自己", "这", "他", "她", "它", "们", "那", "被", "从", "把", "让", "对", "而",
                     "但", "又", "么", "吗", "呢", "吧", "啊", "哦", "嗯", "呀", "啦", "哈"}
        return [t.strip() for t in tokens if t.strip() and t.strip() not in stopwords]

    def _rebuild_bm25(self):
        bm25_cache = self.data_dir / "bm25_index.pkl"
        if bm25_cache.exists():
            try:
                with open(bm25_cache, "rb") as f:
                    cache = pickle.load(f)
                self.bm25_corpus = cache["corpus"]
                self.bm25_metadata = cache["metadata"]
                logger.info(f"BM25 corpus loaded: {len(self.bm25_corpus)} documents (index deferred)")
                self._bm25_ready = False
                return
            except Exception as e:
                logger.warning(f"BM25 cache load failed: {e}, will rebuild on demand")
        self._bm25_ready = False

    def _ensure_bm25(self):
        if self._bm25_ready and self.bm25 is not None:
            return
        if not BM25_AVAILABLE:
            logger.warning("rank_bm25 not available, BM25 search disabled")
            return
        if self.bm25_corpus:
            logger.info("Building BM25 index on demand...")
            self.bm25 = BM25Okapi(self.bm25_corpus)
            self._bm25_ready = True
            logger.info(f"BM25 index ready: {len(self.bm25_corpus)} documents")
        else:
            self._rebuild_bm25_from_chroma()

    def _preload_bm25_background(self):
        if not BM25_AVAILABLE:
            return
        if self.bm25_corpus and not self._bm25_ready:
            def _build():
                try:
                    logger.info("Pre-building BM25 index in background...")
                    self.bm25 = BM25Okapi(self.bm25_corpus)
                    self._bm25_ready = True
                    logger.info(f"BM25 index pre-built: {len(self.bm25_corpus)} documents")
                except Exception as e:
                    logger.warning(f"BM25 background build failed: {e}")
            t = threading.Thread(target=_build, daemon=True)
            t.start()

    def _save_bm25_cache(self):
        bm25_cache = self.data_dir / "bm25_index.pkl"
        try:
            with open(bm25_cache.with_suffix('.pkl.tmp'), "wb") as f:
                pickle.dump({
                    "corpus": self.bm25_corpus,
                    "metadata": self.bm25_metadata,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            bm25_cache.with_suffix('.pkl.tmp').replace(bm25_cache)
            logger.info(f"BM25 cache saved: {len(self.bm25_corpus)} documents")
        except Exception as e:
            logger.error(f"BM25 cache save failed: {e}")

    def _rebuild_bm25_from_chroma(self):
        try:
            total = self.collection.count()
            if total == 0:
                logger.warning("ChromaDB is empty, BM25 not available")
                return
            self.bm25_corpus = []
            self.bm25_metadata = []
            batch_size = 500
            for offset in range(0, total, batch_size):
                limit = min(batch_size, total - offset)
                batch_ids = self.collection.get(
                    include=["documents", "metadatas"],
                    limit=limit,
                    offset=offset
                )
                if batch_ids and batch_ids["documents"]:
                    for i, doc in enumerate(batch_ids["documents"]):
                        meta = batch_ids["metadatas"][i] if batch_ids["metadatas"] else {}
                        tokens = self._tokenize(doc)
                        self.bm25_corpus.append(tokens)
                        self.bm25_metadata.append({
                            "novel_id": meta.get("novel_id", ""),
                            "title": meta.get("title", ""),
                            "chunk_index": meta.get("chunk_index", 0)
                        })
                logger.info(f"BM25 rebuild progress: {min(offset + batch_size, total)}/{total}")
            if self.bm25_corpus:
                if BM25_AVAILABLE:
                    self.bm25 = BM25Okapi(self.bm25_corpus)
                    logger.info(f"BM25 rebuilt from ChromaDB: {len(self.bm25_corpus)} documents")
                self._save_bm25_cache()
            else:
                logger.warning("No documents found, BM25 not available")
        except Exception as e:
            logger.error(f"Failed to rebuild BM25 from ChromaDB: {e}")
    
    def _build_faiss_index(self):
        if not self._use_optimized_search or not self._faiss_store:
            return
        
        logger.info("Building FAISS index from ChromaDB...")
        total = self.collection.count()
        if total == 0:
            logger.warning("No documents in ChromaDB, cannot build FAISS index")
            return
        
        batch_size = 500
        all_vectors = []
        all_ids = []
        all_texts = []
        all_metas = []
        
        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                include=["documents", "metadatas", "embeddings"],
                limit=batch_size,
                offset=offset
            )
            if batch and batch.get("ids"):
                for i, cid in enumerate(batch["ids"]):
                    if batch.get("embeddings") and batch["embeddings"][i]:
                        all_vectors.append(batch["embeddings"][i])
                        all_ids.append(cid)
                        all_texts.append(batch["documents"][i] if batch.get("documents") else "")
                        all_metas.append(batch["metadatas"][i] if batch.get("metadatas") else {})
            
            if offset % 5000 == 0:
                logger.info(f"FAISS build progress: {len(all_vectors)}/{total}")
        
        if all_vectors:
            vectors = np.array(all_vectors, dtype=np.float32)
            self._faiss_store.build_index(vectors, all_ids, all_texts, all_metas)
            logger.info(f"FAISS index built: {self._faiss_store.count} vectors")
    
    def _build_tantivy_index(self):
        if not self._use_optimized_search or not self._tantivy_index:
            return
        
        logger.info("Building Tantivy index from ChromaDB...")
        total = self.collection.count()
        if total == 0:
            return
        
        batch_size = 500
        documents = []
        
        for offset in range(0, total, batch_size):
            batch = self.collection.get(
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
                logger.info(f"Tantivy build progress: {len(documents)}/{total}")
        
        if documents:
            self._tantivy_index.build_index(documents)
            logger.info(f"Tantivy index built: {self._tantivy_index.doc_count} documents")

    def ingest_novel(self, file_path: str, title: str, author: str, genre: str) -> dict:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"文件不存在: {file_path}"}
        text = path.read_text(encoding="utf-8")
        word_count = len(text)
        chunks = self._chunk_text(text)
        novel_id = hashlib.md5(f"{title}_{author}".encode()).hexdigest()[:12]

        existing = self.collection.get(
            where={"novel_id": novel_id}
        )
        if existing and existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

        self.novels[novel_id] = NovelMeta(
            title=title, author=author, genre=genre,
            word_count=word_count, file_path=str(path), chunk_count=len(chunks)
        )
        ids = []
        documents = []
        metadatas = []
        for i, (chunk, chapter_title) in enumerate(chunks):
            chunk_id = f"{novel_id}_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "novel_id": novel_id, "title": title, "author": author,
                "genre": genre, "chunk_index": i, "chapter_title": chapter_title
            })
            tokens = self._tokenize(chunk)
            self.bm25_corpus.append(tokens)
            self.bm25_metadata.append({
                "novel_id": novel_id, "title": title, "chunk_index": i,
                "chapter_title": chapter_title
            })
        if ids:
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                self.collection.upsert(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
        if BM25_AVAILABLE:
            self.bm25 = BM25Okapi(self.bm25_corpus)
        self._save_bm25_cache()
        self._save_state()
        
        if self._use_optimized_search:
            self._build_tantivy_index()
            self._build_faiss_index()
        
        return {
            "novel_id": novel_id, "title": title, "author": author,
            "word_count": word_count, "chunk_count": len(chunks),
            "status": "ingested"
        }

    def semantic_search(self, query: str, n_results: int = 10, novel_filter: Optional[str] = None,
                        genre_filter: Optional[str] = None, chapter_filter: Optional[str] = None) -> list[dict]:
        where_filter = {}
        if novel_filter:
            where_filter["title"] = novel_filter
        if genre_filter:
            where_filter["genre"] = genre_filter
        if chapter_filter:
            where_filter["chapter_title"] = chapter_filter
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                output.append({
                    "text": doc,
                    "metadata": meta,
                    "relevance": round(max(0, 1 - dist / 2), 4),
                    "source": f"{meta.get('title', '')} - {meta.get('author', '')} [{meta.get('chapter_title', '') or 'chunk ' + str(meta.get('chunk_index', ''))}]"
                })
        if not output:
            return [{"status": "no_results", "query": query, "hint": "未找到相关结果，尝试更换关键词或减少过滤条件"}]
        return output

    def bm25_search(self, query: str, n_results: int = 10, novel_filter: Optional[str] = None,
                    genre_filter: Optional[str] = None) -> list[dict]:
        if self._use_optimized_search and self._tantivy_index:
            results = self._tantivy_index.search(query, n_results, novel_filter, genre_filter)
            if results:
                return [{
                    "text": r.text,
                    "metadata": r.metadata,
                    "bm25_score": round(r.score, 4),
                    "source": r.source
                } for r in results]
        
        # Fallback to legacy BM25
        self._ensure_bm25()
        if not self.bm25:
            return [{"status": "index_not_ready", "query": query, "hint": "BM25索引未就绪，请等待索引构建完成或重启服务"}]
        tokens = self._tokenize(query)
        if not tokens:
            return [{"status": "no_results", "query": query, "hint": "分词结果为空，请尝试其他关键词"}]
        scores = self.bm25.get_scores(tokens)
        scored_indices = []
        for idx in range(len(scores)):
            if scores[idx] <= 0:
                continue
            meta = self.bm25_metadata[idx] if idx < len(self.bm25_metadata) else {}
            if novel_filter and meta.get("title", "") != novel_filter:
                continue
            if genre_filter:
                novel_id = meta.get("novel_id", "")
                if novel_id in self.novels and self.novels[novel_id].genre != genre_filter:
                    continue
            scored_indices.append((idx, scores[idx]))
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:n_results]
        results = []
        for idx, score in top_indices:
            meta = self.bm25_metadata[idx] if idx < len(self.bm25_metadata) else {}
            chunk_id = f"{meta.get('novel_id', '')}_{meta.get('chunk_index', 0)}"
            try:
                doc = self.collection.get(ids=[chunk_id], include=["documents"])
                text = doc["documents"][0] if doc["documents"] else ""
            except Exception:
                text = ""
            results.append({
                "text": text,
                "metadata": meta,
                "bm25_score": round(float(score), 4),
                "source": f"{meta.get('title', '')} - {meta.get('author', '')} [{meta.get('chapter_title', '') or 'chunk ' + str(meta.get('chunk_index', ''))}]"
            })
        return results

    def hybrid_search(self, query: str, n_results: int = 10, alpha: float = 0.6,
                      novel_filter: Optional[str] = None, genre_filter: Optional[str] = None) -> list[dict]:
        if self._use_optimized_search and self._hybrid_engine and self._faiss_store:
            if self._faiss_store.count == 0:
                self._build_faiss_index()
            if self._faiss_store.count > 0:
                query_vector = np.array(self.embedding_fn([query])[0], dtype=np.float32)
                return self._hybrid_engine.search(
                    query, query_vector, n_results, alpha, novel_filter, genre_filter
                )
        
        # Fallback to legacy hybrid search
        sem_results = self.semantic_search(query, n_results=n_results * 3,
                                           novel_filter=novel_filter, genre_filter=genre_filter)
        bm25_results = self.bm25_search(query, n_results=n_results * 3,
                                        novel_filter=novel_filter, genre_filter=genre_filter)
        k = 60
        rrf_scores: dict[str, dict] = {}
        for rank, r in enumerate(sem_results):
            if "status" in r:  # skip no_results status dicts
                continue
            key = f"{r['metadata'].get('novel_id', '')}_{r['metadata'].get('chunk_index', '')}"
            if key not in rrf_scores:
                rrf_scores[key] = {"data": r, "sem_score": r["relevance"], "bm25_score": 0}
            rrf_scores[key]["sem_score"] = r["relevance"]
            rrf_scores[key]["sem_rank"] = rank + 1
        for rank, r in enumerate(bm25_results):
            if "status" in r:  # skip no_results/index_not_ready status dicts
                continue
            key = f"{r['metadata'].get('novel_id', '')}_{r['metadata'].get('chunk_index', '')}"
            if key not in rrf_scores:
                rrf_scores[key] = {"data": r, "sem_score": 0, "bm25_score": r.get("bm25_score", 0)}
            rrf_scores[key]["bm25_score"] = r.get("bm25_score", 0)
            rrf_scores[key]["bm25_rank"] = rank + 1
        for key in rrf_scores:
            item = rrf_scores[key]
            sem_rrf = 1.0 / (k + item.get("sem_rank", k * 3)) if "sem_rank" in item else 0
            bm25_rrf = 1.0 / (k + item.get("bm25_rank", k * 3)) if "bm25_rank" in item else 0
            item["hybrid_score"] = round(alpha * sem_rrf + (1 - alpha) * bm25_rrf, 6)
        sorted_items = sorted(rrf_scores.values(), key=lambda x: x["hybrid_score"], reverse=True)
        output = []
        for item in sorted_items[:n_results]:
            result = item["data"].copy()
            result["sem_score"] = item["sem_score"]
            result["bm25_score"] = item["bm25_score"]
            result["hybrid_score"] = item["hybrid_score"]
            output.append(result)
        return output

    def _analyze_section_stats(self, texts: list[str], metas: list[dict]) -> list:
        sections = []
        n = len(texts)
        if n == 0:
            return sections
        segments = [
            ("开篇(前10%)", texts[:max(1, n//10)]),
            ("发展(40%-50%)", texts[max(1, n*2//5):max(2, n//2)]),
            ("高潮(70%-80%)", texts[max(1, n*7//10):max(2, n*4//5)]),
            ("收尾(后10%)", texts[max(1, n*9//10):]),
        ]
        dialogue_re_list = [
            re.compile(r'\u201c(.+?)\u201d'),
            re.compile(r'\u300c(.+?)\u300d'),
            re.compile(r'"(.+?)"'),
        ]
        for label, section_texts in segments:
            if not section_texts:
                continue
            total_chars = sum(len(t) for t in section_texts)
            dialogue_chars = 0
            sentences = []
            for t in section_texts:
                sents = re.split(r'[。！？\n]', t)
                sentences.extend(len(s.strip()) for s in sents if s.strip())
                for dre in dialogue_re_list:
                    for m in dre.finditer(t):
                        dialogue_chars += len(m.group(1))
            avg_sl = round(sum(sentences) / len(sentences), 1) if sentences else 0
            d_ratio = round(dialogue_chars / total_chars, 3) if total_chars else 0
            sample_idx = n//10 if label.startswith("开篇") else (n//2 if label.startswith("发展") else (n*3//4 if label.startswith("高潮") else n*19//20))
            sample_idx = min(sample_idx, n-1)
            sample_text = texts[sample_idx][:800] if sample_idx < len(texts) else ""
            sample_ch = metas[sample_idx].get("chapter_title", "") if sample_idx < len(metas) else ""
            sections.append({
                "section": label,
                "chunk_range": f"{max(1, sample_idx+1)}/{n}",
                "avg_sentence_len": avg_sl,
                "dialogue_ratio": d_ratio,
                "sample_chapter": sample_ch,
                "sample_text": sample_text,
            })
        return sections

    def _extract_humor_scenes(self, texts: list[str], metas: list[dict], exact_title: str) -> list:
        if not self.chat or not texts:
            return []
        n = len(texts)
        sample_indices = [0, n//4, n//2, n*3//4, n-1] if n >= 5 else list(range(n))
        humor_scenes = []
        from webnovel_kb.prompts import HUMOR_SCENE_EXTRACTION_PROMPT
        for idx in sample_indices:
            if idx >= n:
                continue
            chunk_text = texts[idx]
            ch_title = metas[idx].get("chapter_title", "") if idx < len(metas) else ""
            messages = [
                {"role": "system", "content": "你是网文编辑，擅长识别网文中的幽默场景。只提取确实有幽默效果的片段，宁缺毋滥。如果没有幽默内容，直接返回空。"},
                {"role": "user", "content": f"{HUMOR_SCENE_EXTRACTION_PROMPT}\n\ntext:\n{chunk_text}"}
            ]
            try:
                response = self.chat.chat(messages, temperature=0.1, max_tokens=1024)
                if response:
                    matches = re.findall(
                        r'\("humor"<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
                    )
                    for humor_type, snippet, analysis in matches:
                        snippet = snippet.strip()
                        if len(snippet) < 30:
                            continue
                        humor_scenes.append({
                            "type": humor_type.strip(),
                            "chapter": ch_title,
                            "snippet": snippet,
                            "analysis": analysis.strip()
                        })
            except Exception:
                pass
        return humor_scenes[:10]

    def analyze_style(self, novel_title: str) -> dict:
        novel_id = self._find_novel_id(novel_title)
        if not novel_id:
            return {"error": f"未找到小说: {novel_title}"}
        exact_title = self.novels[novel_id].title

        all_chunks_data = self.collection.get(
            where={"title": exact_title},
            include=["documents", "metadatas"]
        )

        if not all_chunks_data or not all_chunks_data.get("documents"):
            return {"error": f"未找到小说内容: {exact_title}"}

        paired = list(zip(
            all_chunks_data["documents"],
            all_chunks_data["metadatas"] or [{}] * len(all_chunks_data["documents"])
        ))
        paired.sort(key=lambda x: x[1].get("chunk_index", 0))
        all_texts = [p[0] for p in paired]
        all_metas = [p[1] for p in paired]

        total_chars = sum(len(t) for t in all_texts)
        if total_chars == 0:
            return {"error": "文本为空"}

        dialogue_chars = 0
        inner_chars = 0
        total_desc_hits = 0
        total_action_hits = 0
        sentence_lengths = []

        dialogue_re_list = [
            re.compile(r'\u201c(.+?)\u201d'),
            re.compile(r'\u300c(.+?)\u300d'),
            re.compile(r'"(.+?)"'),
        ]
        inner_re_list = [
            re.compile(r'\u300e(.+?)\u300f'),
        ]

        for text in all_texts:
            sentences = re.split(r'[。！？\n]', text)
            lengths = [len(s.strip()) for s in sentences if s.strip()]
            sentence_lengths.extend(lengths)

            for dre in dialogue_re_list:
                for m in dre.finditer(text):
                    dialogue_chars += len(m.group(1))
            for ire in inner_re_list:
                for m in ire.finditer(text):
                    inner_chars += len(m.group(1))

            stripped = text
            for dre in dialogue_re_list:
                stripped = dre.sub('', stripped)
            for ire in inner_re_list:
                stripped = ire.sub('', stripped)

        avg_sent_len = round(sum(sentence_lengths) / len(sentence_lengths), 1) if sentence_lengths else 0
        dialogue_ratio = round(dialogue_chars / total_chars, 3) if total_chars else 0
        inner_ratio = round(inner_chars / total_chars, 3) if total_chars else 0

        description_ratio = 0.0
        action_ratio = 0.0

        section_breakdown = self._analyze_section_stats(all_texts, all_metas)
        humor_scenes = self._extract_humor_scenes(all_texts, all_metas, exact_title)

        opening_passages = []
        for i in range(min(3, len(all_texts))):
            opening_passages.append({
                "text": all_texts[i][:800],
                "chapter": all_metas[i].get("chapter_title", "") if i < len(all_metas) else "",
                "position": "opening"
            })
        mid_idx = len(all_texts) // 2
        climax_passages = []
        for j in range(min(3, len(all_texts) - mid_idx)):
            i = mid_idx + j
            climax_passages.append({
                "text": all_texts[i][:800],
                "chapter": all_metas[i].get("chapter_title", "") if i < len(all_metas) else "",
                "position": "climax"
            })
        ending_passages = []
        for j in range(min(3, len(all_texts))):
            i = len(all_texts) - 3 + j
            if i >= 0:
                ending_passages.append({
                    "text": all_texts[i][:800],
                    "chapter": all_metas[i].get("chapter_title", "") if i < len(all_metas) else "",
                    "position": "ending"
                })

        ai_markers_count = 0
        tension_count = 0
        relax_count = 0
        chapter_end_tension = 0
        chapter_end_count = 0
        matched_humor = []

        humor_patterns = [
            r"吐槽", r"自嘲", r"无语", r"呵呵", r"呵呵呵",
            r"这什么鬼", r"搞什么", r"算了", r"算了算了", r"真香",
            r"不是吧", r"不会吧", r"离谱", r"抽象", r"绝了",
            r"好家伙", r"好嘛", r"服了", r"麻了", r"裂开"
        ]
        tension_patterns = [
            r"危险", r"紧迫", r"来不及", r"来不及了",
            r"必须", r"立刻", r"马上", r"冲", r"跑", r"逃",
            r"死", r"杀", r"血", r"疼", r"痛"
        ]
        relax_patterns = [
            r"松了口", r"放松", r"平静", r"安宁", r"悠闲",
            r"舒适", r"温暖", r"安心", r"笑了", r"轻松"
        ]

        ai_patterns = [
            re.compile(r'不禁(?:心头)?(?:一)?(?:颤|愣|动|笑|叹|悲|怒|惊|寒|凛)'),
            re.compile(r'缓缓地?(?:站|走|转|抬|放|伸|收|退|移|开|说|道|开口|闭眼|睁眼|点头|摇头|起身|坐下)'),
            re.compile(r'微微一笑'),
            re.compile(r'嘴角[微轻]?[上扬翘]'),
            re.compile(r'眼中闪过(?:一丝|一抹|一道)?(?:惊|恐|怒|喜|忧|疑|厉|寒|异|凌厉)'),
            re.compile(r'心中暗(?:道|想|叹|惊|喜|怒|说|忖)'),
            re.compile(r'仿佛[^。！？\n]{2,15}一般'),
            re.compile(r'宛如[^。！？\n]{2,15}似的'),
            re.compile(r'犹如[^。！？\n]{2,15}一样'),
            re.compile(r'一股(?:暖流|寒意|力量|气息|杀意|威压|劲风|热流)'),
            re.compile(r'不由自主地?'),
            re.compile(r'情不自禁地?'),
            re.compile(r'若有所思(?:地|地看着)?'),
            re.compile(r'意味深长(?:地|地看着)?'),
            re.compile(r'心念一转'),
            re.compile(r'暗自思忖'),
            re.compile(r'不由得(?:心头)?(?:一)?(?:颤|愣|动|笑|叹|悲|怒|惊|寒)'),
            re.compile(r'淡淡地?(?:说|道|开口|笑|回应|回答|语气)'),
            re.compile(r'深深地?(?:看|望|叹|吸|凝视)'),
            re.compile(r'不禁(?:让|令|使)人'),
            re.compile(r'令人(?:不禁|难以)?(?:心|生|感|觉)'),
        ]

        chapters = {}
        for i, meta in enumerate(all_metas):
            ch = meta.get("chapter_title", "")
            if ch:
                if ch not in chapters:
                    chapters[ch] = []
                chapters[ch].append(i)

        for text in all_texts:
            for p in [re.compile(r) for r in humor_patterns]:
                found = p.findall(text)
                if found:
                    matched_humor.extend(found)
            for p in [re.compile(r) for r in tension_patterns]:
                tension_count += len(p.findall(text))
            for p in [re.compile(r) for r in relax_patterns]:
                relax_count += len(p.findall(text))
            for p in ai_patterns:
                ai_markers_count += len(p.findall(text))

        for ch_name, chunk_indices in chapters.items():
            if len(chunk_indices) >= 1:
                last_idx = chunk_indices[-1]
                if last_idx < len(all_texts):
                    last_text = all_texts[last_idx]
                    tail = last_text[-200:] if len(last_text) > 200 else last_text
                    tail_tension = sum(len(re.compile(p).findall(tail)) for p in tension_patterns)
                    tail_relax = sum(len(re.compile(p).findall(tail)) for p in relax_patterns)
                    if tail_tension > tail_relax:
                        chapter_end_tension += 1
                    chapter_end_count += 1

        chapter_hook_rate = round(chapter_end_tension / chapter_end_count, 2) if chapter_end_count > 0 else 0

        if ai_markers_count > 0 and total_chars > 0:
            markers_per_10k = ai_markers_count / (total_chars / 10000)
            ai_score = min(round(math.log1p(markers_per_10k) * 2.5, 2), 10.0)
        else:
            ai_score = 0.0

        if tension_count + relax_count > 0:
            tr_val = tension_count / (tension_count + relax_count)
            if tr_val > 0.7:
                pace_type = "高压紧绷型"
            elif tr_val > 0.5:
                pace_type = "张弛交替型"
            else:
                pace_type = "舒缓叙事型"
        else:
            pace_type = "无法判断"

        narrative_perspective = "需LLM深度分析"
        if self.chat and all_texts:
            try:
                samples = all_texts[:3]
                sample_text = "\n\n---\n\n".join(s[:800] for s in samples)
                messages = [
                    {"role": "system", "content": "你是网文分析专家。请判断以下文本的叙事人称视角，只返回以下选项之一：第一人称、第三人称限知、第三人称全知、多视角切换。"},
                    {"role": "user", "content": f"分析以下网文片段的叙事视角：\n\n{sample_text}\n\n请只返回视角类型名称。"}
                ]
                result = self.chat.chat(messages, temperature=0.1, max_tokens=32)
                if result:
                    for option in ["第一人称", "第三人称限知", "第三人称全知", "多视角切换"]:
                        if option in result:
                            narrative_perspective = option
                            break
            except Exception as e:
                logger.warning(f"Narrative perspective analysis failed: {e}")

        profile = StyleProfile(
            avg_sentence_len=avg_sent_len,
            dialogue_ratio=dialogue_ratio,
            inner_monologue_ratio=inner_ratio,
            description_ratio=description_ratio,
            action_ratio=action_ratio,
            narrative_perspective=narrative_perspective,
            section_breakdown=section_breakdown,
            humor_scenes=humor_scenes,
            sample_passages=opening_passages + climax_passages + ending_passages,
            ai_fingerprint_score=round(ai_score, 2),
            oral_score=round(min(len(matched_humor) / (total_chars / 5000), 10) if total_chars > 0 else 0, 2),
            chapter_hook_rate=chapter_hook_rate,
            pace_type=pace_type,
            humor_markers=list(dict.fromkeys(matched_humor))[:10],
            pacing_info={
                "tension_markers": tension_count,
                "relax_markers": relax_count,
                "tension_ratio": round(tension_count / (tension_count + relax_count), 2) if (tension_count + relax_count) > 0 else 0,
                "chapter_hook_density": chapter_hook_rate,
                "chapter_end_samples": chapter_end_count
            },
            humor_type="混合型",
            tension_relax_pattern=pace_type,
        )
        self.style_profiles[novel_title] = profile
        self._save_state()
        return asdict(profile)

    def add_entity(self, name: str, entity_type: str, description: str,
                   source_novel: str, attributes: Optional[dict] = None,
                   role: str = "", first_appearance: str = "", arc: str = "") -> dict:
        entity_id = hashlib.md5(f"{name}_{source_novel}".encode()).hexdigest()[:12]
        entity = Entity(
            name=name, entity_type=entity_type, description=description,
            source_novel=source_novel, role=role, first_appearance=first_appearance,
            arc=arc, attributes=attributes or {}
        )
        self.entities[entity_id] = entity
        self.graph.add_node(entity_id, **asdict(entity))
        text = f"{entity_type}: {name}"
        if role:
            text += f" [{role}]"
        if description:
            text += f" - {description}"
        if arc:
            text += f"\n角色弧光: {arc}"
        try:
            self.entities_collection.upsert(
                ids=[f"entity_{entity_id}"],
                documents=[text],
                metadatas=[{
                    "entity_type": entity_type or "",
                    "name": name or "",
                    "source_novel": source_novel or "",
                    "role": role or "",
                    "entity_id": entity_id
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to index new entity: {e}")
        self._save_state()
        return {"entity_id": entity_id, **asdict(entity)}

    def add_relationship(self, source_name: str, target_name: str, rel_type: str,
                         description: str, source_novel: str) -> dict:
        source_id = None
        target_id = None
        for eid, e in self.entities.items():
            if e.name == source_name and e.source_novel == source_novel:
                source_id = eid
            if e.name == target_name and e.source_novel == source_novel:
                target_id = eid
        if not source_id or not target_id:
            return {"error": f"实体未找到: {source_name if not source_id else target_name}"}
        rel = Relationship(
            source=source_name, target=target_name, rel_type=rel_type,
            description=description, source_novel=source_novel
        )
        self.relationships.append(rel)
        self.graph.add_edge(source_id, target_id, rel_type=rel_type, description=description)
        self._save_state()
        return asdict(rel)

    def get_entity_relations(self, entity_name: str, source_novel: Optional[str] = None) -> list[dict]:
        target_id = None
        for eid, e in self.entities.items():
            if e.name == entity_name:
                if source_novel is None or e.source_novel == source_novel:
                    target_id = eid
                    break
        if not target_id:
            return []
        results = []
        for pred in self.graph.predecessors(target_id):
            node = self.graph.nodes[pred]
            edge = self.graph.edges[pred, target_id]
            results.append({
                "direction": "incoming",
                "entity": node.get("name", ""),
                "entity_type": node.get("entity_type", ""),
                "relation": edge.get("rel_type", ""),
                "description": edge.get("description", "")
            })
        for succ in self.graph.successors(target_id):
            node = self.graph.nodes[succ]
            edge = self.graph.edges[target_id, succ]
            results.append({
                "direction": "outgoing",
                "entity": node.get("name", ""),
                "entity_type": node.get("entity_type", ""),
                "relation": edge.get("rel_type", ""),
                "description": edge.get("description", "")
            })
        return results

    def search_entities_semantic(self, query: str, n_results: int = 10,
                                  entity_type: Optional[str] = None,
                                  source_novel: Optional[str] = None) -> list[dict]:
        if self.entities_collection.count() == 0:
            return [{"status": "no_entities_indexed", "hint": "请先提取实体"}]
        where_filter = {}
        if entity_type:
            where_filter["entity_type"] = entity_type
        if source_novel:
            where_filter["source_novel"] = source_novel
        results = self.entities_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                eid = meta.get("entity_id", "")
                if eid and eid in self.entities:
                    entity = asdict(self.entities[eid])
                    entity["semantic_score"] = round(max(0, 1 - dist / 2), 4)
                    relations = self.get_entity_relations(entity["name"], entity.get("source_novel"))
                    entity["relation_count"] = len(relations)
                    output.append(entity)
        return output

    def add_plot_pattern(self, pattern_type: str, description: str, source_novel: str,
                         source_chapter: str, pattern_text: str = "",
                         before_context: str = "", after_context: str = "",
                         effectiveness: str = "") -> dict:
        pattern = PlotPattern(
            pattern_type=pattern_type, description=description,
            source_novel=source_novel, source_chapter=source_chapter,
            pattern_text=pattern_text, before_context=before_context,
            after_context=after_context, effectiveness=effectiveness
        )
        self.plot_patterns.append(pattern)
        idx = len(self.plot_patterns) - 1
        pattern_id = f"pattern_{idx}"
        text = f"{pattern_type}: {description}"
        if pattern_text:
            text += f"\n{pattern_text}"
        try:
            self.patterns_collection.upsert(
                ids=[pattern_id],
                documents=[text],
                metadatas=[{
                    "pattern_type": pattern_type or "",
                    "source_novel": source_novel or "",
                    "effectiveness": effectiveness or "",
                    "index": idx
                }]
            )
        except Exception as e:
            logger.warning(f"Failed to index new plot pattern: {e}")
        self._save_state()
        return asdict(pattern)

    def search_plot_patterns(self, pattern_type: Optional[str] = None,
                             source_novel: Optional[str] = None) -> list[dict]:
        results = []
        for p in self.plot_patterns:
            if pattern_type and p.pattern_type != pattern_type:
                continue
            if source_novel and p.source_novel != source_novel:
                continue
            results.append(asdict(p))
        return results

    def search_plot_patterns_semantic(self, query: str, n_results: int = 10,
                                       pattern_type: Optional[str] = None,
                                       source_novel: Optional[str] = None) -> list[dict]:
        if self.patterns_collection.count() == 0:
            return [{"status": "no_patterns_indexed", "hint": "请先提取情节模式"}]
        where_filter = {}
        if pattern_type:
            where_filter["pattern_type"] = pattern_type
        if source_novel:
            where_filter["source_novel"] = source_novel
        results = self.patterns_collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                idx = meta.get("index", 0)
                if idx < len(self.plot_patterns):
                    pattern = asdict(self.plot_patterns[idx])
                    pattern["semantic_score"] = round(max(0, 1 - dist / 2), 4)
                    output.append(pattern)
        return output

    def compare_styles(self, novel_titles: list[str]) -> dict:
        comparison = {}
        for title in novel_titles:
            if title in self.style_profiles:
                comparison[title] = asdict(self.style_profiles[title])
            else:
                comparison[title] = self.analyze_style(title)
        return comparison

    def rerank_search(self, query: str, n_results: int = 10,
                      novel_filter: Optional[str] = None,
                      genre_filter: Optional[str] = None) -> list[dict]:
        if not self.reranker:
            return self.hybrid_search(query, n_results=n_results,
                                      novel_filter=novel_filter, genre_filter=genre_filter)
        candidates = self.hybrid_search(query, n_results=n_results * 5,
                                        novel_filter=novel_filter, genre_filter=genre_filter)
        if not candidates:
            return []
        documents = [c["text"] for c in candidates]
        rerank_results = self.reranker.rerank(query, documents, top_n=n_results)
        output = []
        for item in rerank_results:
            idx = item.get("index", 0)
            if idx < len(candidates):
                c = candidates[idx].copy()
                c["rerank_score"] = item.get("relevance_score", 0)
                c["search_method"] = "hybrid+rerank"
                output.append(c)
        return output

    def auto_extract_entities(self, novel_title: str, max_chunks: int = 20) -> dict:
        if not self.chat:
            return {"error": "Chat API未配置，无法自动提取实体。请设置LLM_API_KEY。"}
        novel_id = self._find_novel_id(novel_title)
        if not novel_id:
            return {"error": f"未找到小说: {novel_title}"}
        exact_title = self.novels[novel_id].title
        results = self.collection.query(
            query_texts=["角色登场 人物介绍 关系揭示"],
            n_results=max_chunks,
            where={"title": exact_title}
        )
        if not results or not results["documents"]:
            return {"error": f"未找到小说内容: {exact_title}"}
        from webnovel_kb.prompts import ENTITY_EXTRACTION_PROMPT
        all_extracted = {"entities": [], "relationships": []}
        for chunk in results["documents"][0][:max_chunks]:
            prompt = ENTITY_EXTRACTION_PROMPT.replace("{tuple_delimiter}", "<|>")
            prompt = prompt.replace("{record_delimiter}", "|||")
            prompt = prompt.replace("{completion_delimiter}", "<DONE>")
            messages = [
                {"role": "system", "content": "你是网文分析专家，擅长从文本中提取角色、地点、组织等实体及其关系。"},
                {"role": "user", "content": f"{prompt}\n\ntext:\n{chunk[:2000]}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            if response:
                entity_matches = re.findall(
                    r'\("entity"<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
                )
                for name, etype, desc in entity_matches:
                    name = name.strip()
                    etype = etype.strip()
                    desc = desc.strip()
                    if name and etype in ["角色", "地点", "组织", "物品", "能力", "概念", "事件",
                                           "职业", "种族", "势力", "技能", "状态", "伏笔"]:
                        existing = False
                        for eid, e in self.entities.items():
                            if e.name == name and e.source_novel == novel_title:
                                existing = True
                                break
                        if not existing:
                            result = self.add_entity(name, etype, desc, novel_title)
                            all_extracted["entities"].append(result)

                rel_matches = re.findall(
                    r'\("relationship"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(\d+)\)', response
                )
                for source, target, desc, strength in rel_matches:
                    source = source.strip()
                    target = target.strip()
                    desc = desc.strip()
                    if source and target:
                        try:
                            result = self.add_relationship(source, target, "相关", desc, novel_title)
                            if "error" not in result:
                                all_extracted["relationships"].append(result)
                        except Exception:
                            pass
        return {
            "novel": novel_title,
            "entities_extracted": len(all_extracted["entities"]),
            "relationships_extracted": len(all_extracted["relationships"]),
            "entities": all_extracted["entities"][:20],
            "relationships": all_extracted["relationships"][:20]
        }

    def auto_extract_plot_patterns(self, novel_title: str, max_chunks: int = 20) -> dict:
        if not self.chat:
            return {"error": "Chat API未配置，无法自动提取情节模式。请设置LLM_API_KEY。"}
        novel_id = self._find_novel_id(novel_title)
        if not novel_id:
            return {"error": f"未找到小说: {novel_title}"}
        exact_title = self.novels[novel_id].title
        results = self.collection.query(
            query_texts=["悬念 反转 冲突升级 伏笔 情节转折"],
            n_results=max_chunks,
            where={"title": exact_title}
        )
        if not results or not results["documents"]:
            return {"error": f"未找到小说内容: {exact_title}"}
        from webnovel_kb.prompts import PLOT_PATTERN_EXTRACTION_PROMPT
        all_patterns = []
        for chunk in results["documents"][0][:max_chunks]:
            prompt = PLOT_PATTERN_EXTRACTION_PROMPT.replace("{tuple_delimiter}", "<|>")
            prompt = prompt.replace("{record_delimiter}", "|||")
            prompt = prompt.replace("{completion_delimiter}", "<DONE>")
            messages = [
                {"role": "system", "content": "你是资深网文编辑和写作教练，擅长识别真正有学习价值的叙事技巧。你只提取写法精妙、可复用的情节模式，宁缺毋滥。如果文本中没有值得学习的写法，输出空结果。"},
                {"role": "user", "content": f"{prompt}\n\ntext:\n{chunk}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            logger.info(f"LLM response for plot pattern: {response[:500] if response else 'None'}")
            if response:
                pattern_matches = re.findall(
                    r'\("pattern"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
                )
                logger.info(f"Pattern matches found: {len(pattern_matches)}")
                for ptype, desc, before, ptext, after, eff in pattern_matches:
                    ptype = ptype.strip()
                    desc = desc.strip()
                    if ptype and desc:
                        ptext_full = ptext.strip()
                        if len(ptext_full) < 500:
                            continue
                        result = self.add_plot_pattern(
                            pattern_type=ptype,
                            description=desc,
                            source_novel=novel_title,
                            source_chapter="auto",
                            pattern_text=ptext_full,
                            before_context=before.strip(),
                            after_context=after.strip(),
                            effectiveness=eff.strip()
                        )
                        all_patterns.append(result)
        return {
            "novel": novel_title,
            "patterns_extracted": len(all_patterns),
            "patterns": all_patterns[:20]
        }

    def add_writing_template(self, template_type: str, scene_type: str, structure: str,
                             key_beats: list, source_novel: str, source_chapter: str = "",
                             example_text: str = "", effectiveness: str = "") -> dict:
        template = WritingTemplate(
            template_type=template_type, scene_type=scene_type,
            structure=structure, key_beats=key_beats,
            source_novel=source_novel, source_chapter=source_chapter,
            example_text=example_text, effectiveness=effectiveness
        )
        self.writing_templates.append(template)
        self._save_state()
        return asdict(template)

    def search_writing_templates(self, scene_type: Optional[str] = None,
                                 template_type: Optional[str] = None,
                                 source_novel: Optional[str] = None) -> list[dict]:
        results = []
        for t in self.writing_templates:
            if scene_type and t.scene_type != scene_type:
                continue
            if template_type and t.template_type != template_type:
                continue
            if source_novel and t.source_novel != source_novel:
                continue
            results.append(asdict(t))
        return results

    def auto_extract_writing_templates(self, novel_title: str, max_chunks: int = 15) -> dict:
        if not self.chat:
            return {"error": "Chat API未配置，无法自动提取写作模板。请设置LLM_API_KEY。"}
        novel_id = self._find_novel_id(novel_title)
        if not novel_id:
            return {"error": f"未找到小说: {novel_title}"}
        exact_title = self.novels[novel_id].title

        all_chunks_data = self.collection.get(
            where={"title": exact_title},
            include=["documents", "metadatas"]
        )
        if not all_chunks_data or not all_chunks_data.get("documents"):
            return {"error": f"未找到小说内容: {exact_title}"}

        paired = list(zip(
            all_chunks_data["documents"],
            all_chunks_data["metadatas"] or [{}] * len(all_chunks_data["documents"])
        ))
        paired.sort(key=lambda x: x[1].get("chunk_index", 0))

        total_chunks = len(paired)
        step = max(1, total_chunks // max_chunks)
        sampled = paired[::step][:max_chunks]

        from webnovel_kb.prompts import WRITING_TEMPLATE_EXTRACTION_PROMPT
        all_templates = []
        for chunk_text, meta in sampled:
            chapter_title = meta.get("chapter_title", "")
            messages = [
                {"role": "system", "content": "你是资深网文编辑和写作教练，擅长从优秀网文中提取可复用的场景写法模板。你只提取结构清晰、有学习价值的模板，宁缺毋滥。如果文本中没有值得提取的模板，输出空结果。"},
                {"role": "user", "content": f"{WRITING_TEMPLATE_EXTRACTION_PROMPT}\n\ntext:\n{chunk_text}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            if response:
                template_matches = re.findall(
                    r'\("template"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
                )
                for scene_type, structure, beats_str, example, eff in template_matches:
                    scene_type = scene_type.strip()
                    structure = structure.strip()
                    if scene_type and structure:
                        beats = [b.strip() for b in beats_str.split(',') if b.strip()]
                        result = self.add_writing_template(
                            template_type="场景模板",
                            scene_type=scene_type,
                            structure=structure,
                            key_beats=beats,
                            source_novel=novel_title,
                            source_chapter=chapter_title,
                            example_text=example.strip(),
                            effectiveness=eff.strip()
                        )
                        all_templates.append(result)
        return {
            "novel": novel_title,
            "templates_extracted": len(all_templates),
            "templates": all_templates[:20]
        }

    def auto_extract_scene_patterns(self, novel_title: str, max_chunks: int = 15) -> dict:
        if not self.chat:
            return {"error": "Chat API未配置，无法自动提取场景模式。请设置LLM_API_KEY。"}
        novel_id = self._find_novel_id(novel_title)
        if not novel_id:
            return {"error": f"未找到小说: {novel_title}"}
        exact_title = self.novels[novel_id].title

        all_chunks_data = self.collection.get(
            where={"title": exact_title},
            include=["documents", "metadatas"]
        )
        if not all_chunks_data or not all_chunks_data.get("documents"):
            return {"error": f"未找到小说内容: {exact_title}"}

        paired = list(zip(
            all_chunks_data["documents"],
            all_chunks_data["metadatas"] or [{}] * len(all_chunks_data["documents"])
        ))
        paired.sort(key=lambda x: x[1].get("chunk_index", 0))

        total_chunks = len(paired)
        step = max(1, total_chunks // max_chunks)
        sampled = paired[::step][:max_chunks]

        from webnovel_kb.prompts import SCENE_PATTERN_PROMPT
        all_patterns = []
        for chunk_text, meta in sampled:
            chapter = meta.get("chapter_title", "auto")
            messages = [
                {"role": "system", "content": "你是资深网文编辑，擅长识别具体场景中的叙事技巧。你只提取写法精妙、可复用的场景模式，宁缺毋滥。如果文本中没有值得学习的写法，输出空结果。"},
                {"role": "user", "content": f"{SCENE_PATTERN_PROMPT}\n\ntext:\n{chunk_text}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            if response:
                pattern_matches = re.findall(
                    r'\("scene_pattern"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
                )
                for scene_type, technique, analysis, original, reusable in pattern_matches:
                    scene_type = scene_type.strip()
                    technique = technique.strip()
                    if scene_type and technique:
                        result = self.add_plot_pattern(
                            pattern_type=f"场景写法/{scene_type}",
                            description=f"{technique}: {analysis.strip()}",
                            source_novel=novel_title,
                            source_chapter=chapter,
                            pattern_text=original.strip(),
                            effectiveness=reusable.strip()
                        )
                        all_patterns.append(result)
        return {
            "novel": novel_title,
            "scene_patterns_extracted": len(all_patterns),
            "patterns": all_patterns[:20]
        }

    def deai_polish(self, text: str, style_reference: str = "") -> dict:
        if not self.chat:
            return {"error": "Chat API未配置，无法执行去AI味润色。请设置LLM_API_KEY。"}
        from webnovel_kb.prompts import DEAI_POLISH_PROMPT
        style_hint = ""
        if style_reference and style_reference in self.style_profiles:
            profile = self.style_profiles[style_reference]
            style_hint = f"\n\n参考风格：{style_reference}\n平均句长：{profile.avg_sentence_len}\n对话比例：{profile.dialogue_ratio}\n幽默类型：{profile.humor_type}\n张弛模式：{profile.tension_relax_pattern}"
        messages = [
            {"role": "system", "content": "你是顶级网文编辑，擅长将AI生成的文本润色为自然、有网感的中文网文风格。"},
            {"role": "user", "content": f"{DEAI_POLISH_PROMPT}\n\n{style_hint}\n\n原始文本：\n{text}"}
        ]
        response = self.chat.chat(messages, temperature=0.5, max_tokens=8192)
        if response:
            return {"polished_text": response, "original_length": len(text), "polished_length": len(response)}
        return {"error": "润色失败，API未返回结果"}

    def find_similar(self, text: str, n_results: int = 10, novel_filter: Optional[str] = None,
                     genre_filter: Optional[str] = None, use_rerank: bool = False) -> list[dict]:
        if use_rerank and self.reranker:
            return self.rerank_search(text, n_results=n_results,
                                      novel_filter=novel_filter, genre_filter=genre_filter)
        return self.semantic_search(text, n_results=n_results,
                                    novel_filter=novel_filter, genre_filter=genre_filter)

    def _dedupe_results(self, results: list[dict]) -> list[dict]:
        """去除内容高度重叠的重复结果。若一条内容是另一条的子串，保留较长的。"""
        if len(results) <= 1:
            return results
        deduped = []
        for item in results:
            text = (item.get("text") or item.get("description") or item.get("content") or "").strip()
            if not text:
                deduped.append(item)
                continue
            is_dup = False
            for i, kept in enumerate(deduped):
                kept_text = (kept.get("text") or kept.get("description") or kept.get("content") or "").strip()
                if len(kept_text) < 20 or len(text) < 20:
                    continue
                if text in kept_text:
                    is_dup = True
                    break
                if kept_text in text:
                    deduped[i] = item
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(item)
        return deduped

    def _format_search_results(self, raw_results: list[dict], output_format: str = "compact",
                               max_content_length: int = 0, dedupe: bool = False) -> list:
        """
        格式化搜索结果。output_format:
        - raw: 完整结构化数据（调试用）
        - compact: "[来源] 内容..." 作家可直接阅读
        - clean: 仅纯文本内容，无来源标记
        """
        if not raw_results:
            return raw_results
        if len(raw_results) == 1 and "status" in raw_results[0]:
            return raw_results

        items = self._dedupe_results(raw_results) if dedupe else raw_results

        if output_format == "raw":
            return items

        if output_format == "clean":
            output = []
            for item in items:
                text = item.get("text") or item.get("description") or ""
                if max_content_length > 0 and len(text) > max_content_length:
                    text = text[:max_content_length] + "…"
                output.append(text)
            return output

        # compact (default)
        output = []
        for item in items:
            source = item.get("source", "")
            if not source:
                title = ""
                chapter = ""
                meta = item.get("metadata", {})
                if meta:
                    title = meta.get("title", "")
                    chapter = meta.get("chapter_title", "")
                if not title:
                    title = item.get("source_novel", "") or item.get("novel_title", "")
                source_parts = []
                if title:
                    source_parts.append(f"《{title}》")
                if chapter:
                    source_parts.append(chapter)
                source = " ".join(source_parts)
            
            text = item.get("text") or item.get("description") or ""
            if max_content_length > 0 and len(text) > max_content_length:
                text = text[:max_content_length] + "…"

            if source:
                output.append(f"[{source}] {text}")
            else:
                output.append(text)
        return output

    def unified_search(self, query: str, mode: str = "hybrid", n_results: int = 10,
                       novel_filter: Optional[str] = None, genre_filter: Optional[str] = None,
                       chapter_filter: Optional[str] = None, alpha: float = 0.6,
                       use_rerank: bool = False,
                       output_format: str = "compact", max_content_length: int = 0,
                       dedupe: bool = True) -> list:
        """
        统一搜索接口，合并semantic/bm25/hybrid/rerank
        
        mode: semantic(语义), bm25(关键词), hybrid(混合), rerank(精排)
        alpha: hybrid模式下语义权重(0-1)
        use_rerank: 是否使用rerank精排(仅对hybrid模式有效)
        output_format: raw(完整JSON), compact(简洁来源+内容), clean(纯文本)
        max_content_length: 每条内容最大字数(0=不限制)
        dedupe: 是否去重(默认True)
        """
        if mode == "semantic":
            raw = self.semantic_search(query, n_results, novel_filter, genre_filter, chapter_filter)
        elif mode == "bm25":
            raw = self.bm25_search(query, n_results, novel_filter, genre_filter)
        elif mode == "rerank":
            raw = self.rerank_search(query, n_results, novel_filter, genre_filter)
        else:  # hybrid (默认)
            if use_rerank and self.reranker:
                raw = self.rerank_search(query, n_results, novel_filter, genre_filter)
            else:
                raw = self.hybrid_search(query, n_results, alpha, novel_filter, genre_filter)
        return self._format_search_results(raw, output_format, max_content_length, dedupe)

    def search_knowledge(self, query: str = "", knowledge_type: str = "plot_patterns",
                         n_results: int = 10, use_semantic: bool = True,
                         type_filter: str = "", source_novel: str = "",
                         output_format: str = "compact", max_content_length: int = 0,
                         dedupe: bool = True) -> list:
        """
        统一知识搜索接口，合并情节模式/写法模板的搜索
        
        knowledge_type: plot_patterns(情节模式), writing_templates(写法模板)
        use_semantic: True用语义搜索，False用关键字过滤
        type_filter: 情节模式类型或场景类型
        output_format: raw(完整JSON), compact(简洁来源+内容), clean(纯文本)
        max_content_length: 每条内容最大字数(0=不限制)
        dedupe: 是否去重(默认True)
        """
        if knowledge_type == "writing_templates":
            if use_semantic and query:
                results = []
                for t in self.writing_templates:
                    if type_filter and t.scene_type != type_filter:
                        continue
                    if source_novel and t.source_novel != source_novel:
                        continue
                    results.append(asdict(t))
                if query:
                    query_lower = query.lower()
                    results = [r for r in results if 
                               query_lower in r.get("scene_type", "").lower() or
                               query_lower in r.get("description", "").lower() or
                               query_lower in str(r.get("steps", "")).lower()]
                raw = results[:n_results]
            else:
                raw = self.search_writing_templates(
                    scene_type=type_filter or None,
                    source_novel=source_novel or None
                )
        else:  # plot_patterns
            if use_semantic and query:
                raw = self.search_plot_patterns_semantic(
                    query, n_results,
                    pattern_type=type_filter or None,
                    source_novel=source_novel or None
                )
            else:
                raw = self.search_plot_patterns(
                    pattern_type=type_filter or None,
                    source_novel=source_novel or None
                )
        return self._format_search_results(raw, output_format, max_content_length, dedupe)

    def _run_async_extraction(self, task_id: str, novel_title: str, max_chunks: int, extract_type: str):
        with self._task_lock:
            if task_id not in self._background_tasks:
                return
            self._background_tasks[task_id]["status"] = "running"
            self._background_tasks[task_id]["started_at"] = datetime.now().isoformat()

        try:
            if extract_type == "plot_patterns":
                result = self._do_extract_plot_patterns(novel_title, max_chunks)
            elif extract_type == "writing_templates":
                result = self.auto_extract_writing_templates(novel_title, max_chunks)
            elif extract_type == "scene_patterns":
                result = self.auto_extract_scene_patterns(novel_title, max_chunks)
            else:
                result = self._do_extract_entities(novel_title, max_chunks)

            with self._task_lock:
                if task_id in self._background_tasks:
                    self._background_tasks[task_id]["status"] = "completed"
                    self._background_tasks[task_id]["completed_at"] = datetime.now().isoformat()
                    self._background_tasks[task_id]["result"] = result
        except Exception as e:
            logger.error(f"Async extraction task {task_id} failed: {e}", exc_info=True)
            with self._task_lock:
                if task_id in self._background_tasks:
                    self._background_tasks[task_id]["status"] = "failed"
                    self._background_tasks[task_id]["error"] = str(e)

    def _do_extract_plot_patterns(self, novel_title: str, max_chunks: int) -> dict:
        if not self.chat:
            return {"error": "Chat API未配置"}
        novel_id = self._find_novel_id(novel_title)
        if not novel_id:
            return {"error": f"未找到小说: {novel_title}"}
        exact_title = self.novels[novel_id].title

        all_chunks_data = self.collection.get(
            where={"title": exact_title},
            include=["documents", "metadatas"]
        )
        if not all_chunks_data or not all_chunks_data.get("documents"):
            return {"error": f"未找到小说内容: {exact_title}"}

        chunks_with_meta = list(zip(
            all_chunks_data["documents"],
            all_chunks_data["metadatas"]
        ))
        chunks_with_meta.sort(key=lambda x: x[1].get("chunk_index", 0))

        total_chunks = len(chunks_with_meta)
        step = max(1, total_chunks // max_chunks)
        sampled = chunks_with_meta[::step][:max_chunks]

        from webnovel_kb.prompts import PLOT_TIMELINE_PROMPT, PLOT_PATTERN_CROSS_CHUNK_PROMPT

        with self._task_lock:
            for tid, task in list(self._background_tasks.items()):
                if task.get("type") == "plot_patterns" and task.get("novel") == novel_title:
                    task["progress"] = f"构建时间线 0/{len(sampled)}"

        timeline_events = []
        chunk_map = {}
        for idx, (chunk_text, meta) in enumerate(sampled):
            chunk_index = meta.get("chunk_index", idx)
            chapter = meta.get("chapter_title", f"片段{idx+1}")
            chunk_map[idx] = {"text": chunk_text, "chapter": chapter, "chunk_index": chunk_index}

            prompt = PLOT_TIMELINE_PROMPT.replace("{tuple_delimiter}", "<|>")
            prompt = prompt.replace("{record_delimiter}", "|||")
            prompt = prompt.replace("{completion_delimiter}", "<DONE>")
            messages = [
                {"role": "system", "content": "你是网文情节分析助手，擅长从文本中提取关键情节事件。"},
                {"role": "user", "content": f"{prompt}\n\ntext:\n{chunk_text}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            if response:
                for line in response.strip().split("\n"):
                    line = line.strip()
                    m = re.match(r'\[(\d+)\]\s*(.+)', line)
                    if m:
                        event_desc = m.group(2).strip()
                        if event_desc and event_desc != "<DONE>":
                            timeline_events.append(f"[{len(timeline_events)+1}] {event_desc}")

            with self._task_lock:
                for tid, task in list(self._background_tasks.items()):
                    if task.get("type") == "plot_patterns" and task.get("novel") == novel_title:
                        task["progress"] = f"构建时间线 {idx+1}/{len(sampled)}"

        if not timeline_events:
            return {"novel": novel_title, "patterns_extracted": 0, "patterns": []}

        timeline_text = "\n".join(timeline_events)
        logger.info(f"Timeline for {novel_title}: {len(timeline_events)} events")

        with self._task_lock:
            for tid, task in list(self._background_tasks.items()):
                if task.get("type") == "plot_patterns" and task.get("novel") == novel_title:
                    task["progress"] = "分析长线模式..."

        prompt = PLOT_PATTERN_CROSS_CHUNK_PROMPT.replace("{tuple_delimiter}", "<|>")
        prompt = prompt.replace("{record_delimiter}", "|||")
        prompt = prompt.replace("{completion_delimiter}", "<DONE>")
        messages = [
            {"role": "system", "content": "你是资深网文编辑和写作教练，擅长识别跨章节的长线叙事模式。你只提取纵览全局才能发现的模式，单章节内的手法不值得提取。"},
            {"role": "user", "content": f"{prompt}\n\n时间线：\n{timeline_text}"}
        ]
        response = self.chat.chat(messages, temperature=0.3, max_tokens=8192)
        logger.info(f"Cross-chunk pattern response: {response[:500] if response else 'None'}")

        all_patterns = []
        if response:
            pattern_matches = re.findall(
                r'\("pattern"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
            )
            logger.info(f"Cross-chunk pattern matches found: {len(pattern_matches)}")
            for ptype, desc, setup_text, payoff_text, bridge, eff in pattern_matches:
                ptype = ptype.strip()
                desc = desc.strip()
                setup_text = setup_text.strip()
                payoff_text = payoff_text.strip()
                if ptype and desc and (len(setup_text) > 50 or len(payoff_text) > 50):
                    pattern_text = f"【起点】\n{setup_text}\n\n【终点】\n{payoff_text}"
                    result = self.add_plot_pattern(
                        pattern_type=ptype,
                        description=desc,
                        source_novel=novel_title,
                        source_chapter="跨章节",
                        pattern_text=pattern_text,
                        before_context=setup_text[:200],
                        after_context=payoff_text[:200],
                        effectiveness=eff.strip()
                    )
                    all_patterns.append(result)

        with self._task_lock:
            for tid, task in list(self._background_tasks.items()):
                if task.get("type") == "plot_patterns" and task.get("novel") == novel_title:
                    task["progress"] = "完成"

        return {
            "novel": novel_title,
            "patterns_extracted": len(all_patterns),
            "timeline_events": len(timeline_events),
            "patterns": all_patterns[:20]
        }

    def _do_extract_entities(self, novel_title: str, max_chunks: int = 50) -> dict:
        if not self.chat:
            return {"error": "Chat API未配置"}
        novel_id = self._find_novel_id(novel_title)
        if not novel_id:
            return {"error": f"未找到小说: {novel_title}"}
        exact_title = self.novels[novel_id].title

        all_chunks_data = self.collection.get(
            where={"title": exact_title},
            include=["documents", "metadatas"]
        )
        if not all_chunks_data or not all_chunks_data.get("documents"):
            return {"error": f"未找到小说内容: {exact_title}"}

        chunks_with_meta = list(zip(
            all_chunks_data["documents"],
            all_chunks_data["metadatas"]
        ))
        chunks_with_meta.sort(key=lambda x: x[1].get("chunk_index", 0))

        total_chunks = len(chunks_with_meta)
        
        key_scene_queries = [
            "角色关系 冲突 对峙 背叛 结盟",
            "师徒 朋友 敌人 宿敌 暗恋",
            "家族 亲情 爱情 兄弟 姐妹"
        ]
        key_scene_indices = set()
        for query in key_scene_queries:
            key_results = self.collection.query(
                query_texts=[query],
                n_results=min(15, total_chunks // 3),
                where={"title": exact_title}
            )
            if key_results and key_results.get("metadatas"):
                for meta in key_results["metadatas"][0]:
                    idx = meta.get("chunk_index", -1)
                    if idx >= 0:
                        key_scene_indices.add(idx)
        
        uniform_step = max(1, total_chunks // (max_chunks - len(key_scene_indices)))
        uniform_sampled = chunks_with_meta[::uniform_step]
        
        key_sampled = []
        for idx, (doc, meta) in enumerate(chunks_with_meta):
            if meta.get("chunk_index", -1) in key_scene_indices:
                key_sampled.append((doc, meta))
        
        sampled = list(uniform_sampled)
        for item in key_sampled:
            if item not in sampled:
                sampled.append(item)
        sampled = sampled[:max_chunks + len(key_sampled)]
        
        from webnovel_kb.prompts import ENTITY_TIMELINE_PROMPT, ENTITY_CROSS_CHUNK_PROMPT

        with self._task_lock:
            for tid, task in list(self._background_tasks.items()):
                if task.get("type") == "entities" and task.get("novel") == novel_title:
                    task["progress"] = f"构建实体时间线 0/{len(sampled)}"

        entity_timeline = []
        for idx, (chunk_text, meta) in enumerate(sampled):
            prompt = ENTITY_TIMELINE_PROMPT.replace("{tuple_delimiter}", "<|>")
            prompt = prompt.replace("{record_delimiter}", "|||")
            prompt = prompt.replace("{completion_delimiter}", "<DONE>")
            messages = [
                {"role": "system", "content": "你是网文角色分析助手，擅长从文本中提取角色和关键实体信息。"},
                {"role": "user", "content": f"{prompt}\n\ntext:\n{chunk_text}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            if response:
                for line in response.strip().split("\n"):
                    line = line.strip()
                    if line and line != "<DONE>" and not line.startswith("#"):
                        entity_timeline.append(line)

            with self._task_lock:
                for tid, task in list(self._background_tasks.items()):
                    if task.get("type") == "entities" and task.get("novel") == novel_title:
                        task["progress"] = f"构建实体时间线 {idx+1}/{len(sampled)}"

        if not entity_timeline:
            return {"novel": novel_title, "entities_extracted": 0, "relationships_extracted": 0}

        seen = set()
        deduped = []
        for line in entity_timeline:
            key = line[:80]
            if key not in seen:
                seen.add(key)
                deduped.append(line)
        entity_timeline = deduped
        logger.info(f"Entity timeline for {novel_title}: {len(entity_timeline)} unique entries")

        with self._task_lock:
            for tid, task in list(self._background_tasks.items()):
                if task.get("type") == "entities" and task.get("novel") == novel_title:
                    task["progress"] = "分析角色弧光与关系..."

        all_extracted = {"entities": [], "relationships": []}
        
        max_timeline_chars = 20000
        timeline_text = ""
        for line in entity_timeline:
            if len(timeline_text) + len(line) + 1 > max_timeline_chars:
                break
            timeline_text += line + "\n"

        prompt = ENTITY_CROSS_CHUNK_PROMPT.replace("{tuple_delimiter}", "<|>")
        prompt = prompt.replace("{record_delimiter}", "|||")
        prompt = prompt.replace("{completion_delimiter}", "<DONE>")
        messages = [
            {"role": "system", "content": "你是资深网文编辑，擅长分析角色弧光、关系演变和能力体系。你只提取跨章节才能发现的深层信息。"},
            {"role": "user", "content": f"{prompt}\n\n时间线：\n{timeline_text}"}
        ]
        response = self.chat.chat(messages, temperature=0.3, max_tokens=16384)
        logger.info(f"Cross-chunk entity response: {response[:500] if response else 'None'}")

        if response:
            entity_matches = re.findall(
                r'\("entity"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.*?)\)', response
            )
            for match in entity_matches:
                name = match[0].strip()
                etype = match[1].strip()
                desc = match[2].strip()
                role = match[3].strip()
                first_appear = match[4].strip()
                arc = match[5].strip()
                if name and etype:
                    existing = False
                    for eid, e in self.entities.items():
                        if e.name == name and e.source_novel == novel_title:
                            if arc and not e.arc:
                                e.arc = arc
                                e.description = desc
                                e.role = role
                                self._save_state()
                            existing = True
                            break
                    if not existing:
                        result = self.add_entity(name, etype, desc, novel_title,
                                                 role=role, first_appearance=first_appear, arc=arc)
                        all_extracted["entities"].append(result)

            rel_matches = re.findall(
                r'\("relationship"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.*?)\)', response
            )
            for match in rel_matches:
                source = match[0].strip()
                target = match[1].strip()
                rel_type = match[2].strip()
                desc = match[3].strip()
                evolution = match[4].strip()
                if source and target and rel_type:
                    try:
                        result = self.add_relationship(source, target, rel_type, desc, novel_title)
                        if "error" not in result and evolution:
                            self.relationships[-1].evolution = evolution
                            self._save_state()
                        if "error" not in result:
                            all_extracted["relationships"].append(result)
                    except Exception:
                        pass

        with self._task_lock:
            for tid, task in list(self._background_tasks.items()):
                if task.get("type") == "entities" and task.get("novel") == novel_title:
                    task["progress"] = "完成"

        return {
            "novel": novel_title,
            "entities_extracted": len(all_extracted["entities"]),
            "relationships_extracted": len(all_extracted["relationships"]),
            "entities": all_extracted["entities"][:20],
            "relationships": all_extracted["relationships"][:20]
        }

    def start_async_extraction(self, novel_title: str, max_chunks: int = 20, extract_type: str = "plot_patterns") -> dict:
        task_id = str(uuid.uuid4())[:8]
        task_info = {
            "task_id": task_id,
            "type": extract_type,
            "novel": novel_title,
            "max_chunks": max_chunks,
            "status": "pending",
            "progress": "0/0",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }
        with self._task_lock:
            self._background_tasks[task_id] = task_info

        thread = threading.Thread(
            target=self._run_async_extraction,
            args=(task_id, novel_title, max_chunks, extract_type),
            daemon=True
        )
        thread.start()

        return {
            "task_id": task_id,
            "status": "started",
            "message": f"异步提取任务已启动，请使用 get_task_status(task_id='{task_id}') 查询进度和结果",
            "novel": novel_title,
            "extract_type": extract_type,
            "max_chunks": max_chunks
        }

    def get_task_status(self, task_id: str) -> dict:
        with self._task_lock:
            if task_id not in self._background_tasks:
                return {"error": f"任务不存在: {task_id}", "hint": "请检查task_id是否正确"}
            task = self._background_tasks[task_id].copy()
        return task

    def _find_novel_id(self, novel_title: str) -> Optional[str]:
        """模糊匹配小说标题，返回 novel_id 或 None"""
        normalized = _normalize_title(novel_title)
        for nid, meta in self.novels.items():
            if _normalize_title(meta.title) == normalized:
                return nid
            if normalized in _normalize_title(meta.title):
                return nid
            if _normalize_title(meta.title) in normalized:
                return nid
        return None

    def novel_stats(self, novel_title: str) -> dict:
        novel_id = self._find_novel_id(novel_title)
        if not novel_id:
            available = [meta.title for meta in self.novels.values()]
            return {"error": f"未找到小说: {novel_title}", "available_novels": available}
        meta = self.novels[novel_id]
        chunks = self.collection.get(
            where={"novel_id": novel_id},
            include=["documents", "metadatas"],
            limit=10
        )
        total_chunks_for_novel = 0
        total_chars = 0
        chapter_set = set()
        chapter_stats = {}  # chapter_title -> {chunk_count, char_count}
        dialogue_chars = 0
        if chunks and chunks["documents"]:
            all_chunks = self.collection.get(
                where={"novel_id": novel_id},
                include=["documents", "metadatas"]
            )
            if all_chunks and all_chunks["documents"]:
                total_chunks_for_novel = len(all_chunks["documents"])
                for doc, m in zip(all_chunks["documents"], all_chunks["metadatas"] or []):
                    total_chars += len(doc)
                    for dre in [re.compile(r'\u201c(.+?)\u201d'), re.compile(r'\u300c(.+?)\u300d'), re.compile(r'"(.+?)"')]:
                        for dm in dre.finditer(doc):
                            dialogue_chars += len(dm.group(1))
                    ch = m.get("chapter_title", "")
                    if ch:
                        chapter_set.add(ch)
                        if ch not in chapter_stats:
                            chapter_stats[ch] = {"chunk_count": 0, "char_count": 0}
                        chapter_stats[ch]["chunk_count"] += 1
                        chapter_stats[ch]["char_count"] += len(doc)
        avg_chunk_len = total_chars / total_chunks_for_novel if total_chunks_for_novel else 0
        dialogue_ratio = dialogue_chars / total_chars if total_chars else 0
        return {
            "title": meta.title,
            "author": meta.author,
            "genre": meta.genre,
            "word_count": meta.word_count,
            "chunk_count": total_chunks_for_novel,
            "chapter_count": len(chapter_set),
            "avg_chunk_length": round(avg_chunk_len, 0),
            "dialogue_ratio": round(dialogue_ratio, 3),
            "chapters": sorted(list(chapter_set))[:50],
            "chapter_details": {ch: {"chunks": s["chunk_count"], "chars": s["char_count"]}
                                for ch, s in sorted(chapter_stats.items())[:50]}
        }

    def list_novels(self) -> list[dict]:
        return [asdict(v) for v in self.novels.values()]

    def get_stats(self) -> dict:
        return {
            "total_novels": len(self.novels),
            "total_chunks": self.collection.count() if self.collection else 0,
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "total_plot_patterns": len(self.plot_patterns),
            "total_writing_templates": len(self.writing_templates),
            "style_profiles": len(self.style_profiles),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "bm25_ready": self._use_optimized_search and self._tantivy_index is not None or self.bm25 is not None,
            "bm25_corpus_size": len(self.bm25_corpus) if self.bm25_corpus else 0,
            "reranker_available": self.reranker is not None,
            "chat_available": self.chat is not None
        }


kb = WebNovelKnowledgeBase()


def _safe_tool(name: str, func, *args, **kwargs):
    """Wrap a kb method call with exception handling for MCP tools"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"MCP tool '{name}' failed: {e}", exc_info=True)
        return {"error": str(e), "tool": name}


@mcp.tool()
def ingest_novel(file_path: str, title: str, author: str, genre: str) -> dict:
    """导入一本网文到知识库。file_path为小说txt文件路径，title为书名，author为作者，genre为类型（如奇幻/悬疑/赛博朋克）"""
    return _safe_tool("ingest_novel", kb.ingest_novel, file_path, title, author, genre)


@mcp.tool()
def search(query: str, mode: str = "hybrid", n_results: int = 10,
           novel_filter: str = "", genre_filter: str = "",
           chapter_filter: str = "", alpha: float = 0.6,
           use_rerank: bool = False,
           output_format: str = "compact", max_content_length: int = 0,
           dedupe: bool = True) -> list:
    """统一检索接口。mode可选：semantic(语义), bm25(关键词), hybrid(混合), rerank(精排)。alpha控制hybrid模式语义权重。use_rerank对hybrid模式启用精排。output_format: raw/compact/clean(默认compact)。max_content_length每条字数上限(0=不限制)。dedupe去重(默认True)。"""
    return _safe_tool("unified_search", kb.unified_search,
        query, mode, n_results,
        novel_filter=novel_filter or None,
        genre_filter=genre_filter or None,
        chapter_filter=chapter_filter or None,
        alpha=alpha,
        use_rerank=use_rerank,
        output_format=output_format,
        max_content_length=max_content_length,
        dedupe=dedupe
    )


@mcp.tool()
def search_knowledge(query: str = "", knowledge_type: str = "plot_patterns",
                     n_results: int = 10, use_semantic: bool = True,
                     type_filter: str = "", source_novel: str = "",
                     output_format: str = "compact", max_content_length: int = 0,
                     dedupe: bool = True) -> list:
    """统一知识搜索。knowledge_type: plot_patterns(情节模式)或writing_templates(写法模板)。use_semantic=True用语义搜索，False用关键字过滤。output_format: raw/compact/clean(默认compact)。max_content_length每条字数上限(0=不限制)。dedupe去重(默认True)。"""
    return _safe_tool("search_knowledge", kb.search_knowledge,
        query, knowledge_type, n_results, use_semantic,
        type_filter=type_filter or None,
        source_novel=source_novel or None,
        output_format=output_format,
        max_content_length=max_content_length,
        dedupe=dedupe
    )


@mcp.tool()
def analyze_style(novel_title: str) -> dict:
    """分析指定小说的写作风格，包括分段节奏变化（开篇/发展/高潮/收尾的句长和对话比）、原文幽默场景提取、叙事视角、章节钩子密度等"""
    return _safe_tool("analyze_style", kb.analyze_style, novel_title)


@mcp.tool()
def compare_styles(novel_titles: str) -> dict:
    """对比多本小说的写作风格。novel_titles为逗号分隔的书名列表，如'隐秘死角,没钱修什么仙'"""
    titles = [t.strip() for t in novel_titles.split(",") if t.strip()]
    return _safe_tool("compare_styles", kb.compare_styles, titles)


@mcp.tool()
def novel_stats(novel_title: str) -> dict:
    """获取单本小说的细粒度统计：章节数、平均分块长度、对话占比、章节列表等"""
    return _safe_tool("novel_stats", kb.novel_stats, novel_title)


@mcp.tool()
def search_entities(query: str, n_results: int = 10,
                    entity_type: str = "", source_novel: str = "",
                    output_format: str = "compact", max_content_length: int = 0,
                    dedupe: bool = True) -> list:
    """语义搜索实体。用自然语言描述查找相关角色、地点、组织等，如'反派角色'、'修炼功法'。output_format: raw/compact/clean(默认compact)。"""
    raw = _safe_tool("search_entities_semantic", kb.search_entities_semantic,
        query, n_results,
        entity_type=entity_type or None,
        source_novel=source_novel or None
    )
    if isinstance(raw, list) and not (len(raw) == 1 and isinstance(raw[0], dict) and "error" in raw[0]):
        return kb._format_search_results(raw, output_format, max_content_length, dedupe)
    return raw


@mcp.tool()
def get_entity_relations(entity_name: str, source_novel: str = "") -> list[dict]:
    """查询实体的所有关系。返回该实体的入边和出边关系"""
    return _safe_tool("get_entity_relations", kb.get_entity_relations, entity_name, source_novel or None)


@mcp.tool()
def extract(novel_title: str, extract_type: str = "plot_patterns",
            max_chunks: int = 20, async_mode: bool = False) -> dict:
    """统一提取接口。extract_type可选：plot_patterns(情节模式), entities(实体), writing_templates(写法模板), scene_patterns(场景模式)。async_mode=True时后台运行返回task_id"""
    if async_mode:
        return _safe_tool("start_async_extraction", kb.start_async_extraction, novel_title, max_chunks, extract_type)
    else:
        if extract_type == "plot_patterns":
            return _safe_tool("_do_extract_plot_patterns", kb._do_extract_plot_patterns, novel_title, max_chunks)
        elif extract_type == "entities":
            return _safe_tool("auto_extract_entities", kb.auto_extract_entities, novel_title, max_chunks)
        elif extract_type == "writing_templates":
            return _safe_tool("auto_extract_writing_templates", kb.auto_extract_writing_templates, novel_title, max_chunks)
        elif extract_type == "scene_patterns":
            return _safe_tool("auto_extract_scene_patterns", kb.auto_extract_scene_patterns, novel_title, max_chunks)
        else:
            return {"error": f"未知提取类型: {extract_type}", "hint": "可选：plot_patterns, entities, writing_templates, scene_patterns"}


@mcp.tool()
def get_task_status(task_id: str) -> dict:
    """查询异步任务状态和结果。task_id从extract(async_mode=True)返回"""
    return _safe_tool("get_task_status", kb.get_task_status, task_id)


@mcp.tool()
def list_novels() -> list[dict]:
    """列出知识库中所有已导入的小说"""
    return _safe_tool("list_novels", kb.list_novels)


@mcp.tool()
def get_stats() -> dict:
    """获取知识库统计信息：小说数量、分块数、实体数、关系数、情节模式数等"""
    return _safe_tool("get_stats", kb.get_stats)
