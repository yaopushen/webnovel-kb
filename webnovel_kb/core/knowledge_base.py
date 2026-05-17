"""Core Knowledge Base class that coordinates all modules."""
import hashlib
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import asdict

import chromadb
import networkx as nx

from webnovel_kb.config import (
    DATA_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_CHAT_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_EMBEDDING_DIMENSIONS,
    LLM_RERANK_MODEL, LLM_CHAT_MODEL, EMBEDDING_CACHE_PATH,
    QUERY_CACHE_SIZE, QUERY_CACHE_TTL,
)
from webnovel_kb.data_models import (
    NovelMeta, StyleProfile, PlotPattern,
    Entity, Relationship, WritingTemplate
)

from webnovel_kb.core.chunker import TextChunker
from webnovel_kb.core.state import StateManager
from webnovel_kb.core.indexer import IndexManager
from webnovel_kb.search.semantic import SemanticSearch
from webnovel_kb.search.bm25_search import BM25Search
from webnovel_kb.search.hybrid import HybridSearch
from webnovel_kb.search.rerank import RerankSearch
from webnovel_kb.search.unified import UnifiedSearch
from webnovel_kb.extraction.entities import EntityExtractor
from webnovel_kb.extraction.plot_patterns import PlotPatternExtractor
from webnovel_kb.extraction.writing_templates import WritingTemplateExtractor
from webnovel_kb.extraction.scene_patterns import ScenePatternExtractor
from webnovel_kb.analysis.style import StyleAnalyzer
from webnovel_kb.analysis.humor import HumorExtractor
from webnovel_kb.api.clients import RemoteEmbeddingFunction, RemoteReranker, RemoteChatClient
from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.exceptions import IngestError, SearchError, ExtractionError
from webnovel_kb.utils.query_cache import QueryCache

logger = get_logger("core.knowledge_base")


class WebNovelKnowledgeBase:
    """网文知识库核心类，协调各模块工作。"""

    def __init__(self, data_dir: Path = None, use_reranker: bool = False):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.novels: Dict[str, NovelMeta] = {}
        self.style_profiles: Dict[str, StyleProfile] = {}
        self.plot_patterns: List[PlotPattern] = []
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.writing_templates: List[WritingTemplate] = []
        self.graph = nx.DiGraph()

        self._query_cache = QueryCache(max_size=QUERY_CACHE_SIZE, ttl_seconds=QUERY_CACHE_TTL)

        self._setup_apis(use_reranker)
        self._setup_database()
        self._setup_modules()
        self._load_state()
        self._setup_indexes()
        self._init_background_tasks()

        self._async_tasks: Dict[str, dict] = {}
        self._task_lock = threading.Lock()

    def _setup_apis(self, use_reranker: bool):
        """设置 API 客户端。"""
        self.embedding_fn = None
        self.reranker = None
        self.chat = None

        if LLM_API_KEY:
            self.embedding_fn = RemoteEmbeddingFunction(
                api_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                model=LLM_EMBEDDING_MODEL,
                dimensions=LLM_EMBEDDING_DIMENSIONS,
                cache_path=EMBEDDING_CACHE_PATH
            )

            if LLM_CHAT_BASE_URL and LLM_CHAT_MODEL:
                self.chat = RemoteChatClient(
                    api_url=LLM_CHAT_BASE_URL,
                    api_key=LLM_API_KEY,
                    model=LLM_CHAT_MODEL
                )

            if use_reranker and LLM_RERANK_MODEL:
                self.reranker = RemoteReranker(
                    api_url=LLM_BASE_URL,
                    api_key=LLM_API_KEY,
                    model=LLM_RERANK_MODEL
                )

    def _setup_database(self):
        """设置 ChromaDB 数据库。"""
        chroma_dir = self.data_dir / "chroma_db"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))

        self.collection = self.client.get_or_create_collection(
            name="webnovel_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        self.patterns_collection = self.client.get_or_create_collection(
            name="plot_patterns"
        )
        self.entities_collection = self.client.get_or_create_collection(
            name="entities"
        )

    def _setup_modules(self):
        """设置各功能模块。"""
        self.chunker = TextChunker()
        self.state_manager = StateManager(self.data_dir)
        self.index_manager = IndexManager(
            self.data_dir, self.collection, self.patterns_collection,
            self.entities_collection, self.embedding_fn
        )

        self.semantic_search = SemanticSearch(self.collection, self.embedding_fn)
        self.bm25_search = BM25Search(self.index_manager, self.collection, self.novels)
        self.hybrid_search = HybridSearch(
            self.index_manager, self.semantic_search, self.bm25_search,
            self.embedding_fn
        )
        self.rerank_search = RerankSearch(self.reranker, self.hybrid_search)
        self.unified_search = UnifiedSearch(
            self.semantic_search, self.bm25_search, self.hybrid_search,
            self.rerank_search, self.reranker, self._query_cache
        )

        self.entity_extractor = EntityExtractor(
            self.chat, self.collection, self.entities, self.relationships,
            self.graph, self._add_entity, self._add_relationship,
            self._save_state, self.entities_collection
        )
        self.plot_extractor = PlotPatternExtractor(
            self.chat, self.collection, self.plot_patterns,
            self._add_plot_pattern, self._save_state
        )
        self.template_extractor = WritingTemplateExtractor(
            self.chat, self.collection, self.writing_templates,
            self._add_writing_template, self._save_state
        )
        self.scene_extractor = ScenePatternExtractor(
            self.chat, self.collection, self._add_plot_pattern
        )

        self.humor_extractor = HumorExtractor(self.chat)
        self.style_analyzer = StyleAnalyzer(
            self.chat, self.collection, self.style_profiles, self._save_state
        )

    def _load_state(self):
        """加载持久化状态。"""
        self.state_manager.load_all(
            self.novels, self.style_profiles, self.plot_patterns,
            self.entities, self.relationships, self.writing_templates,
            self.graph
        )
        logger.info(f"Loaded {len(self.novels)} novels, {len(self.plot_patterns)} patterns, "
                    f"{len(self.entities)} entities, {len(self.relationships)} relationships")

    def _save_state(self):
        """保存状态。"""
        self.state_manager.save_all(
            self.novels, self.style_profiles, self.plot_patterns,
            self.entities, self.relationships, self.writing_templates,
            self.graph
        )

    def _setup_indexes(self):
        """设置索引。"""
        need_rebuild = self.index_manager.init_optimized_search()

        total = self.collection.count()
        if total > 0:
            logger.info(f"ChromaDB contains {total} chunks")

            if need_rebuild or not (self.data_dir / "faiss_index.faiss").exists():
                logger.info("Building optimized indexes...")
                self.index_manager.build_all_indexes(self.novels)

            if not self.index_manager.bm25_corpus:
                self._rebuild_bm25_from_chroma()

            self.index_manager.index_plot_patterns(self.plot_patterns)
            self.index_manager.index_entities(self.entities)

        self.index_manager.preload_bm25_background()

    def _init_background_tasks(self):
        """初始化后台任务。"""
        pass

    def ingest_novel(self, file_path: str, title: str, author: str, genre: str) -> dict:
        """导入小说。"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise IngestError(f"文件不存在: {file_path}")

            text = path.read_text(encoding="utf-8")
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            existing = self.collection.get(
                where={"title": title},
                include=[]
            )
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])

            novel_id = hashlib.md5(f"{title}_{author}".encode()).hexdigest()[:12]
            self.novels[novel_id] = NovelMeta(
                title=title, author=author, genre=genre,
                word_count=len(text), chapter_count=0
            )

            chunks = self.chunker.chunk(text)

            chunk_ids = []
            documents = []
            metadatas = []
            embeddings = []

            batch_size = 50
            for i, (chunk_text, chapter_title) in enumerate(chunks):
                chunk_id = f"{novel_id}_{i}"
                chunk_ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append({
                    "novel_id": novel_id,
                    "title": title,
                    "author": author,
                    "genre": genre,
                    "chunk_index": i,
                    "chapter_title": chapter_title
                })

                if len(chunk_ids) >= batch_size:
                    batch_emb = self.embedding_fn(documents) if self.embedding_fn else None
                    if batch_emb:
                        embeddings.extend(batch_emb)
                    self.collection.add(
                        ids=chunk_ids, documents=documents,
                        metadatas=metadatas, embeddings=embeddings
                    )
                    chunk_ids, documents, metadatas, embeddings = [], [], [], []

            if chunk_ids:
                batch_emb = self.embedding_fn(documents) if self.embedding_fn else None
                self.collection.add(
                    ids=chunk_ids, documents=documents,
                    metadatas=metadatas, embeddings=batch_emb
                )

            self.novels[novel_id].chapter_count = len(set(c[1] for c in chunks if c[1]))
            self._save_state()

            self._query_cache.clear()

            self._rebuild_bm25_from_chroma()
            self.index_manager.build_all_indexes(self.novels)

            return {
                "status": "success",
                "novel_id": novel_id,
                "title": title,
                "chunks_indexed": len(chunks),
                "word_count": len(text)
            }
        except IngestError:
            raise
        except Exception as e:
            logger.error(f"Failed to ingest novel: {e}", exc_info=True)
            raise IngestError(f"导入小说失败: {e}", detail=str(e))

    def _rebuild_bm25_from_chroma(self):
        """从 ChromaDB 重建 BM25。"""
        self.index_manager._rebuild_bm25_from_chroma()
        self.index_manager.bm25_corpus = self.index_manager.bm25_corpus
        self.index_manager.bm25_metadata = self.index_manager.bm25_metadata

    def search(self, query: str, mode: str = "hybrid", n_results: int = 10,
               novel_filter: Optional[str] = None, genre_filter: Optional[str] = None,
               chapter_filter: Optional[str] = None, alpha: float = 0.6,
               use_rerank: bool = False, output_format: str = "compact",
               max_content_length: int = 0, dedupe: bool = True) -> List:
        """统一搜索。"""
        return self.unified_search.search(
            query, mode, n_results, novel_filter, genre_filter,
            chapter_filter, alpha, use_rerank, output_format,
            max_content_length, dedupe
        )

    def search_knowledge(self, query: str = "", knowledge_type: str = "plot_patterns",
                         n_results: int = 10, use_semantic: bool = True,
                         type_filter: Optional[str] = None,
                         source_novel: Optional[str] = None,
                         output_format: str = "compact",
                         max_content_length: int = 0,
                         dedupe: bool = True) -> List[dict]:
        """搜索知识库。"""
        from webnovel_kb.utils.dedupe import dedupe_results
        from webnovel_kb.utils.format import format_search_results

        cache_key = self._query_cache.make_key(
            query, knowledge_type=knowledge_type, n_results=n_results,
            use_semantic=use_semantic, type_filter=type_filter,
            source_novel=source_novel, output_format=output_format,
            max_content_length=max_content_length, dedupe=dedupe
        )
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        collection = self.patterns_collection if knowledge_type == "plot_patterns" else self.entities_collection
        data_list = self.plot_patterns if knowledge_type == "plot_patterns" else self.writing_templates

        if use_semantic and query:
            where = {}
            if source_novel:
                where["source_novel"] = source_novel
            if type_filter:
                where["pattern_type"] = type_filter

            query_params = {"n_results": n_results}
            if where:
                query_params["where"] = where

            if self.embedding_fn:
                try:
                    query_vec = self.embedding_fn([query])[0]
                    query_params["query_embeddings"] = [query_vec]
                except Exception as e:
                    logger.warning(f"Knowledge embedding failed, falling back to query_texts: {e}")
                    query_params["query_texts"] = [query]
            else:
                query_params["query_texts"] = [query]

            results = collection.query(**query_params)

            raw = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results["metadatas"] else {}
                    raw.append({
                        "text": doc,
                        "metadata": meta,
                        "source": meta.get("source_novel", "")
                    })
            result = format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)
        else:
            raw = []
            for item in data_list:
                if source_novel and item.source_novel != source_novel:
                    continue
                if type_filter and item.pattern_type != type_filter:
                    continue
                if query and query.lower() not in str(asdict(item)).lower():
                    continue
                raw.append({
                    "text": f"{item.pattern_type}: {item.description}",
                    "metadata": asdict(item),
                    "source": item.source_novel
                })
                if len(raw) >= n_results:
                    break
            result = format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)

        self._query_cache.put(cache_key, result)
        return result

    def analyze_style(self, novel_title: str) -> dict:
        """分析风格。"""
        exact_title = None
        for n in self.novels.values():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                break

        if not exact_title:
            raise SearchError(f"未找到小说: {novel_title}")

        novel_id = None
        for nid, n in self.novels.items():
            if n.title == exact_title:
                novel_id = nid
                break

        return self.style_analyzer.analyze(novel_title, novel_id, exact_title, self.humor_extractor)

    def compare_styles(self, novel_titles) -> dict:
        """对比风格。"""
        if isinstance(novel_titles, str):
            titles = [t.strip() for t in novel_titles.split(",") if t.strip()]
        else:
            titles = novel_titles
        if len(titles) < 2:
            raise SearchError("需要至少两本小说进行对比", detail=f"收到 {len(titles)} 本")

        results = {}
        for title in titles:
            if title in self.style_profiles:
                results[title] = asdict(self.style_profiles[title])
            else:
                results[title] = self.analyze_style(title)

        return {
            "novels": titles,
            "comparison": results,
            "summary": self._generate_style_comparison_summary(results)
        }

    def _generate_style_comparison_summary(self, results: dict) -> str:
        """生成风格对比摘要。"""
        summaries = []
        for title, profile in results.items():
            if "error" not in profile:
                summaries.append(
                    f"《{title}》: 平均句长{profile.get('avg_sentence_len', 0)}字, "
                    f"对话占比{profile.get('dialogue_ratio', 0)*100:.1f}%, "
                    f"节奏类型:{profile.get('pace_type', '未知')}"
                )
        return "\n".join(summaries)

    def novel_stats(self, novel_title: str) -> dict:
        """小说统计。"""
        exact_title = None
        novel_id = None
        for nid, n in self.novels.items():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                novel_id = nid
                break

        if not exact_title:
            raise SearchError(f"未找到小说: {novel_title}")

        meta = self.novels[novel_id]
        chunks = self.collection.get(
            where={"title": exact_title},
            include=["documents"]
        )
        chunk_count = len(chunks["ids"]) if chunks else 0

        has_style = novel_title in self.style_profiles
        entity_count = sum(1 for e in self.entities.values() if e.source_novel == exact_title)
        pattern_count = sum(1 for p in self.plot_patterns if p.source_novel == exact_title)

        return {
            "title": exact_title,
            "author": meta.author,
            "genre": meta.genre,
            "word_count": meta.word_count,
            "chunk_count": meta.chunk_count,
            "chunks_indexed": chunk_count,
            "has_style_analysis": has_style,
            "entities_extracted": entity_count,
            "patterns_extracted": pattern_count
        }

    def list_novels(self) -> List[dict]:
        """列出所有小说。"""
        return [asdict(n) for n in self.novels.values()]

    def get_stats(self) -> dict:
        """获取统计信息。"""
        stats = {
            "total_novels": len(self.novels),
            "total_chunks": self.collection.count(),
            "total_patterns": len(self.plot_patterns),
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "total_templates": len(self.writing_templates),
            "bm25_ready": self.index_manager.bm25_ready,
            "optimized_search": self.index_manager.use_optimized_search,
        }
        stats["query_cache"] = self._query_cache.stats()
        return stats

    def search_entities(self, entity_type: Optional[str] = None,
                        name: Optional[str] = None,
                        source_novel: Optional[str] = None,
                        n_results: int = 10) -> List[dict]:
        """搜索实体。"""
        results = []
        for eid, e in self.entities.items():
            if entity_type and e.entity_type != entity_type:
                continue
            if name and name.lower() not in e.name.lower():
                continue
            if source_novel and e.source_novel != source_novel:
                continue
            results.append(asdict(e))
            if len(results) >= n_results:
                break
        return results

    def search_entities_semantic(self, query: str, n_results: int = 10,
                                 entity_type: Optional[str] = None,
                                 source_novel: Optional[str] = None) -> list[dict]:
        """语义搜索实体。"""
        cache_key = self._query_cache.make_key(
            f"entity:{query}", entity_type=entity_type,
            source_novel=source_novel, n_results=n_results
        )
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.entities_collection.count() == 0:
            if not self.entities:
                return [{"status": "no_entities_indexed", "hint": "请先提取实体"}]
            output = []
            for eid, e in self.entities.items():
                if entity_type and e.entity_type != entity_type:
                    continue
                if source_novel and e.source_novel != source_novel:
                    continue
                if query and query.lower() not in f"{e.name} {e.description}".lower():
                    continue
                output.append(asdict(e))
                if len(output) >= n_results:
                    break
            return output

        where_filter = {}
        if entity_type:
            where_filter["entity_type"] = entity_type
        if source_novel:
            where_filter["source_novel"] = source_novel

        query_params = {"n_results": n_results}
        if where_filter:
            query_params["where"] = where_filter

        if self.embedding_fn:
            try:
                query_vec = self.embedding_fn([query])[0]
                query_params["query_embeddings"] = [query_vec]
            except Exception as e:
                logger.warning(f"Entity embedding failed, falling back to query_texts: {e}")
                query_params["query_texts"] = [query]
        else:
            query_params["query_texts"] = [query]

        try:
            results = self.entities_collection.query(**query_params)
        except Exception as e:
            logger.warning(f"Semantic entity search failed: {e}")
            return [{"status": "semantic_unavailable", "error": str(e)[:100],
                     "hint": "Embedding dimension mismatch. Use keyword search instead."}]
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
                else:
                    output.append({
                        "text": doc,
                        "metadata": meta,
                        "source": meta.get("source_novel", "")
                    })
        if not output:
            return [{"status": "no_results", "query": query}]

        self._query_cache.put(cache_key, output)
        return output

    def _format_search_results(self, raw_results: list, output_format: str = "compact",
                               max_content_length: int = 0, dedupe: bool = True) -> list:
        """格式化搜索结果。"""
        from webnovel_kb.utils.dedupe import dedupe_results
        from webnovel_kb.utils.format import format_search_results
        return format_search_results(raw_results, output_format, max_content_length, dedupe, dedupe_results)

    def get_entity_relations(self, entity_name: str, source_novel: Optional[str] = None) -> List[dict]:
        """获取实体关系。"""
        results = []
        for r in self.relationships:
            if r.source == entity_name or r.target == entity_name:
                if source_novel and r.source_novel != source_novel:
                    continue
                results.append(asdict(r))
        return results

    def start_async_extraction(self, novel_title: str, max_chunks: int = 20,
                               extract_type: str = "plot_patterns") -> dict:
        """启动异步提取任务。"""
        import threading
        import uuid

        exact_title = None
        novel_id = None
        for nid, n in self.novels.items():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                novel_id = nid
                break

        if not exact_title:
            raise ExtractionError(f"未找到小说: {novel_title}")

        task_id = str(uuid.uuid4())[:8]
        self._async_tasks[task_id] = {
            "status": "running",
            "novel": exact_title,
            "extract_type": extract_type,
            "progress": 0,
            "result": None
        }

        def _run():
            try:
                with self._task_lock:
                    self._async_tasks[task_id]["progress"] = 10

                def progress_cb(current, total):
                    pct = 10 + int(current / total * 80)
                    with self._task_lock:
                        if task_id in self._async_tasks:
                            self._async_tasks[task_id]["progress"] = min(pct, 90)

                if extract_type == "plot_patterns":
                    result = self.plot_extractor.extract_cross_chunk(
                        exact_title, novel_id, exact_title, max_chunks, progress_cb
                    )
                elif extract_type == "entities":
                    result = self.entity_extractor.extract_cross_chunk(
                        exact_title, novel_id, exact_title, max_chunks, progress_cb
                    )
                else:
                    result = self.extract(exact_title, extract_type, max_chunks, False)

                with self._task_lock:
                    if task_id in self._async_tasks:
                        self._async_tasks[task_id]["status"] = "completed"
                        self._async_tasks[task_id]["progress"] = 100
                        self._async_tasks[task_id]["result"] = result

                self._query_cache.clear()
            except Exception as e:
                with self._task_lock:
                    if task_id in self._async_tasks:
                        self._async_tasks[task_id]["status"] = "error"
                        self._async_tasks[task_id]["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"task_id": task_id, "status": "started", "novel": exact_title}

    def extract(self, novel_title: str, extract_type: str = "all",
                max_chunks: int = 20, cross_chunk: bool = False) -> dict:
        """提取知识。"""
        exact_title = None
        novel_id = None
        for nid, n in self.novels.items():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                novel_id = nid
                break

        if not exact_title:
            raise ExtractionError(f"未找到小说: {novel_title}")

        results = {"novel": exact_title}

        if extract_type in ["entities", "all"]:
            if cross_chunk:
                results["entities"] = self.entity_extractor.extract_cross_chunk(
                    novel_title, novel_id, exact_title, max_chunks
                )
            else:
                results["entities"] = self.entity_extractor.extract(
                    novel_title, novel_id, exact_title, max_chunks
                )

        if extract_type in ["plot_patterns", "all"]:
            if cross_chunk:
                results["plot_patterns"] = self.plot_extractor.extract_cross_chunk(
                    novel_title, novel_id, exact_title, max_chunks
                )
            else:
                results["plot_patterns"] = self.plot_extractor.extract(
                    novel_title, novel_id, exact_title, max_chunks
                )

        if extract_type in ["writing_templates", "all"]:
            results["writing_templates"] = self.template_extractor.extract(
                novel_title, novel_id, exact_title, max_chunks
            )

        if extract_type in ["scene_patterns", "all"]:
            results["scene_patterns"] = self.scene_extractor.extract(
                novel_title, novel_id, exact_title, max_chunks
            )

        self._query_cache.clear()
        return results

    def get_task_status(self, task_id: str) -> dict:
        """获取任务状态。"""
        with self._task_lock:
            return self._async_tasks.get(task_id, {"error": f"任务不存在: {task_id}"})

    def read_chapter(self, novel_title: str, chapter: int = 1) -> dict:
        """读取指定章节的完整正文。chapter为章节序号(1-based)。"""
        exact_title, novel_id = self._resolve_novel(novel_title)
        if not exact_title:
            return {"error": f"未找到小说: {novel_title}"}

        cn_num_map = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
                      6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
        cn = cn_num_map.get(chapter, str(chapter))
        patterns = [f"第{chapter}章", f"第{cn}章"]

        if chapter == 0:
            patterns = [""]

        all_chunks = []
        for pat in patterns:
            try:
                result = self.collection.get(
                    where={"$and": [{"title": exact_title}, {"chapter_title": pat}]},
                    include=["documents", "metadatas"]
                )
                if result and result.get("ids"):
                    for j in range(len(result["ids"])):
                        meta = result["metadatas"][j]
                        all_chunks.append((
                            meta.get("chunk_index", 0),
                            result["documents"][j],
                            meta.get("chapter_title", "")
                        ))
            except Exception as e:
                logger.warning(f"ChromaDB query for chapter '{pat}' failed: {e}")

        if not all_chunks:
            return {
                "error": f"未找到第{chapter}章",
                "novel": exact_title,
                "hint": "尝试使用 list_chapters 查看可用章节"
            }

        all_chunks.sort(key=lambda x: x[0])
        full_text = "\n".join(doc for _, doc, _ in all_chunks)

        return {
            "novel": exact_title,
            "chapter_number": chapter,
            "chapter_title": all_chunks[0][2],
            "content": full_text,
            "word_count": len(full_text),
            "chunk_count": len(all_chunks)
        }

    def list_chapters(self, novel_title: str) -> dict:
        """列出小说的所有章节标题及序号。"""
        exact_title, novel_id = self._resolve_novel(novel_title)
        if not exact_title:
            return {"error": f"未找到小说: {novel_title}"}

        result = self.collection.get(
            where={"title": exact_title},
            include=["metadatas"]
        )

        if not result or not result.get("ids"):
            return {"error": f"小说 {exact_title} 没有已索引的内容"}

        chapter_map = {}
        for j in range(len(result["metadatas"])):
            meta = result["metadatas"][j]
            ct = meta.get("chapter_title", "")
            ci = meta.get("chunk_index", 0)
            if ct not in chapter_map:
                chapter_map[ct] = {"first_chunk": ci, "last_chunk": ci, "chunk_count": 0}
            entry = chapter_map[ct]
            entry["last_chunk"] = max(entry["last_chunk"], ci)
            entry["first_chunk"] = min(entry["first_chunk"], ci)
            entry["chunk_count"] += 1

        import re
        chapters = []
        for title, info in sorted(chapter_map.items(), key=lambda x: x[1]["first_chunk"]):
            num_match = re.search(r'第(\d+)章', title)
            num = int(num_match.group(1)) if num_match else -1
            chapters.append({
                "number": num,
                "title": title,
                "first_chunk": info["first_chunk"],
                "last_chunk": info["last_chunk"],
                "chunk_count": info["chunk_count"]
            })

        return {
            "novel": exact_title,
            "total_chapters": len(chapters),
            "chapters": chapters
        }

    def _resolve_novel(self, novel_title: str) -> tuple:
        """解析小说标题，返回 (exact_title, novel_id)。"""
        exact_title = None
        novel_id = None
        for nid, n in self.novels.items():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                novel_id = nid
                break
        return exact_title, novel_id

    def _add_entity(self, name: str, entity_type: str, description: str,
                    source_novel: str, role: str = "", first_appearance: str = "",
                    arc: str = "") -> dict:
        """添加实体。"""
        eid = hashlib.md5(f"{name}_{source_novel}".encode()).hexdigest()[:12]
        entity = Entity(
            entity_id=eid,
            name=name,
            entity_type=entity_type,
            description=description,
            source_novel=source_novel,
            role=role,
            first_appearance=first_appearance,
            arc=arc
        )
        self.entities[eid] = entity
        self.graph.add_node(name, **asdict(entity))
        self._save_state()
        return asdict(entity)

    def _add_relationship(self, source: str, target: str, relation_type: str,
                          description: str, source_novel: str) -> dict:
        """添加关系。"""
        rel = Relationship(
            source=source,
            target=target,
            relation_type=relation_type,
            description=description,
            source_novel=source_novel
        )
        self.relationships.append(rel)
        self.graph.add_edge(source, target, **asdict(rel))
        self._save_state()
        return asdict(rel)

    def _add_plot_pattern(self, pattern_type: str, description: str,
                          source_novel: str, source_chapter: str,
                          pattern_text: str = "", before_context: str = "",
                          after_context: str = "", effectiveness: str = "") -> dict:
        """添加情节模式。"""
        pattern = PlotPattern(
            pattern_type=pattern_type,
            description=description,
            source_novel=source_novel,
            source_chapter=source_chapter,
            pattern_text=pattern_text,
            before_context=before_context,
            after_context=after_context,
            effectiveness=effectiveness
        )
        self.plot_patterns.append(pattern)
        self._save_state()
        return asdict(pattern)

    def _add_writing_template(self, template_type: str, scene_type: str,
                              structure: str, key_beats: List[str],
                              source_novel: str, source_chapter: str,
                              example_text: str = "", effectiveness: str = "") -> dict:
        """添加写作模板。"""
        template = WritingTemplate(
            template_type=template_type,
            scene_type=scene_type,
            structure=structure,
            key_beats=key_beats,
            source_novel=source_novel,
            source_chapter=source_chapter,
            example_text=example_text,
            effectiveness=effectiveness
        )
        self.writing_templates.append(template)
        self._save_state()
        return asdict(template)
