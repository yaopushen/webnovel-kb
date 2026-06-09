"""Core Knowledge Base class — 门面模式，协调各子模块。"""
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
from dataclasses import asdict

import chromadb
import networkx as nx

from webnovel_kb.config import (
    DATA_DIR, LLM_API_KEY, LLM_BASE_URL, LLM_CHAT_BASE_URL,
    LLM_EMBEDDING_MODEL, LLM_EMBEDDING_DIMENSIONS,
    LLM_RERANK_MODEL, LLM_CHAT_MODEL, EMBEDDING_CACHE_PATH,
    QUERY_CACHE_SIZE, QUERY_CACHE_TTL,
    ZHIHU_ACCESS_SECRET, ZHIHU_SEARCH_URL, ZHIHU_GLOBAL_SEARCH_URL, ZHIHU_ZHIDA_URL,
)
from webnovel_kb.data_models import (
    NovelMeta, StyleProfile, PlotPattern,
    Entity, Relationship, WritingTemplate, ChapterOutline,
)

from webnovel_kb.utils.chinese_numbers import int_to_cn
from webnovel_kb.core.chunker import TextChunker
from webnovel_kb.core.state import StateManager
from webnovel_kb.core.indexer import IndexManager
from webnovel_kb.search.semantic import SemanticSearch
from webnovel_kb.search.bm25_search import BM25Search
from webnovel_kb.search.hybrid import HybridSearch
from webnovel_kb.search.rerank import RerankSearch
from webnovel_kb.search.unified import UnifiedSearch
from webnovel_kb.search.external import ExternalSearch
from webnovel_kb.search.smart import SmartSearchEngine
from webnovel_kb.extraction.entities import EntityExtractor
from webnovel_kb.extraction.plot_patterns import PlotPatternExtractor
from webnovel_kb.extraction.writing_templates import WritingTemplateExtractor
from webnovel_kb.extraction.scene_patterns import ScenePatternExtractor
from webnovel_kb.extraction.outlines import ChapterOutlineExtractor
from webnovel_kb.analysis.style import StyleAnalyzer
from webnovel_kb.analysis.humor import HumorExtractor
from webnovel_kb.api.clients import RemoteEmbeddingFunction, RemoteReranker, RemoteChatClient
from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.exceptions import IngestError, SearchError, ExtractionError
from webnovel_kb.utils.query_cache import QueryCache
from webnovel_kb.utils.format import clean_text
from webnovel_kb.utils.novel_resolver import resolve_novel, resolve_novel_title

logger = get_logger("core.knowledge_base")


class WebNovelKnowledgeBase:
    """网文知识库门面类——协调各子模块工作。"""

    def __init__(self, data_dir: Path = None, use_reranker: bool = False):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.novels: Dict[str, NovelMeta] = {}
        self.style_profiles: Dict[str, StyleProfile] = {}
        self.plot_patterns: List[PlotPattern] = []
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.writing_templates: List[WritingTemplate] = []
        self.chapter_outlines: Dict[str, ChapterOutline] = {}
        self.graph = nx.DiGraph()

        self._query_cache = QueryCache(max_size=QUERY_CACHE_SIZE, ttl_seconds=QUERY_CACHE_TTL)

        self._setup_apis(use_reranker)
        self._setup_database()
        self._setup_modules()
        self._load_state()
        self._setup_indexes()

    # ── 初始化 ──────────────────────────────────────────────

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
                from webnovel_kb.config import LLM_CHAT_API_KEY
                chat_key = LLM_CHAT_API_KEY or LLM_API_KEY
                self.chat = RemoteChatClient(
                    api_url=LLM_CHAT_BASE_URL,
                    api_key=chat_key,
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
            name="webnovel_chunks", metadata={"hnsw:space": "cosine"}
        )
        self.patterns_collection = self.client.get_or_create_collection(name="plot_patterns")
        self.entities_collection = self.client.get_or_create_collection(name="entities")
        self.outlines_collection = self.client.get_or_create_collection(name="chapter_outlines")
        self.agent_knowledge_collection = self.client.get_or_create_collection(name="agent_knowledge")

    def _setup_modules(self):
        """设置各功能模块。"""
        self.chunker = TextChunker()
        self.state_manager = StateManager(self.data_dir)
        self.index_manager = IndexManager(
            self.data_dir, self.collection, self.patterns_collection,
            self.entities_collection, self.embedding_fn
        )

        # 搜索模块
        self.semantic_search = SemanticSearch(self.collection, self.embedding_fn)
        self.bm25_search = BM25Search(self.index_manager, self.collection, self.novels)
        self.hybrid_search = HybridSearch(
            self.index_manager, self.semantic_search, self.bm25_search, self.embedding_fn
        )
        self.rerank_search = RerankSearch(self.reranker, self.hybrid_search)
        self.unified_search = UnifiedSearch(
            self.semantic_search, self.bm25_search, self.hybrid_search,
            self.rerank_search, self.reranker, self._query_cache
        )

        # 外部搜索
        self.external_search = ExternalSearch(
            access_secret=ZHIHU_ACCESS_SECRET,
            search_url=ZHIHU_SEARCH_URL,
            global_search_url=ZHIHU_GLOBAL_SEARCH_URL,
            zhida_url=ZHIHU_ZHIDA_URL,
        )

        # 提取器
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
        self.outline_extractor = ChapterOutlineExtractor(self.chat, self)

        # 分析器
        self.humor_extractor = HumorExtractor(self.chat)
        self.style_analyzer = StyleAnalyzer(
            self.chat, self.collection, self.style_profiles, self._save_state
        )

        # Agent 知识层
        from webnovel_kb.core.knowledge_store import KnowledgeStore
        self.knowledge_store = KnowledgeStore(
            self.agent_knowledge_collection, self.embedding_fn,
            self.chat, self.state_manager
        )

        # 文抄公
        from webnovel_kb.creation.sample_generator import SampleGenerator
        self.sample_generator = SampleGenerator(self.chat, self)

        # 章纲管理器
        from webnovel_kb.core.outline_manager import OutlineManager
        self.outline_manager = OutlineManager(
            outlines_collection=self.outlines_collection,
            embedding_fn=self.embedding_fn,
            outline_extractor=self.outline_extractor,
            chapter_outlines=self.chapter_outlines,
            novels=self.novels,
            save_state_fn=self._save_state
        )

        # 小说读取器
        from webnovel_kb.core.novel_reader import NovelReader
        self.novel_reader = NovelReader(self.collection, self.novels)

        # 任务管理器
        from webnovel_kb.core.task_manager import TaskManager
        self.task_manager = TaskManager(
            novels=self.novels,
            query_cache=self._query_cache,
            plot_extractor=self.plot_extractor,
            entity_extractor=self.entity_extractor,
            template_extractor=self.template_extractor,
            scene_extractor=self.scene_extractor
        )

        # 智能搜索引擎
        self.smart_engine = SmartSearchEngine(
            chat=self.chat,
            unified_search=self.unified_search,
            knowledge_store=self.knowledge_store,
            external_search=self.external_search,
            search_knowledge_fn=self.search_knowledge,
            resolve_fn=self.resolve_novel_title,
        )

    def _load_state(self):
        """加载持久化状态。"""
        self.state_manager.load_all(
            self.novels, self.style_profiles, self.plot_patterns,
            self.entities, self.relationships, self.writing_templates,
            self.graph, chapter_outlines=self.chapter_outlines
        )
        logger.info(f"Loaded {len(self.novels)} novels, {len(self.plot_patterns)} patterns, "
                    f"{len(self.entities)} entities, {len(self.relationships)} relationships")

    def _save_state(self):
        """保存状态。"""
        self.state_manager.save_all(
            self.novels, self.style_profiles, self.plot_patterns,
            self.entities, self.relationships, self.writing_templates,
            self.graph, chapter_outlines=self.chapter_outlines
        )

    def _setup_indexes(self):
        """设置索引。"""
        need_rebuild = self.index_manager.init_optimized_search()
        total = self.collection.count()
        if total > 0:
            logger.info(f"ChromaDB contains {total} chunks")
            if need_rebuild:
                logger.info("Building Tantivy index...")
                self.index_manager.build_all_indexes(self.novels)

    # ── 小说导入（保留，含唯一业务逻辑）──────────────────────

    def ingest_novel(self, file_path: str, title: str, author: str, genre: str) -> dict:
        """导入小说。"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise IngestError(f"文件不存在: {file_path}")

            text = path.read_text(encoding="utf-8")
            text = text.replace("\r\n", "\n").replace("\r", "\n")

            existing = self.collection.get(where={"title": title}, include=[])
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])

            novel_id = hashlib.md5(f"{title}_{author}".encode()).hexdigest()[:12]
            self.novels[novel_id] = NovelMeta(
                title=title, author=author, genre=genre,
                word_count=len(text), chapter_count=0
            )

            chunks = self.chunker.chunk(text)
            chunk_ids, documents, metadatas, embeddings = [], [], [], []

            batch_size = 50
            for i, (chunk_text, chapter_title) in enumerate(chunks):
                chunk_id = f"{novel_id}_{i}"
                chunk_ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append({
                    "novel_id": novel_id, "title": title, "author": author,
                    "genre": genre, "chunk_index": i, "chapter_title": chapter_title
                })

                if len(chunk_ids) >= batch_size:
                    batch_emb = self.embedding_fn(documents) if self.embedding_fn else None
                    if batch_emb:
                        embeddings.extend(batch_emb)
                    self.collection.add(ids=chunk_ids, documents=documents,
                                        metadatas=metadatas, embeddings=embeddings)
                    chunk_ids, documents, metadatas, embeddings = [], [], [], []

            if chunk_ids:
                batch_emb = self.embedding_fn(documents) if self.embedding_fn else None
                self.collection.add(ids=chunk_ids, documents=documents,
                                    metadatas=metadatas, embeddings=batch_emb)

            import re as _re
            chapter_titles = set(c[1] for c in chunks if c[1])
            self.novels[novel_id].chapter_count = len(chapter_titles)
            self.novels[novel_id].chunk_count = len(chunks)

            chapter_nums = []
            for ct in chapter_titles:
                m = _re.search(r'第(\d+)[章节回]', ct)
                if not m:
                    m = _re.match(r'^(\d+)[、.\s]', ct)
                if m:
                    chapter_nums.append(int(m.group(1)))
            if chapter_nums:
                self.novels[novel_id].first_chapter = min(chapter_nums)
                self.novels[novel_id].last_chapter = max(chapter_nums)

            self._save_state()
            self._query_cache.clear()
            self.index_manager.build_all_indexes(self.novels)

            return {"status": "success", "novel_id": novel_id, "title": title,
                    "chunks_indexed": len(chunks), "word_count": len(text)}
        except IngestError:
            raise
        except Exception as e:
            logger.error(f"Failed to ingest novel: {e}", exc_info=True)
            raise IngestError(f"导入小说失败: {e}", detail=str(e))

    # ── 搜索（保留，跨模块协调）─────────────────────────────

    async def search(self, query: str, mode: str = "hybrid", n_results: int = 10,
               novel_filter: Optional[str] = None, genre_filter: Optional[str] = None,
               chapter_filter: Optional[str] = None, alpha: float = 0.6,
               use_rerank: bool = False, output_format: str = "compact",
               max_content_length: int = 0, dedupe: bool = True) -> List:
        """统一搜索。"""
        return await self.unified_search.search(
            query, mode, n_results, novel_filter, genre_filter,
            chapter_filter, alpha, use_rerank, output_format,
            max_content_length, dedupe
        )

    async def search_with_scope(self, query: str, scope: str = "chunks",
                                n_results: int = 10, novel_filter: str = "",
                                genre_filter: str = "", mode: str = "hybrid",
                                alpha: float = 0.6, use_rerank: bool = False,
                                output_format: str = "compact",
                                max_content_length: int = 0, dedupe: bool = True) -> list:
        """带 scope 的统一搜索——chunks/outlines/agent_knowledge/all。"""
        if scope == "chunks":
            return await self.unified_search.search(
                query, mode, n_results, novel_filter or None, genre_filter or None,
                None, alpha, use_rerank, output_format, max_content_length, dedupe
            )

        if scope == "outlines":
            raw = await self.outline_manager.search_outlines(
                query, n_results=n_results, novel_filter=novel_filter,
                output_format="raw", max_content_length=max_content_length)
            from webnovel_kb.utils.dedupe import dedupe_results
            from webnovel_kb.utils.format import format_search_results
            return format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)

        if scope == "agent_knowledge":
            raw = await self.knowledge_store.search(query, n_results=n_results)
            for r in raw:
                r["_source_type"] = "knowledge"
            from webnovel_kb.utils.dedupe import dedupe_results
            from webnovel_kb.utils.format import format_search_results
            return format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)

        if scope == "all":
            chunk_task = self.unified_search.search(
                query, mode, n_results,
                novel_filter or None, genre_filter or None, None, alpha,
                use_rerank, "raw", max_content_length, False
            )
            outline_task = self.outline_manager.search_outlines(
                query, n_results=n_results, novel_filter=novel_filter, output_format="raw")
            knowledge_task = self.knowledge_store.search(query, n_results=n_results)

            chunk_results, outline_results, knowledge_results = await asyncio.gather(
                chunk_task, outline_task, knowledge_task, return_exceptions=True
            )

            all_results = []
            if isinstance(chunk_results, list):
                for r in chunk_results:
                    if isinstance(r, dict):
                        r["_source_type"] = "chunk"
                all_results.extend(chunk_results)
            if isinstance(outline_results, list):
                for r in outline_results:
                    if isinstance(r, dict):
                        r["_source_type"] = "outline"
                all_results.extend(outline_results)
            if isinstance(knowledge_results, list):
                for r in knowledge_results:
                    if isinstance(r, dict):
                        r["_source_type"] = "knowledge"
                all_results.extend(knowledge_results)

            from webnovel_kb.utils.dedupe import dedupe_results
            from webnovel_kb.utils.format import format_search_results
            return format_search_results(all_results, output_format, max_content_length, dedupe, dedupe_results)

        return []

    def stats_with_scope(self, scope: str = "global", novel_title: str = "") -> dict:
        """带 scope 的统计——global/novels/knowledge/书名。"""
        if scope == "global" and not novel_title:
            return self.get_stats()
        if scope == "novels":
            return {"novels": self.list_novels()}
        if scope == "knowledge":
            return {"knowledge": self.knowledge_store.list_all()}
        return self.novel_stats(novel_title or scope)

    # ── 知识搜索（保留，被 SmartSearchEngine 引用）──────────

    async def search_knowledge(self, query: str = "", knowledge_type: str = "plot_patterns",
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
                    query_vec = (await self.embedding_fn([query]))[0]
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
                    raw.append({"text": doc, "metadata": meta, "source": meta.get("source_novel", "")})
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
                    "metadata": asdict(item), "source": item.source_novel
                })
                if len(raw) >= n_results:
                    break
            result = format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)

        self._query_cache.put(cache_key, result)
        return result

    # ── 风格分析（保留）────────────────────────────────────

    async def analyze_style(self, novel_title: str) -> dict:
        """分析风格。"""
        exact_title, novel_id = resolve_novel(self.novels, novel_title)
        if not exact_title:
            raise SearchError(f"未找到小说: {novel_title}")
        return await self.style_analyzer.analyze(novel_title, novel_id, exact_title, self.humor_extractor)

    async def compare_styles(self, novel_titles) -> dict:
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
                results[title] = await self.analyze_style(title)

        return {"novels": titles, "comparison": results,
                "summary": self._generate_style_comparison_summary(results)}

    def _generate_style_comparison_summary(self, results: dict) -> str:
        summaries = []
        for title, profile in results.items():
            if "error" not in profile:
                summaries.append(
                    f"《{title}》: 平均句长{profile.get('avg_sentence_len', 0)}字, "
                    f"对话占比{profile.get('dialogue_ratio', 0)*100:.1f}%, "
                    f"节奏类型:{profile.get('pace_type', '未知')}"
                )
        return "\n".join(summaries)

    # ── 统计（保留）────────────────────────────────────────

    def novel_stats(self, novel_title: str) -> dict:
        """小说统计。"""
        exact_title, novel_id = resolve_novel(self.novels, novel_title)
        if not exact_title:
            raise SearchError(f"未找到小说: {novel_title}")

        meta = self.novels[novel_id]
        chunks = self.collection.get(where={"title": exact_title}, include=["documents"])
        chunk_count = len(chunks["ids"]) if chunks else 0

        has_style = novel_title in self.style_profiles
        entity_count = sum(1 for e in self.entities.values() if e.source_novel == exact_title)
        pattern_count = sum(1 for p in self.plot_patterns if p.source_novel == exact_title)

        return {
            "title": exact_title, "author": meta.author, "genre": meta.genre,
            "word_count": meta.word_count, "chapter_count": meta.chapter_count,
            "first_chapter": meta.first_chapter, "last_chapter": meta.last_chapter,
            "chunk_count": meta.chunk_count, "chunks_indexed": chunk_count,
            "has_style_analysis": has_style, "entities_extracted": entity_count,
            "patterns_extracted": pattern_count
        }

    def list_novels(self) -> List[dict]:
        """列出所有小说。"""
        return [{k: v for k, v in asdict(n).items() if k != "file_path"} for n in self.novels.values()]

    def get_stats(self) -> dict:
        """获取统计信息。"""
        stats = {
            "total_novels": len(self.novels),
            "total_chunks": self.collection.count(),
            "total_patterns": len(self.plot_patterns),
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "total_templates": len(self.writing_templates),
            "tantivy_ready": self.index_manager.bm25_ready,
            "optimized_search": self.index_manager.use_optimized_search,
        }
        stats["query_cache"] = self._query_cache.stats()
        return stats

    # ── 数据回调（被提取器使用，保留）──────────────────────

    def _add_entity(self, name: str, entity_type: str, description: str,
                    source_novel: str, role: str = "", first_appearance: str = "",
                    arc: str = "") -> dict:
        eid = hashlib.md5(f"{name}_{source_novel}".encode()).hexdigest()[:12]
        entity = Entity(entity_id=eid, name=name, entity_type=entity_type,
                        description=description, source_novel=source_novel,
                        role=role, first_appearance=first_appearance, arc=arc)
        self.entities[eid] = entity
        self.graph.add_node(name, **asdict(entity))
        self._save_state()
        return asdict(entity)

    def _add_relationship(self, source: str, target: str, relation_type: str,
                          description: str, source_novel: str) -> dict:
        rel = Relationship(source=source, target=target, relation_type=relation_type,
                           description=description, source_novel=source_novel)
        self.relationships.append(rel)
        self.graph.add_edge(source, target, **asdict(rel))
        self._save_state()
        return asdict(rel)

    def _add_plot_pattern(self, pattern_type: str, description: str,
                          source_novel: str, source_chapter: str,
                          pattern_text: str = "", before_context: str = "",
                          after_context: str = "", effectiveness: str = "") -> dict:
        pattern = PlotPattern(pattern_type=pattern_type, description=description,
                              source_novel=source_novel, source_chapter=source_chapter,
                              pattern_text=pattern_text, before_context=before_context,
                              after_context=after_context, effectiveness=effectiveness)
        self.plot_patterns.append(pattern)
        self._save_state()
        return asdict(pattern)

    def _add_writing_template(self, template_type: str, scene_type: str,
                              structure: str, key_beats: List[str],
                              source_novel: str, source_chapter: str,
                              example_text: str = "", effectiveness: str = "") -> dict:
        template = WritingTemplate(template_type=template_type, scene_type=scene_type,
                                   structure=structure, key_beats=key_beats,
                                   source_novel=source_novel, source_chapter=source_chapter,
                                   example_text=example_text, effectiveness=effectiveness)
        self.writing_templates.append(template)
        self._save_state()
        return asdict(template)

    # ── 代理方法：小说读取 ──────────────────────────────────

    def read_chapter(self, novel_title: str, chapter: int = 1) -> dict:
        """读取指定章节的完整正文。"""
        return self.novel_reader.read_chapter(novel_title, chapter)

    def list_chapters(self, novel_title: str) -> dict:
        """列出小说的所有章节标题及序号。"""
        return self.novel_reader.list_chapters(novel_title)

    def _resolve_novel(self, novel_title: str) -> tuple:
        """解析小说标题，返回 (exact_title, novel_id)。"""
        return resolve_novel(self.novels, novel_title)

    def resolve_novel_title(self, novel_title: str) -> str:
        """将模糊书名解析为精确书名。"""
        return resolve_novel_title(self.novels, novel_title)

    # ── 代理方法：章纲管理 ──────────────────────────────────

    async def save_outline(self, novel_title: str, outlines: Union[List[dict], dict],
                     overwrite: bool = False) -> dict:
        """保存章纲。"""
        return await self.outline_manager.save_outline(novel_title, outlines, overwrite)

    def get_outline(self, novel_title: str, chapter=None) -> dict:
        """获取章纲。"""
        return self.outline_manager.get_outline(novel_title, chapter)

    async def search_outlines(self, query: str, n_results: int = 10,
                        novel_filter: str = "", outline_type: str = "",
                        output_format: str = "compact",
                        max_content_length: int = 0) -> list:
        """语义搜索章纲。"""
        return await self.outline_manager.search_outlines(
            query, n_results, novel_filter, outline_type, output_format, max_content_length)

    def delete_outline(self, novel_title: str, chapter: int) -> dict:
        """删除章纲。"""
        return self.outline_manager.delete_outline(novel_title, chapter)

    async def extract_outline(self, novel_title: str, chapter: int) -> dict:
        """提取单章章纲。"""
        return await self.outline_manager.extract_outline(novel_title, chapter)

    async def start_outline_extraction(self, novel_title: str, start_chapter: int,
                                 end_chapter: int) -> dict:
        """启动异步批量章纲提取。"""
        return await self.outline_manager.start_outline_extraction(
            novel_title, start_chapter, end_chapter)

    # ── 代理方法：任务管理 ──────────────────────────────────

    def start_async_extraction(self, novel_title: str, max_chunks: int = 20,
                               extract_type: str = "plot_patterns") -> dict:
        """启动异步提取任务。"""
        return self.task_manager.start_async_extraction(novel_title, max_chunks, extract_type)

    def get_task_status(self, task_id: str) -> dict:
        """获取任务状态。优先从 Worker 文件系统查找，若无再从内存查找。"""
        from webnovel_kb.worker import get_task_status as worker_get_status
        from webnovel_kb.config import DATA_DIR
        queue_dir = DATA_DIR / "task_queue"
        worker_status = worker_get_status(queue_dir, task_id)
        if worker_status.get("status") != "not_found":
            return worker_status
        return self.task_manager.get_task_status(task_id)

    def cancel_task(self, task_id: str) -> dict:
        """取消任务。优先从 Worker 文件系统取消，若无再从内存取消。"""
        from webnovel_kb.worker import cancel_task as worker_cancel_task
        from webnovel_kb.config import DATA_DIR
        queue_dir = DATA_DIR / "task_queue"
        worker_res = worker_cancel_task(queue_dir, task_id)
        if worker_res.get("status") == "cancelling":
            return worker_res
        return self.task_manager.cancel_task(task_id)

    # ── 代理方法：智能搜索 ──────────────────────────────────

    async def smart_search(self, query: str, n_results: int = 5,
                     novel_filter: Optional[str] = None,
                     genre_filter: Optional[str] = None,
                     output_format: str = "compact") -> dict:
        """智能搜索——LLM 函数调用模式。"""
        return await self.smart_engine.search(
            query, n_results, novel_filter, genre_filter, output_format,
            novels=self.novels)

    # ── 代理方法：外部搜索 ──────────────────────────────────

    async def _tavily_search(self, query: str, n_results: int = 5) -> list:
        """调用知乎全网搜索 API。"""
        return await self.external_search.global_search(query, n_results)

    async def _zhihu_search(self, query: str, n_results: int = 5) -> list:
        """调用知乎站内搜索 API。"""
        return await self.external_search.zhihu_search(query, n_results)

    async def _zhida(self, query: str, model: str = "zhida-fast-1p5") -> list:
        """调用知乎直答 API。"""
        return await self.external_search.zhihu_zhida(query, model)
