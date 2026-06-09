"""Unified search interface with output formatting."""
from typing import Optional, List, Any

from webnovel_kb.utils.dedupe import dedupe_results
from webnovel_kb.utils.format import format_search_results


class UnifiedSearch:
    """统一搜索接口，整合所有搜索模式。"""
    
    def __init__(self, semantic_search, bm25_search, hybrid_search, rerank_search, reranker, query_cache=None,
                 outlines_search_fn=None, knowledge_search_fn=None):
        self.semantic_search = semantic_search
        self.bm25_search = bm25_search
        self.hybrid_search = hybrid_search
        self.rerank_search = rerank_search
        self.reranker = reranker
        self.query_cache = query_cache
        self.outlines_search_fn = outlines_search_fn
        self.knowledge_search_fn = knowledge_search_fn
    
    async def search(
        self,
        query: str,
        mode: str = "hybrid",
        n_results: int = 10,
        novel_filter: Optional[str] = None,
        genre_filter: Optional[str] = None,
        chapter_filter: Optional[str] = None,
        alpha: float = 0.6,
        use_rerank: bool = False,
        output_format: str = "compact",
        max_content_length: int = 0,
        dedupe: bool = True,
        scope: str = "chunks"
    ) -> List[Any]:
        """
        统一搜索接口。
        
        Args:
            query: 搜索查询
            mode: 搜索模式 - semantic/bm25/hybrid/rerank
            n_results: 返回结果数量
            novel_filter: 小说标题过滤
            genre_filter: 类型过滤
            chapter_filter: 章节过滤
            alpha: hybrid模式下语义权重
            use_rerank: 是否使用rerank精排
            output_format: 输出格式 - raw/compact/clean
            max_content_length: 内容最大长度
            dedupe: 是否去重
            scope: 搜索范围 - chunks/outlines/agent_knowledge/all
        
        Returns:
            搜索结果列表
        """
        # scope="outlines" — 委托给 outlines 搜索函数
        if scope == "outlines" and self.outlines_search_fn:
            raw = await self.outlines_search_fn(query, n_results, novel_filter)
            return format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)

        # scope="agent_knowledge" — 委托给 knowledge 搜索函数
        if scope == "agent_knowledge" and self.knowledge_search_fn:
            raw = await self.knowledge_search_fn(query, n_results)
            return format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)

        # scope="all" — 搜索所有三个集合，合并结果并标记来源
        if scope == "all":
            all_results = []

            # 1. 搜索 chunks（现有逻辑）
            if mode == "semantic":
                chunk_raw = await self.semantic_search.search(query, n_results, novel_filter, genre_filter, chapter_filter)
            elif mode == "bm25":
                chunk_raw = await self.bm25_search.search(query, n_results, novel_filter, genre_filter)
            elif mode == "rerank":
                chunk_raw = await self.rerank_search.search(query, n_results, novel_filter, genre_filter)
            else:
                if use_rerank and self.reranker:
                    chunk_raw = await self.rerank_search.search(query, n_results, novel_filter, genre_filter)
                else:
                    chunk_raw = await self.hybrid_search.search(query, n_results, alpha, novel_filter, genre_filter)

            for r in chunk_raw:
                if isinstance(r, dict):
                    r["_source_type"] = "chunk"
            all_results.extend(chunk_raw)

            # 2. 搜索 outlines
            if self.outlines_search_fn:
                outline_raw = await self.outlines_search_fn(query, n_results, novel_filter)
                for r in outline_raw:
                    if isinstance(r, dict):
                        r["_source_type"] = "outline"
                all_results.extend(outline_raw)

            # 3. 搜索 agent knowledge
            if self.knowledge_search_fn:
                knowledge_raw = await self.knowledge_search_fn(query, n_results)
                for r in knowledge_raw:
                    if isinstance(r, dict):
                        r["_source_type"] = "knowledge"
                all_results.extend(knowledge_raw)

            return format_search_results(all_results, output_format, max_content_length, dedupe, dedupe_results)

        # scope="chunks"（默认）— 原有逻辑
        if mode == "semantic":
            raw = await self.semantic_search.search(query, n_results, novel_filter, genre_filter, chapter_filter)
        elif mode == "bm25":
            raw = await self.bm25_search.search(query, n_results, novel_filter, genre_filter)
        elif mode == "rerank":
            raw = await self.rerank_search.search(query, n_results, novel_filter, genre_filter)
        else:
            if use_rerank and self.reranker:
                raw = await self.rerank_search.search(query, n_results, novel_filter, genre_filter)
            else:
                raw = await self.hybrid_search.search(query, n_results, alpha, novel_filter, genre_filter)

        return format_search_results(raw, output_format, max_content_length, dedupe, dedupe_results)
