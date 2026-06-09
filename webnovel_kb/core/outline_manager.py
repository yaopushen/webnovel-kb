"""章纲管理——保存、获取、搜索、删除、提取。"""
import asyncio
import json
import threading
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Union

from webnovel_kb.data_models import ChapterOutline
from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.exceptions import ExtractionError, SearchError
from webnovel_kb.utils.novel_resolver import resolve_novel

logger = get_logger("core.outline_manager")


class OutlineManager:
    """章纲管理器。"""

    def __init__(self, outlines_collection, embedding_fn, outline_extractor,
                 chapter_outlines: dict, novels: dict, save_state_fn):
        self.outlines_collection = outlines_collection
        self.embedding_fn = embedding_fn
        self.outline_extractor = outline_extractor
        self.chapter_outlines = chapter_outlines
        self.novels = novels
        self._save_state = save_state_fn
        self._async_tasks: Dict[str, dict] = {}
        self._task_lock = threading.Lock()

    async def _save_single_outline(self, exact_title: str, chapter: int,
                             content: str, outline_type: str = "章纲",
                             tags: Optional[List[str]] = None,
                             now: str = "") -> dict:
        """保存单条章纲到内存和 ChromaDB。"""
        tags = tags or []
        outline_id = f"{exact_title}_{chapter}"
        existing = self.chapter_outlines.get(outline_id)
        created_at = existing.created_at if existing else now

        outline = ChapterOutline(
            outline_id=outline_id,
            novel_title=exact_title,
            chapter=chapter,
            outline_type=outline_type,
            content=content,
            created_at=created_at,
            updated_at=now,
            tags=tags
        )
        self.chapter_outlines[outline_id] = outline

        try:
            meta = {
                "novel_title": exact_title,
                "chapter": chapter,
                "outline_type": outline_type,
                "tags": json.dumps(tags)
            }
            emb = (await self.embedding_fn([content]))[0] if self.embedding_fn else None
            existing_ids = self.outlines_collection.get(ids=[outline_id])
            if existing_ids and existing_ids.get("ids"):
                self.outlines_collection.update(
                    ids=[outline_id], documents=[content], metadatas=[meta],
                    embeddings=[emb] if emb else None
                )
            else:
                self.outlines_collection.add(
                    ids=[outline_id], documents=[content],
                    metadatas=[meta], embeddings=[emb] if emb else None
                )
        except Exception as e:
            logger.warning(f"Failed to index outline in ChromaDB: {e}")

        return asdict(outline)

    async def save_outline(self, novel_title: str, outlines: Union[List[dict], dict],
                     overwrite: bool = False) -> dict:
        """保存章纲。支持单条 dict 或批量 list。"""
        if isinstance(outlines, dict):
            outlines = [outlines]

        exact_title = None
        for n in self.novels.values():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                break
        if not exact_title:
            raise SearchError(f"未找到小说: {novel_title}")

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        saved = []
        skipped = []
        errors = []
        for i, item in enumerate(outlines):
            chapter = item.get("chapter")
            content = item.get("content", "")
            if chapter is None or not content:
                errors.append({"index": i, "error": "缺少 chapter 或 content"})
                continue

            outline_id = f"{exact_title}_{chapter}"
            if not overwrite and outline_id in self.chapter_outlines:
                skipped.append({"chapter": chapter, "status": "already_exists"})
                continue

            try:
                o = await self._save_single_outline(
                    exact_title, chapter, content,
                    outline_type=item.get("outline_type", "章纲"),
                    tags=item.get("tags"),
                    now=now
                )
                saved.append(o)
            except Exception as e:
                errors.append({"index": i, "chapter": chapter, "error": str(e)})

        self._save_state()
        saved_chapters = [o["chapter"] for o in saved]
        return {
            "status": "ok",
            "saved_count": len(saved),
            "saved_chapters": saved_chapters,
            "skipped_count": len(skipped),
            "skipped": skipped if skipped else None,
            "error_count": len(errors),
            "errors": errors if errors else None
        }

    def get_outline(self, novel_title: str, chapter=None) -> dict:
        """获取章纲。chapter 参数:
            - 0 或 None: 返回章节列表（仅章节号和类型，不含正文）
            - 正整数: 返回该章完整章纲
            - "full" (str): 返回全书全量章纲文本（按章节号排序串联）"""
        exact_title = None
        for n in self.novels.values():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                break
        if not exact_title:
            raise SearchError(f"未找到小说: {novel_title}")

        # chapter="full": 返回全书全量章纲文本
        if isinstance(chapter, str) and chapter.lower() == "full":
            items = []
            for o in self.chapter_outlines.values():
                if o.novel_title == exact_title:
                    items.append((o.chapter, o))
            if not items:
                return {"novel": exact_title, "total": 0, "full_text": ""}
            items.sort(key=lambda x: x[0])
            full_text = "\n\n---\n\n".join(
                f"第{o.chapter}章 ({o.outline_type})\n{o.content}"
                for _, o in items
            )
            return {
                "novel": exact_title,
                "total": len(items),
                "total_chars": len(full_text),
                "full_text": full_text
            }

        chapter_int = int(chapter) if chapter is not None else 0
        if chapter_int > 0:
            outline_id = f"{exact_title}_{chapter_int}"
            outline = self.chapter_outlines.get(outline_id)
            if not outline:
                return {"novel": exact_title, "chapter": chapter_int, "outline": None,
                        "hint": "该章尚无章纲，可用 save_outline 或 extract_outline 创建"}
            return {
                "novel": exact_title,
                "chapter": chapter_int,
                "outline_type": outline.outline_type,
                "content": outline.content,
                "tags": outline.tags
            }

        chapters = []
        for o in self.chapter_outlines.values():
            if o.novel_title == exact_title:
                entry = {"chapter": o.chapter, "outline_type": o.outline_type}
                if o.tags:
                    entry["tags"] = o.tags
                chapters.append(entry)
        chapters.sort(key=lambda x: x["chapter"])
        return {
            "novel": exact_title,
            "total": len(chapters),
            "chapters": chapters
        }

    async def search_outlines(self, query: str, n_results: int = 10,
                        novel_filter: str = "",
                        outline_type: str = "",
                        output_format: str = "compact",
                        max_content_length: int = 0) -> list:
        """语义搜索章纲。"""
        collection = self.outlines_collection
        if collection.count() == 0:
            return [{"hint": "尚无章纲数据，请先使用 save_outline 创建"}]

        exact_novel = None
        if novel_filter:
            exact_novel = resolve_novel(self.novels, novel_filter)[0]

        where = {}
        if exact_novel:
            where["novel_title"] = exact_novel
        if outline_type:
            where["outline_type"] = outline_type

        try:
            query_params = {"n_results": n_results}
            if where:
                query_params["where"] = where

            if self.embedding_fn:
                query_vec = (await self.embedding_fn([query]))[0]
                query_params["query_embeddings"] = [query_vec]
            else:
                query_params["query_texts"] = [query]

            results = collection.query(**query_params)
        except Exception as e:
            logger.error(f"Outline search failed: {e}")
            return [{"error": f"搜索章纲失败: {e}"}]

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        raw = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            dist = results["distances"][0][i] if results.get("distances") else 0
            raw.append({
                "novel_title": meta.get("novel_title", ""),
                "chapter": meta.get("chapter", 0),
                "outline_type": meta.get("outline_type", ""),
                "content": doc[:max_content_length] if max_content_length else doc,
                "tags": json.loads(meta.get("tags", "[]")),
                "score": round(1 - dist, 4)
            })

        if output_format == "compact":
            return [
                f"[{r['novel_title']} 第{r['chapter']}章 ({r['outline_type']})] {r['content'][:200]}..."
                for r in raw
            ]
        elif output_format == "clean":
            return [r["content"][:max_content_length] if max_content_length else r["content"] for r in raw]
        return raw

    def delete_outline(self, novel_title: str, chapter: int) -> dict:
        """删除章纲。"""
        exact_title = None
        for n in self.novels.values():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                break
        if not exact_title:
            raise SearchError(f"未找到小说: {novel_title}")

        outline_id = f"{exact_title}_{chapter}"
        if outline_id not in self.chapter_outlines:
            return {"status": "not_found", "novel": exact_title, "chapter": chapter,
                    "hint": "该章尚无章纲"}

        del self.chapter_outlines[outline_id]
        try:
            self.outlines_collection.delete(ids=[outline_id])
        except Exception as e:
            logger.warning(f"Failed to delete outline from ChromaDB: {e}")

        self._save_state()
        return {"status": "deleted", "novel": exact_title, "chapter": chapter}

    async def extract_outline(self, novel_title: str, chapter: int) -> dict:
        """提取单章章纲。读取章节→LLM提取→自动存入。"""
        if not self.outline_extractor:
            raise ExtractionError("Chat API 未配置，无法提取章纲")
        return await self.outline_extractor.extract_single(novel_title, chapter, save=True)

    async def start_outline_extraction(self, novel_title: str, start_chapter: int,
                                 end_chapter: int) -> dict:
        """启动异步串行批量章纲提取。"""
        if not self.outline_extractor:
            raise ExtractionError("Chat API 未配置，无法提取章纲")

        exact_title = None
        for n in self.novels.values():
            if n.title == novel_title or novel_title in n.title:
                exact_title = n.title
                break
        if not exact_title:
            raise ExtractionError(f"未找到小说: {novel_title}")

        if start_chapter > end_chapter:
            raise ExtractionError(
                f"起始章节({start_chapter})不能大于结束章节({end_chapter})"
            )

        total = end_chapter - start_chapter + 1

        from webnovel_kb.worker import submit_task
        from webnovel_kb.config import DATA_DIR

        queue_dir = DATA_DIR / "task_queue"
        task_id = submit_task(
            queue_dir=queue_dir,
            task_type="extract_outline_batch",
            params={
                "novel_title": exact_title,
                "start_chapter": start_chapter,
                "end_chapter": end_chapter
            }
        )

        return {
            "task_id": task_id,
            "status": "started",
            "novel": exact_title,
            "chapter_range": f"{start_chapter}-{end_chapter}",
            "total": total
        }

