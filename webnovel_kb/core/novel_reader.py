"""小说章节读取和查询。"""
import re
from typing import Dict, Optional, Tuple, List

from webnovel_kb.core.chunker import TextChunker
from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.format import clean_text
from webnovel_kb.utils.chinese_numbers import int_to_cn
from webnovel_kb.utils.novel_resolver import resolve_novel

logger = get_logger("core.novel_reader")


class NovelReader:
    """小说章节读取。"""

    def __init__(self, collection, novels: dict):
        self.collection = collection
        self.novels = novels

    def resolve_novel(self, novel_title: str) -> Tuple[Optional[str], Optional[str]]:
        """解析小说标题，返回 (exact_title, novel_id)。"""
        return resolve_novel(self.novels, novel_title)

    def resolve_novel_title(self, novel_title: str) -> str:
        """将模糊书名解析为精确书名。找不到时返回原始输入。"""
        exact, _ = resolve_novel(self.novels, novel_title)
        return exact if exact else novel_title

    def read_chapter(self, novel_title: str, chapter: int = 1) -> dict:
        """读取指定章节的完整正文。chapter为章节序号(1-based)。"""
        exact_title, novel_id = self.resolve_novel(novel_title)
        if not exact_title:
            return {"error": f"未找到小说: {novel_title}"}

        cn = int_to_cn(chapter)
        zero_padded = str(chapter).zfill(3)
        patterns = [
            f"第{chapter}章",
            f"第{cn}章",
            f"{chapter}、",
            zero_padded,
        ]

        if chapter == 0:
            patterns = [""]

        seen = set()
        unique_patterns = []
        for p in patterns:
            if p not in seen:
                seen.add(p)
                unique_patterns.append(p)

        all_chunks = []
        for pat in unique_patterns:
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
                "hint": "使用 stats 查看 chapter_count 确认有效范围"
            }

        all_chunks.sort(key=lambda x: x[0])
        chunk_texts = [doc for _, doc, _ in all_chunks]
        full_text = TextChunker.reassemble(chunk_texts)
        full_text = clean_text(full_text)

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
        exact_title, novel_id = self.resolve_novel(novel_title)
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
