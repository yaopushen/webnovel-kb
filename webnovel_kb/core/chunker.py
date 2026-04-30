"""Text chunking utilities."""
import re
from typing import List, Tuple
from webnovel_kb.config import CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:
    """文本分块器，支持章节识别和智能切分。"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[Tuple[str, str]]:
        """
        将文本分块，返回 (chunk_text, chapter_title) 元组列表。
        优先按章节切分，章节过长时再细分。
        """
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
                if len(chapter_text) <= self.chunk_size * 1.5:
                    chunk = chapter_text.strip()
                    if chunk:
                        chunks.append((chunk, current_chapter))
                else:
                    sub_chunks = self._chunk_simple(chapter_text)
                    for sc in sub_chunks:
                        chunks.append((sc, current_chapter))
            return chunks
        return [(c, "") for c in self._chunk_simple(text)]
    
    def _chunk_simple(self, text: str) -> List[str]:
        """简单分块，在句子边界处切分。"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                for sep in ["。", "！", "？", "\n\n", "\n", "；"]:
                    pos = text.rfind(sep, start + self.chunk_size // 2, end)
                    if pos > start:
                        end = pos + len(sep)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.overlap if end < len(text) else end
        return chunks
