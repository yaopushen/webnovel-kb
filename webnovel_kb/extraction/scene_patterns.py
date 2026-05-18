"""Scene pattern extraction from novels."""
import re
from typing import Dict, List, Any

from webnovel_kb.prompts import SCENE_PATTERN_PROMPT


class ScenePatternExtractor:
    """场景模式提取器。"""
    
    def __init__(self, chat, collection, add_pattern_fn):
        self.chat = chat
        self.collection = collection
        self.add_pattern = add_pattern_fn
    
    def extract(self, novel_title: str, novel_id: str, exact_title: str,
                max_chunks: int = 15) -> Dict[str, Any]:
        """提取场景模式。"""
        if not self.chat:
            return {"error": "Chat API未配置，无法自动提取场景模式。请设置LLM_API_KEY。"}
        
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
        
        all_patterns = []
        for chunk_text, meta in sampled:
            chapter = meta.get("chapter_title", "auto")
            messages = [
                {"role": "system", "content": "你是资深网文编辑，擅长识别具体场景中的叙事技巧。你只提取写法精妙、可复用的场景模式，宁缺毋滥。如果文本中没有值得学习的写法，输出空结果。"},
                {"role": "user", "content": f"{SCENE_PATTERN_PROMPT}\n\ntext:\n{chunk_text}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            if response:
                self._parse_scene_response(response, novel_title, chapter, all_patterns)
        
        return {
            "novel": novel_title,
            "scene_patterns_extracted": len(all_patterns),
            "patterns": all_patterns[:20]
        }
    
    def _parse_scene_response(self, response: str, novel_title: str,
                               chapter: str, all_patterns: list) -> None:
        """解析场景模式响应。"""
        pattern_matches = re.findall(
            r'\("scene_pattern"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
        )
        for scene_type, technique, analysis, original, reusable in pattern_matches:
            scene_type = scene_type.strip()
            technique = technique.strip()
            if scene_type and technique:
                result = self.add_pattern(
                    pattern_type=f"场景写法/{scene_type}",
                    description=f"{technique}: {analysis.strip()}",
                    source_novel=novel_title,
                    source_chapter=chapter,
                    pattern_text=original.strip(),
                    effectiveness=reusable.strip()
                )
                all_patterns.append(result)
