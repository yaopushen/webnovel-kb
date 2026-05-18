"""Humor scene extraction from novels."""
import re
from typing import List, Dict, Any

from webnovel_kb.prompts import HUMOR_SCENE_EXTRACTION_PROMPT


class HumorExtractor:
    """幽默场景提取器。"""
    
    def __init__(self, chat):
        self.chat = chat
    
    def extract(self, texts: List[str], metas: List[dict], exact_title: str) -> List[Dict[str, Any]]:
        """提取幽默场景。"""
        if not self.chat or not texts:
            return []
        
        n = len(texts)
        sample_indices = [0, n//4, n//2, n*3//4, n-1] if n >= 5 else list(range(n))
        humor_scenes = []
        
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
