"""Writing template extraction from novels."""
import re
from typing import Dict, List, Any
from dataclasses import asdict

from webnovel_kb.data_models import WritingTemplate
from webnovel_kb.prompts import WRITING_TEMPLATE_EXTRACTION_PROMPT


class WritingTemplateExtractor:
    """写作模板提取器。"""
    
    def __init__(self, chat, collection, writing_templates: list, add_template_fn, save_state_fn):
        self.chat = chat
        self.collection = collection
        self.writing_templates = writing_templates
        self.add_template = add_template_fn
        self.save_state = save_state_fn
    
    def extract(self, novel_title: str, novel_id: str, exact_title: str,
                max_chunks: int = 15) -> Dict[str, Any]:
        """提取写作模板。"""
        if not self.chat:
            return {"error": "Chat API未配置，无法自动提取写作模板。请设置LLM_API_KEY。"}
        
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
        
        all_templates = []
        for chunk_text, meta in sampled:
            chapter_title = meta.get("chapter_title", "")
            messages = [
                {"role": "system", "content": "你是资深网文编辑和写作教练，擅长从优秀网文中提取可复用的场景写法模板。你只提取结构清晰、有学习价值的模板，宁缺毋滥。如果文本中没有值得提取的模板，输出空结果。"},
                {"role": "user", "content": f"{WRITING_TEMPLATE_EXTRACTION_PROMPT}\n\ntext:\n{chunk_text}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            if response:
                self._parse_template_response(response, novel_title, chapter_title, all_templates)
        
        return {
            "novel": novel_title,
            "templates_extracted": len(all_templates),
            "templates": all_templates[:20]
        }
    
    def _parse_template_response(self, response: str, novel_title: str, 
                                  chapter_title: str, all_templates: list) -> None:
        """解析写作模板响应。"""
        template_matches = re.findall(
            r'\("template"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
        )
        for scene_type, structure, beats_str, example, eff in template_matches:
            scene_type = scene_type.strip()
            structure = structure.strip()
            if scene_type and structure:
                beats = [b.strip() for b in beats_str.split(',') if b.strip()]
                result = self.add_template(
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
