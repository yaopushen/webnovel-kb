"""Entity extraction from novels."""
import hashlib
import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from webnovel_kb.data_models import Entity
from webnovel_kb.prompts import ENTITY_EXTRACTION_PROMPT, ENTITY_TIMELINE_PROMPT, ENTITY_CROSS_CHUNK_PROMPT

logger = logging.getLogger("webnovel-kb")


class EntityExtractor:
    """实体提取器，从小说中提取角色、地点、组织等实体。"""
    
    def __init__(self, chat, collection, entities: Dict[str, Entity], 
                 relationships: list, graph, add_entity_fn, add_relationship_fn,
                 save_state_fn, entities_collection):
        self.chat = chat
        self.collection = collection
        self.entities = entities
        self.relationships = relationships
        self.graph = graph
        self.add_entity = add_entity_fn
        self.add_relationship = add_relationship_fn
        self.save_state = save_state_fn
        self.entities_collection = entities_collection
    
    def extract(self, novel_title: str, novel_id: str, exact_title: str, 
                max_chunks: int = 20) -> Dict[str, Any]:
        """快速提取实体（单chunk模式）。"""
        if not self.chat:
            return {"error": "Chat API未配置，无法自动提取实体。请设置LLM_API_KEY。"}
        
        results = self.collection.query(
            query_texts=["角色登场 人物介绍 关系揭示"],
            n_results=max_chunks,
            where={"title": exact_title}
        )
        if not results or not results["documents"]:
            return {"error": f"未找到小说内容: {exact_title}"}
        
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
                self._parse_entity_response(response, novel_title, all_extracted)
                self._parse_relationship_response(response, novel_title, all_extracted)
        
        return {
            "novel": novel_title,
            "entities_extracted": len(all_extracted["entities"]),
            "relationships_extracted": len(all_extracted["relationships"]),
            "entities": all_extracted["entities"][:20],
            "relationships": all_extracted["relationships"][:20]
        }
    
    def extract_cross_chunk(self, novel_title: str, novel_id: str, exact_title: str,
                            max_chunks: int = 50, progress_callback=None) -> Dict[str, Any]:
        """跨章节提取实体（带时间线分析）。"""
        if not self.chat:
            return {"error": "Chat API未配置"}
        
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
        key_scene_indices = self._find_key_scenes(exact_title, total_chunks)
        
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
        
        entity_timeline = self._build_entity_timeline(sampled, progress_callback)
        if not entity_timeline:
            return {"novel": novel_title, "entities_extracted": 0, "relationships_extracted": 0}
        
        entity_timeline = self._dedupe_timeline(entity_timeline)
        logger.info(f"Entity timeline for {novel_title}: {len(entity_timeline)} unique entries")
        
        all_extracted = self._analyze_cross_chunk_entities(entity_timeline, novel_title)
        
        return {
            "novel": novel_title,
            "entities_extracted": len(all_extracted["entities"]),
            "relationships_extracted": len(all_extracted["relationships"]),
            "entities": all_extracted["entities"][:20],
            "relationships": all_extracted["relationships"][:20]
        }
    
    def _find_key_scenes(self, exact_title: str, total_chunks: int) -> set:
        """查找关键场景索引。"""
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
        return key_scene_indices
    
    def _build_entity_timeline(self, sampled: list, progress_callback=None) -> list:
        """构建实体时间线。"""
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
            
            if progress_callback:
                progress_callback(idx + 1, len(sampled))
        
        return entity_timeline
    
    def _dedupe_timeline(self, timeline: list) -> list:
        """去重时间线。"""
        seen = set()
        deduped = []
        for line in timeline:
            key = line[:80]
            if key not in seen:
                seen.add(key)
                deduped.append(line)
        return deduped
    
    def _analyze_cross_chunk_entities(self, timeline: list, novel_title: str) -> dict:
        """分析跨章节实体。"""
        max_timeline_chars = 20000
        timeline_text = ""
        for line in timeline:
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
        
        all_extracted = {"entities": [], "relationships": []}
        if response:
            self._parse_cross_chunk_entity_response(response, novel_title, all_extracted)
            self._parse_cross_chunk_relationship_response(response, novel_title, all_extracted)
        
        return all_extracted
    
    def _parse_entity_response(self, response: str, novel_title: str, all_extracted: dict) -> None:
        """解析实体提取响应。"""
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
    
    def _parse_relationship_response(self, response: str, novel_title: str, all_extracted: dict) -> None:
        """解析关系提取响应。"""
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
    
    def _parse_cross_chunk_entity_response(self, response: str, novel_title: str, all_extracted: dict) -> None:
        """解析跨章节实体响应。"""
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
                            self.save_state()
                        existing = True
                        break
                if not existing:
                    result = self.add_entity(name, etype, desc, novel_title,
                                             role=role, first_appearance=first_appear, arc=arc)
                    all_extracted["entities"].append(result)
    
    def _parse_cross_chunk_relationship_response(self, response: str, novel_title: str, all_extracted: dict) -> None:
        """解析跨章节关系响应。"""
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
                        self.save_state()
                    if "error" not in result:
                        all_extracted["relationships"].append(result)
                except Exception:
                    pass
