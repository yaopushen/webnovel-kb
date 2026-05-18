"""Plot pattern extraction from novels."""
import re
import logging
from typing import Dict, List, Any
from dataclasses import asdict

from webnovel_kb.data_models import PlotPattern
from webnovel_kb.prompts import PLOT_PATTERN_EXTRACTION_PROMPT, PLOT_TIMELINE_PROMPT, PLOT_PATTERN_CROSS_CHUNK_PROMPT

logger = logging.getLogger("webnovel-kb")


class PlotPatternExtractor:
    """情节模式提取器。"""
    
    def __init__(self, chat, collection, plot_patterns: list, add_pattern_fn, save_state_fn):
        self.chat = chat
        self.collection = collection
        self.plot_patterns = plot_patterns
        self.add_pattern = add_pattern_fn
        self.save_state = save_state_fn
    
    def extract(self, novel_title: str, novel_id: str, exact_title: str,
                max_chunks: int = 20) -> Dict[str, Any]:
        """快速提取情节模式（单chunk模式）。"""
        if not self.chat:
            return {"error": "Chat API未配置，无法自动提取情节模式。请设置LLM_API_KEY。"}
        
        results = self.collection.query(
            query_texts=["悬念 反转 冲突升级 伏笔 情节转折"],
            n_results=max_chunks,
            where={"title": exact_title}
        )
        if not results or not results["documents"]:
            return {"error": f"未找到小说内容: {exact_title}"}
        
        all_patterns = []
        for chunk in results["documents"][0][:max_chunks]:
            prompt = PLOT_PATTERN_EXTRACTION_PROMPT.replace("{tuple_delimiter}", "<|>")
            prompt = prompt.replace("{record_delimiter}", "|||")
            prompt = prompt.replace("{completion_delimiter}", "<DONE>")
            messages = [
                {"role": "system", "content": "你是资深网文编辑和写作教练，擅长识别真正有学习价值的叙事技巧。你只提取写法精妙、可复用的情节模式，宁缺毋滥。如果文本中没有值得学习的写法，输出空结果。"},
                {"role": "user", "content": f"{prompt}\n\ntext:\n{chunk}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            logger.info(f"LLM response for plot pattern: {response[:500] if response else 'None'}")
            if response:
                self._parse_pattern_response(response, novel_title, all_patterns)
        
        return {
            "novel": novel_title,
            "patterns_extracted": len(all_patterns),
            "patterns": all_patterns[:20]
        }
    
    def extract_cross_chunk(self, novel_title: str, novel_id: str, exact_title: str,
                            max_chunks: int = 20, progress_callback=None) -> Dict[str, Any]:
        """跨章节提取情节模式（带时间线分析）。"""
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
        step = max(1, total_chunks // max_chunks)
        sampled = chunks_with_meta[::step][:max_chunks]
        
        timeline_events = self._build_timeline(sampled, progress_callback)
        if not timeline_events:
            return {"novel": novel_title, "patterns_extracted": 0, "patterns": []}
        
        timeline_text = "\n".join(timeline_events)
        logger.info(f"Timeline for {novel_title}: {len(timeline_events)} events")
        
        all_patterns = self._analyze_cross_chunk_patterns(timeline_text, novel_title)
        
        return {
            "novel": novel_title,
            "patterns_extracted": len(all_patterns),
            "timeline_events": len(timeline_events),
            "patterns": all_patterns[:20]
        }
    
    def _build_timeline(self, sampled: list, progress_callback=None) -> list:
        """构建情节时间线。"""
        timeline_events = []
        for idx, (chunk_text, meta) in enumerate(sampled):
            prompt = PLOT_TIMELINE_PROMPT.replace("{tuple_delimiter}", "<|>")
            prompt = prompt.replace("{record_delimiter}", "|||")
            prompt = prompt.replace("{completion_delimiter}", "<DONE>")
            messages = [
                {"role": "system", "content": "你是网文情节分析助手，擅长从文本中提取关键情节事件。"},
                {"role": "user", "content": f"{prompt}\n\ntext:\n{chunk_text}"}
            ]
            response = self.chat.chat(messages, temperature=0.1)
            if response:
                for line in response.strip().split("\n"):
                    line = line.strip()
                    m = re.match(r'\[(\d+)\]\s*(.+)', line)
                    if m:
                        event_desc = m.group(2).strip()
                        if event_desc and event_desc != "<DONE>":
                            timeline_events.append(f"[{len(timeline_events)+1}] {event_desc}")
            
            if progress_callback:
                progress_callback(idx + 1, len(sampled))
        
        return timeline_events
    
    def _analyze_cross_chunk_patterns(self, timeline_text: str, novel_title: str) -> list:
        """分析跨章节情节模式。"""
        prompt = PLOT_PATTERN_CROSS_CHUNK_PROMPT.replace("{tuple_delimiter}", "<|>")
        prompt = prompt.replace("{record_delimiter}", "|||")
        prompt = prompt.replace("{completion_delimiter}", "<DONE>")
        messages = [
            {"role": "system", "content": "你是资深网文编辑和写作教练，擅长识别跨章节的长线叙事模式。你只提取纵览全局才能发现的模式，单章节内的手法不值得提取。"},
            {"role": "user", "content": f"{prompt}\n\n时间线：\n{timeline_text}"}
        ]
        response = self.chat.chat(messages, temperature=0.3, max_tokens=8192)
        logger.info(f"Cross-chunk pattern response: {response[:500] if response else 'None'}")
        
        all_patterns = []
        if response:
            pattern_matches = re.findall(
                r'\("pattern"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
            )
            logger.info(f"Cross-chunk pattern matches found: {len(pattern_matches)}")
            for ptype, desc, setup_text, payoff_text, bridge, eff in pattern_matches:
                ptype = ptype.strip()
                desc = desc.strip()
                setup_text = setup_text.strip()
                payoff_text = payoff_text.strip()
                if ptype and desc and (len(setup_text) > 50 or len(payoff_text) > 50):
                    pattern_text = f"【起点】\n{setup_text}\n\n【终点】\n{payoff_text}"
                    result = self.add_pattern(
                        pattern_type=ptype,
                        description=desc,
                        source_novel=novel_title,
                        source_chapter="跨章节",
                        pattern_text=pattern_text,
                        before_context=setup_text[:200],
                        after_context=payoff_text[:200],
                        effectiveness=eff.strip()
                    )
                    all_patterns.append(result)
        
        return all_patterns
    
    def _parse_pattern_response(self, response: str, novel_title: str, all_patterns: list) -> None:
        """解析情节模式响应。"""
        pattern_matches = re.findall(
            r'\("pattern"<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)<\|>(.+?)\)', response
        )
        logger.info(f"Pattern matches found: {len(pattern_matches)}")
        for ptype, desc, before, ptext, after, eff in pattern_matches:
            ptype = ptype.strip()
            desc = desc.strip()
            if ptype and desc:
                ptext_full = ptext.strip()
                if len(ptext_full) < 500:
                    continue
                result = self.add_pattern(
                    pattern_type=ptype,
                    description=desc,
                    source_novel=novel_title,
                    source_chapter="auto",
                    pattern_text=ptext_full,
                    before_context=before.strip(),
                    after_context=after.strip(),
                    effectiveness=eff.strip()
                )
                all_patterns.append(result)
