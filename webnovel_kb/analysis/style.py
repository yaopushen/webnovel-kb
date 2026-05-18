"""Style analysis for novels."""
import re
import math
import logging
from typing import Dict, List, Any
from dataclasses import asdict

from webnovel_kb.data_models import StyleProfile

logger = logging.getLogger("webnovel-kb")


class StyleAnalyzer:
    """风格分析器。"""
    
    def __init__(self, chat, collection, style_profiles: dict, save_state_fn):
        self.chat = chat
        self.collection = collection
        self.style_profiles = style_profiles
        self.save_state = save_state_fn
        
        self.dialogue_re_list = [
            re.compile(r'\u201c(.+?)\u201d'),
            re.compile(r'\u300c(.+?)\u300d'),
            re.compile(r'"(.+?)"'),
        ]
        self.inner_re_list = [
            re.compile(r'\u300e(.+?)\u300f'),
        ]
        
        self.humor_patterns = [
            r"吐槽", r"自嘲", r"无语", r"呵呵", r"呵呵呵",
            r"这什么鬼", r"搞什么", r"算了", r"算了算了", r"真香",
            r"不是吧", r"不会吧", r"离谱", r"抽象", r"绝了",
            r"好家伙", r"好嘛", r"服了", r"麻了", r"裂开"
        ]
        self.tension_patterns = [
            r"危险", r"紧迫", r"来不及", r"来不及了",
            r"必须", r"立刻", r"马上", r"冲", r"跑", r"逃",
            r"死", r"杀", r"血", r"疼", r"痛"
        ]
        self.relax_patterns = [
            r"松了口", r"放松", r"平静", r"安宁", r"悠闲",
            r"舒适", r"温暖", r"安心", r"笑了", r"轻松"
        ]
        
        self.ai_patterns = [
            re.compile(r'不禁(?:心头)?(?:一)?(?:颤|愣|动|笑|叹|悲|怒|惊|寒|凛)'),
            re.compile(r'缓缓地?(?:站|走|转|抬|放|伸|收|退|移|开|说|道|开口|闭眼|睁眼|点头|摇头|起身|坐下)'),
            re.compile(r'微微一笑'),
            re.compile(r'嘴角[微轻]?[上扬翘]'),
            re.compile(r'眼中闪过(?:一丝|一抹|一道)?(?:惊|恐|怒|喜|忧|疑|厉|寒|异|凌厉)'),
            re.compile(r'心中暗(?:道|想|叹|惊|喜|怒|说|忖)'),
            re.compile(r'仿佛[^。！？\n]{2,15}一般'),
            re.compile(r'宛如[^。！？\n]{2,15}似的'),
            re.compile(r'犹如[^。！？\n]{2,15}一样'),
            re.compile(r'一股(?:暖流|寒意|力量|气息|杀意|威压|劲风|热流)'),
            re.compile(r'不由自主地?'),
            re.compile(r'情不自禁地?'),
            re.compile(r'若有所思(?:地|地看着)?'),
            re.compile(r'意味深长(?:地|地看着)?'),
            re.compile(r'心念一转'),
            re.compile(r'暗自思忖'),
            re.compile(r'不由得(?:心头)?(?:一)?(?:颤|愣|动|笑|叹|悲|怒|惊|寒)'),
            re.compile(r'淡淡地?(?:说|道|开口|笑|回应|回答|语气)'),
            re.compile(r'深深地?(?:看|望|叹|吸|凝视)'),
            re.compile(r'不禁(?:让|令|使)人'),
            re.compile(r'令人(?:不禁|难以)?(?:心|生|感|觉)'),
        ]
    
    def analyze(self, novel_title: str, novel_id: str, exact_title: str,
                humor_extractor=None) -> Dict[str, Any]:
        """分析小说风格。"""
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
        all_texts = [p[0] for p in paired]
        all_metas = [p[1] for p in paired]
        
        total_chars = sum(len(t) for t in all_texts)
        if total_chars == 0:
            return {"error": "文本为空"}
        
        dialogue_chars = 0
        inner_chars = 0
        sentence_lengths = []
        
        for text in all_texts:
            sentences = re.split(r'[。！？\n]', text)
            lengths = [len(s.strip()) for s in sentences if s.strip()]
            sentence_lengths.extend(lengths)
            
            for dre in self.dialogue_re_list:
                for m in dre.finditer(text):
                    dialogue_chars += len(m.group(1))
            for ire in self.inner_re_list:
                for m in ire.finditer(text):
                    inner_chars += len(m.group(1))
        
        avg_sent_len = round(sum(sentence_lengths) / len(sentence_lengths), 1) if sentence_lengths else 0
        dialogue_ratio = round(dialogue_chars / total_chars, 3) if total_chars else 0
        inner_ratio = round(inner_chars / total_chars, 3) if total_chars else 0
        
        section_breakdown = self._analyze_sections(all_texts, all_metas)
        
        humor_scenes = []
        if humor_extractor:
            humor_scenes = humor_extractor.extract(all_texts, all_metas, exact_title)
        
        sample_passages = self._extract_sample_passages(all_texts, all_metas)
        
        ai_score, matched_humor = self._analyze_ai_patterns(all_texts, total_chars)
        
        tension_count, relax_count, chapter_hook_rate, pace_type = self._analyze_pacing(
            all_texts, all_metas
        )
        
        narrative_perspective = self._analyze_perspective(all_texts)
        
        profile = StyleProfile(
            avg_sentence_len=avg_sent_len,
            dialogue_ratio=dialogue_ratio,
            inner_monologue_ratio=inner_ratio,
            description_ratio=0.0,
            action_ratio=0.0,
            narrative_perspective=narrative_perspective,
            section_breakdown=section_breakdown,
            humor_scenes=humor_scenes,
            sample_passages=sample_passages,
            ai_fingerprint_score=round(ai_score, 2),
            oral_score=round(min(len(matched_humor) / (total_chars / 5000), 10) if total_chars > 0 else 0, 2),
            chapter_hook_rate=chapter_hook_rate,
            pace_type=pace_type,
            humor_markers=list(dict.fromkeys(matched_humor))[:10],
            pacing_info={
                "tension_markers": tension_count,
                "relax_markers": relax_count,
                "tension_ratio": round(tension_count / (tension_count + relax_count), 2) if (tension_count + relax_count) > 0 else 0,
                "chapter_hook_density": chapter_hook_rate,
            },
            humor_type="混合型",
            tension_relax_pattern=pace_type,
        )
        self.style_profiles[novel_title] = profile
        self.save_state()
        return asdict(profile)
    
    def _analyze_sections(self, texts: list, metas: list) -> list:
        """分析各段落统计。"""
        sections = []
        n = len(texts)
        if n == 0:
            return sections
        
        segments = [
            ("开篇(前10%)", texts[:max(1, n//10)]),
            ("发展(40%-50%)", texts[max(1, n*2//5):max(2, n//2)]),
            ("高潮(70%-80%)", texts[max(1, n*7//10):max(2, n*4//5)]),
            ("收尾(后10%)", texts[max(1, n*9//10):]),
        ]
        
        for label, section_texts in segments:
            if not section_texts:
                continue
            total_chars = sum(len(t) for t in section_texts)
            dialogue_chars = 0
            sentences = []
            for t in section_texts:
                sents = re.split(r'[。！？\n]', t)
                sentences.extend(len(s.strip()) for s in sents if s.strip())
                for dre in self.dialogue_re_list:
                    for m in dre.finditer(t):
                        dialogue_chars += len(m.group(1))
            avg_sl = round(sum(sentences) / len(sentences), 1) if sentences else 0
            d_ratio = round(dialogue_chars / total_chars, 3) if total_chars else 0
            
            sample_idx = n//10 if label.startswith("开篇") else (n//2 if label.startswith("发展") else (n*3//4 if label.startswith("高潮") else n*19//20))
            sample_idx = min(sample_idx, n-1)
            sample_text = texts[sample_idx][:800] if sample_idx < len(texts) else ""
            sample_ch = metas[sample_idx].get("chapter_title", "") if sample_idx < len(metas) else ""
            sections.append({
                "section": label,
                "chunk_range": f"{max(1, sample_idx+1)}/{n}",
                "avg_sentence_len": avg_sl,
                "dialogue_ratio": d_ratio,
                "sample_chapter": sample_ch,
                "sample_text": sample_text,
            })
        return sections
    
    def _extract_sample_passages(self, texts: list, metas: list) -> list:
        """提取样本段落。"""
        passages = []
        for i in range(min(3, len(texts))):
            passages.append({
                "text": texts[i][:800],
                "chapter": metas[i].get("chapter_title", "") if i < len(metas) else "",
                "position": "opening"
            })
        
        mid_idx = len(texts) // 2
        for j in range(min(3, len(texts) - mid_idx)):
            i = mid_idx + j
            passages.append({
                "text": texts[i][:800],
                "chapter": metas[i].get("chapter_title", "") if i < len(metas) else "",
                "position": "climax"
            })
        
        for j in range(min(3, len(texts))):
            i = len(texts) - 3 + j
            if i >= 0:
                passages.append({
                    "text": texts[i][:800],
                    "chapter": metas[i].get("chapter_title", "") if i < len(metas) else "",
                    "position": "ending"
                })
        return passages
    
    def _analyze_ai_patterns(self, texts: list, total_chars: int) -> tuple:
        """分析 AI 写作模式。"""
        ai_markers_count = 0
        matched_humor = []
        
        for text in texts:
            for p in [re.compile(r) for r in self.humor_patterns]:
                found = p.findall(text)
                if found:
                    matched_humor.extend(found)
            for p in self.ai_patterns:
                ai_markers_count += len(p.findall(text))
        
        if ai_markers_count > 0 and total_chars > 0:
            markers_per_10k = ai_markers_count / (total_chars / 10000)
            ai_score = min(round(math.log1p(markers_per_10k) * 2.5, 2), 10.0)
        else:
            ai_score = 0.0
        
        return ai_score, matched_humor
    
    def _analyze_pacing(self, texts: list, metas: list) -> tuple:
        """分析节奏。"""
        tension_count = 0
        relax_count = 0
        chapter_end_tension = 0
        chapter_end_count = 0
        
        chapters = {}
        for i, meta in enumerate(metas):
            ch = meta.get("chapter_title", "")
            if ch:
                if ch not in chapters:
                    chapters[ch] = []
                chapters[ch].append(i)
        
        for text in texts:
            for p in [re.compile(r) for r in self.tension_patterns]:
                tension_count += len(p.findall(text))
            for p in [re.compile(r) for r in self.relax_patterns]:
                relax_count += len(p.findall(text))
        
        for ch_name, chunk_indices in chapters.items():
            if len(chunk_indices) >= 1:
                last_idx = chunk_indices[-1]
                if last_idx < len(texts):
                    last_text = texts[last_idx]
                    tail = last_text[-200:] if len(last_text) > 200 else last_text
                    tail_tension = sum(len(re.compile(p).findall(tail)) for p in self.tension_patterns)
                    tail_relax = sum(len(re.compile(p).findall(tail)) for p in self.relax_patterns)
                    if tail_tension > tail_relax:
                        chapter_end_tension += 1
                    chapter_end_count += 1
        
        chapter_hook_rate = round(chapter_end_tension / chapter_end_count, 2) if chapter_end_count > 0 else 0
        
        if tension_count + relax_count > 0:
            tr_val = tension_count / (tension_count + relax_count)
            if tr_val > 0.7:
                pace_type = "高压紧绷型"
            elif tr_val > 0.5:
                pace_type = "张弛交替型"
            else:
                pace_type = "舒缓叙事型"
        else:
            pace_type = "无法判断"
        
        return tension_count, relax_count, chapter_hook_rate, pace_type
    
    def _analyze_perspective(self, texts: list) -> str:
        """分析叙事视角。"""
        narrative_perspective = "需LLM深度分析"
        if self.chat and texts:
            try:
                samples = texts[:3]
                sample_text = "\n\n---\n\n".join(s[:800] for s in samples)
                messages = [
                    {"role": "system", "content": "你是网文分析专家。请判断以下文本的叙事人称视角，只返回以下选项之一：第一人称、第三人称限知、第三人称全知、多视角切换。"},
                    {"role": "user", "content": f"分析以下网文片段的叙事视角：\n\n{sample_text}\n\n请只返回视角类型名称。"}
                ]
                result = self.chat.chat(messages, temperature=0.1, max_tokens=32)
                if result:
                    for option in ["第一人称", "第三人称限知", "第三人称全知", "多视角切换"]:
                        if option in result:
                            narrative_perspective = option
                            break
            except Exception as e:
                logger.warning(f"Narrative perspective analysis failed: {e}")
        return narrative_perspective
