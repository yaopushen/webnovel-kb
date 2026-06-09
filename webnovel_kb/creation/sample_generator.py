"""素材范例生成器——文抄公模式。

指定小说章节，通过 LLM 进行最小限度重写，保留原文风格、节奏、结构骨架，
仅修改必要信息，产出可直接参考的素材范例。
"""
import json
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("webnovel-kb")

REWRITE_PROMPTS = {
    "minimal": """你是网文素材改写助手。对以下章节进行最小限度改写：

改写规则：
1. 替换所有人名、地名、组织名为虚构名称（保持音节数一致以保持节奏）
2. 保留原文 95% 以上的句式结构
3. 保留所有情节转折点和因果链
4. 保留对话比例和心理描写密度
5. 仅修改字面信息，不改变叙事逻辑

{style_constraints}

{custom_instructions}

{preserve_section}

输出要求：
1. 直接输出改写后的正文，不要加任何前缀说明
2. 在正文后另起一行，用 "---变更记录---" 分隔，列出所有变更：
   - 人物改名：原名 → 新名
   - 地名改名：原名 → 新名
   - 其他变更：说明

原文：
{original_text}""",

    "light": """你是网文素材改写助手。对以下章节进行轻度改写：

改写规则：
1. 替换所有人名、地名、组织名为虚构名称
2. 微调环境描写（换季节、换天气、换时间）
3. 调整部分细节（数字、时间、颜色等）
4. 保留核心情节走向不变
5. 保留原文 90% 以上的句式结构

{style_constraints}

{custom_instructions}

{preserve_section}

输出要求：
1. 直接输出改写后的正文，不要加任何前缀说明
2. 在正文后另起一行，用 "---变更记录---" 分隔，列出所有变更

原文：
{original_text}""",

    "moderate": """你是网文素材改写助手。对以下章节进行中度改写：

改写规则：
1. 替换所有人名、地名、组织名
2. 调整部分情节走向（替换冲突类型、更换动机）
3. 保留叙事节奏和章节结构
4. 保留风格特征（句长、对话比例、叙述视角）
5. 保留原文 80% 以上的核心叙事骨架

{style_constraints}

{custom_instructions}

{preserve_section}

输出要求：
1. 直接输出改写后的正文，不要加任何前缀说明
2. 在正文后另起一行，用 "---变更记录---" 分隔，列出所有变更

原文：
{original_text}"""
}


class SampleGenerator:
    """素材范例生成器——文抄公模式。"""

    def __init__(self, chat, kb):
        self.chat = chat
        self.kb = kb

    async def generate(self, novel_title: str, chapter: int,
                       rewrite_level: str = "minimal",
                       custom_instructions: str = "",
                       preserve_elements: list = None) -> dict:
        """生成素材范例。

        Args:
            novel_title: 书名（模糊匹配）
            chapter: 章节号（1-based）
            rewrite_level: 重写程度 minimal/light/moderate
            custom_instructions: 自定义重写指令
            preserve_elements: 必须保留的元素列表

        Returns:
            dict with sample, changes_summary, etc.
        """
        if rewrite_level not in REWRITE_PROMPTS:
            return {"error": f"无效的重写级别: {rewrite_level}，可选: minimal/light/moderate"}

        # 1. Read original chapter
        chapter_data = self.kb.read_chapter(novel_title, chapter)
        if isinstance(chapter_data, dict) and "error" in chapter_data:
            return chapter_data

        original_text = chapter_data.get("content", "")
        if not original_text:
            return {"error": "章节内容为空"}

        exact_title = chapter_data["novel"]
        chapter_title = chapter_data.get("chapter_title", "")
        original_wc = len(original_text)

        # 2. Get style info (cached if available)
        style_constraints = ""
        try:
            if exact_title in self.kb.style_profiles:
                profile = self.kb.style_profiles[exact_title]
                style_constraints = f"""风格约束（必须保持一致）：
- 平均句长：{profile.avg_sentence_len} 字
- 对话占比：{profile.dialogue_ratio*100:.1f}%
- 叙事视角：{profile.narrative_perspective}
- 节奏类型：{profile.pace_type}"""
        except Exception:
            pass

        # 3. Build prompt
        custom_section = ""
        if custom_instructions:
            custom_section = f"额外指令：{custom_instructions}"

        preserve_section = ""
        if preserve_elements:
            preserve_section = f"必须保留的元素：{', '.join(preserve_elements)}"

        prompt = REWRITE_PROMPTS[rewrite_level].format(
            style_constraints=style_constraints,
            custom_instructions=custom_section,
            preserve_section=preserve_section,
            original_text=original_text[:15000]  # Limit to avoid token overflow
        )

        # 4. LLM rewrite
        try:
            response = await self.chat.chat(
                messages=[
                    {"role": "system", "content": "你是网文素材改写助手，擅长在保留原文风格和骨架的前提下进行改写。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max(original_wc * 2, 8192)
            )
        except Exception as e:
            return {"error": f"LLM 调用失败: {e}"}

        if not response:
            return {"error": "LLM 返回空内容"}

        # 5. Parse result
        return self._parse_result(response, chapter_data, rewrite_level, original_wc)

    def _parse_result(self, response: str, chapter_data: dict,
                      rewrite_level: str, original_wc: int) -> dict:
        """解析 LLM 返回，分离正文和变更记录。"""
        parts = response.split("---变更记录---")

        sample_text = parts[0].strip()
        changes_text = parts[1].strip() if len(parts) > 1 else ""

        # Parse changes
        changes_summary = {
            "characters_renamed": {},
            "locations_renamed": {},
            "plot_adjustments": []
        }

        if changes_text:
            for line in changes_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                if "→" in line or "->" in line:
                    arrow = "→" if "→" in line else "->"
                    parts_c = line.split(arrow)
                    if len(parts_c) == 2:
                        old = parts_c[0].strip().lstrip("- ")
                        new = parts_c[1].strip()
                        if any(kw in line for kw in ["人物", "角色", "人名"]):
                            changes_summary["characters_renamed"][old] = new
                        elif any(kw in line for kw in ["地名", "地点", "城市"]):
                            changes_summary["locations_renamed"][old] = new
                        else:
                            changes_summary["characters_renamed"][old] = new

        sample_wc = len(sample_text)

        return {
            "status": "ok",
            "novel": chapter_data["novel"],
            "chapter": chapter_data["chapter_number"],
            "chapter_title": chapter_data.get("chapter_title", ""),
            "original_word_count": original_wc,
            "sample_word_count": sample_wc,
            "rewrite_level": rewrite_level,
            "changes_summary": changes_summary,
            "sample": sample_text,
            "preserved_elements": ["对话节奏", "悬念钩子", "心理描写密度"]
        }