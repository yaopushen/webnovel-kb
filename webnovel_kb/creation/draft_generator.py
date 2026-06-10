"""初稿生成器——模仿模式与创造模式。

模仿模式：在保留原文风格和核心骨架的前提下进行中度改写（只保留原改动最大的 moderate 模式）。
创造模式：使用 mimo-v2.5-pro 思考链模型，加载硬编码的写作模板与指定的真实章节作为双重风格种子，
          并根据 Agent 基于章纲提供的 prompt 创作全新的章节初稿。
"""
import os
import json
import httpx
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger("webnovel-kb")

# ============ 模仿模式 (Imitate) 改写 Prompt ============
IMITATE_PROMPT = """你是网文素材重写大师。对以下章节进行 moderate 级别的深度重写：

重写规则：
1. 替换所有人名、地名、组织名。
2. 严禁进行机械的词汇对拷或简单的同义词替换。你必须用全新的词汇、句式结构和环境细节重新描写所有情节，但保持背后的故事脉络、转折点和骨架不变。
3. 调整部分情节走向或细节（如：冲突的发生方式、角色的具体动作、动机等），展现出丰富的联想力。
4. 保留原文的叙事节奏和章节结构，保持相似的心理描写密度与对话比例。
5. 确保重写后的内容具有网文阅读的人气与爽快感，而不是生硬的机器改写。

{style_constraints}

{custom_instructions}

{preserve_section}

输出要求：
1. 直接输出重写后的正文，不要加任何前缀说明。
2. 在正文后另起一行，用 "---变更记录---" 分隔，列出所有重要的情节与名称变更。

原文：
{original_text}"""

# ============ 创造模式 (Create) 预置与风格种子 ============
DEVELOPER_MSG = """\
老者："你想报仇？"
少年："我被强者反复侮辱，被师尊视为垃圾，我怎么可能不想报仇？"
老者摸了摸少年的脑袋，叹道："好孩子，我来传功给你吧。"
少年惊道："前辈！这怎么行？"
老者伸出手："把你手机给我。"
少年看着手机上的变化，震惊道："前辈！这哪里来的百年功力？"
老者微微一笑："好孩子，这是你在天庭的备用功力，以后急用的时候随用随取，别再被人侮辱了。"
少年皱眉："这不是法力贷吗？我怕……"
老者："天庭是大平台，新用户借百年功力有30天免息，日息最低半天功力，还没你吐纳一周天多。"
……张羽冷哼一声，关掉了上面的广告。\
"""

SYSTEM_MSG = """\
按网文节奏扩充，只输出小说正文。不输出解释、点评、建议、总结、元叙述。字数3000-4000字。
允许使用以下写法：
- 《修真聊天群》第839章："错的不是我，是这个世界" —— 角色世界观台词
- 《大王遥命》第327章："不是没有这种可能" —— 双重否定表达不确定性
禁止使用以下偷懒写法：
- "不是不想，是不能" （展示角色内心挣扎型）
- "不是X，是Y"句式（否定-肯定对比水字数，没有新信息）
- "X，不是因为Y，而是因为Z"句式（解释型）。\
"""

STYLE_SEED_HARDCODED = """\
第十一章 治疗
　　“啊？”孙杰克嘴巴张到了老大，无比震惊的看着眼前这极其荒诞的一幕。“这....这也可以吗？”
　　如此生死危机，只要靠钱就可以解决吗？
　　裂着嘴笑 of 宋6对着自己扬了扬大拇指。“bro，现在知道我为什么要玩命赚钱了吧？因为钱真的可以买命啊！”
　　“那....为什么他们不用？”孙杰克转过身来指着身后废墟中，那些支离破碎的雇佣兵尸体问道。
　　“因为他们没钱啊。”
　　“没钱就该死吗？”
　　“对，没钱就该死。”
　　“没钱就该死吗？”
　　“对啊，没钱就该死。”
　　“艹”
　　看着废墟里面如同蝼蚁般死去的雇佣兵们，孙杰克对这个世界又有了新的认识。
　　“快点走吧，我只买了三分钟。”
　　借着宋6那用钱买来的时间，最终在天空红光再次亮起的那一刻，他们的双脚终于是踩在了腐烂的垃圾上。
　　三人几乎是瘫 in 垃圾上，看着远处的一切。
　　一艘艘飞船从航天母舰降下，开始有条不紊的收拾战场。
　　“不管怎么说，总算是逃出来了。”孙杰克低头再次吐了一口血水，大口大口地喘着粗气。
　　“瞧瞧，家人们，这就是公司所作所为啊。”宋6说着，再次把直播打开，仿佛此时此刻变成了战地记者。
　　“这就是公司吗？”孙杰克看着这一切喃喃自语地说道，他对公司这个名词有了全新的认知。
　　不管这个世界过去发生了什么，但是孙杰克此刻终于深刻地明白，这個世界跟当初自己生活的世界已经截然不同。
　　“杰克，你的体温正在升高，你已经高度感染了，我们必须找到医生。”塔派说着就准备带着孙杰克翻过这座垃圾山。
　　一旁的宋6却拦住了他。“bro，这里到大都会有七十多里，你带他这么走过去，尸体都臭了，我已经呼叫了治疗中心，他们马上就派飞船过来。”
　　“那你刚刚在里面怎么不呼叫？”无力的孙杰克歪着脑袋看着他，他的眼前开始发黑。
　　“what 7 you say？刚刚在干仗啊，那种情况下，你觉得治疗中心会接单吗？别人开治疗中心是为了赚钱的，又不是救死扶伤，赔本的买卖当然不会做。”
　　“艹”听到这话，孙杰克对这个世界彻底绝望了。
　　很快一道从天而降的红色激光在他们面前划出了一个方形区域，紧接着一艘画着红十字的白色飞船，穿破云层精准地落在那里。
　　紧接着两位白大褂的医生，带着四个白色的高瘦机器人迅速迎来上来。
　　“区域安全，可以开始营救。”机器人整个裂开快速变形成机械担架，柔软的义肢迅速但又轻柔的把宋6跟孙杰克扶上担架。
　　昏昏沉沉中，孙杰克感觉到有什么东西扎入自己的小臂。“10314C1用户安全，接入生物信号，生命检测面板启动，强心剂已注射，多巴胺70毫克，去甲肾上腺素110，血纤维蛋白800....”
　　随着那冰冷的声音不断响起，本应该正在逐渐失去意识的孙杰克居然缓缓清醒了过来。
　　躺在床上的孙杰克低头看着自己那切开的腹部，只见那如同蟹腿般的机械臂，快速又精准的把自己身上那些狰狞伤口，从里到外全部消毒杀菌一点缝隙都不放过。
　　清洗完成后，开始迅速缝合，甚至连缝合口都如拉链一般致密，这手术精致的如同绣花一般。
　　这期间任何痛觉都感觉不到，就仿佛那些都是别人的肉一样。
　　孙杰克再次一次被这个世界的科技所震撼，自己这么重的伤居然就跟普通感冒一样被轻易救回来。
　　他还以为自己这一次受这么重的伤再被酸雨淋肯定九死一生了，没想到被救活居然如此简单。
　　“怎么样？牛B吧？哥们我可是给咱们定的尊享套餐。”
　　同样躺在孙杰克旁边，享受着相同待遇的宋6PUS不知道从哪掏出一根电子烟开始吞云吐雾起来，他那干瘪的脑袋不知道什么时候已经回归正常了。
　　“怎么样？我这下总不是什么用都没有吧？”他似乎对于之前孙杰克抱怨的话很是在意。
　　“我们现在去哪？”孙杰克看着简约雪白一片的治疗飞船内部。
　　宋6PUS一甩脏辫，“当然是去大都会了。怎么？你难不成还想跨过核爆辐射区，去蛾摩拉？”
　　瞧见孙杰克脸上的神情，宋6pus很是诧异地问道：“看来伱不是本地人？你们是哪人啊？”
　　“无可奉告。”孙杰克直接回绝了这个问题。现在什么都不了解的情况下，瞎编只能被人看出破绽。
　　“哈哈哈，无所谓，不管你们是哪人，你们还是救了我D命，混我们这行的，讲的就是一个道义，等到了大都会，我给你们好好接风洗尘。”
　　说着宋6PUS举起手中的电子烟在墙上的一个按钮上轻轻一戳，“刷”的一声，左侧的墙壁直接透明化。
　　外面还在下雨，不过地上已经没垃圾了，取而代之的是各种残垣断壁的水泥森林。
　　这种画面再加上雨天，一切都是昏暗色调，呈呈现现在他面前的是如同世界末日一般的景色。
　　外面的凄凉跟干净整洁的浮空车内部反差感无比之强。
		不用别人介绍，孙杰克也明白，这些东西应该都是智械危机之前的世界，只是没有人修复它们，它们被废弃遗忘了，就跟自己还有塔派一样。
		马上就要真正接触这个世界了，此刻孙杰克又开始有些患得患失起来，随着他向着一旁的塔派轻轻一挥手，对方靠了过去，两人开始耳语起来。
		“帮我搜索一下这个大都会是个什么样的地方，以咱们两现在黑户的身份去那地方安全不安全。”
		随着塔派轻轻一点头，他的屏幕上开始出现了不断重复的省略号。
		“大都会常驻人口3000万，进出人口流量特别大，其中鱼龙混杂什么样的人都有，根据我的计算，较小概率引起其他势力的注意。”
		“你确定吗？就你这身打扮，不会引起别人的注意？”孙杰克看着塔派浑身钢铁，非常的怀疑。"""


class DraftGenerator:
    """初稿生成器。"""

    def __init__(self, chat, kb):
        self.chat = chat
        self.kb = kb

    async def generate(self, novel_title: str, chapter: int,
                       mode: str = "imitate",
                       prompt: str = "",
                       custom_instructions: str = "",
                       preserve_elements: list = None) -> dict:
        """生成初稿。

        Args:
            novel_title: 书名（模糊匹配）
            chapter: 章节号（1-based），在 imitate 下是待改写章节，在 create 下是动态风格种子的来源章节
            mode: 运行模式: 'imitate'（模仿改写模式）或 'create'（创造初稿模式）
            prompt: 写作大纲/提示词（仅在 create 模式下生效）
            custom_instructions: 自定义指令
            preserve_elements: 必须保留的元素列表（仅在 imitate 模式下生效）

        Returns:
            dict containing draft, metadata, etc.
        """
        # 1. 读取章节原文（两种模式都需要，但用途不同）
        chapter_data = self.kb.read_chapter(novel_title, chapter)
        if isinstance(chapter_data, dict) and "error" in chapter_data:
            return chapter_data

        original_text = chapter_data.get("content", "")
        if not original_text:
            return {"error": "指定章节内容为空"}

        exact_title = chapter_data["novel"]
        chapter_title = chapter_data.get("chapter_title", "")
        original_wc = len(original_text)

        if mode == "imitate":
            return await self._generate_imitate(
                chapter_data, original_text, exact_title,
                custom_instructions, preserve_elements
            )
        elif mode == "create":
            if not prompt:
                return {"error": "创造模式下 prompt 参数（大纲/写作提示）不能为空"}
            return await self._generate_create(
                exact_title, chapter, chapter_title, original_text, prompt
            )
        else:
            return {"error": f"无效的模式: {mode}，可选 'imitate' 或 'create'"}

    async def _generate_imitate(self, chapter_data: dict, original_text: str, exact_title: str,
                                custom_instructions: str, preserve_elements: list) -> dict:
        """执行模仿改写（中度改写）。"""
        original_wc = len(original_text)
        style_constraints = ""
        try:
            if exact_title in self.kb.style_profiles:
                profile = self.kb.style_profiles[exact_title]
                style_constraints = f"风格约束（必须保持一致）：\\n- 平均句长：{profile.avg_sentence_len} 字\\n- 对话占比：{profile.dialogue_ratio*100:.1f}%\\n- 叙事视角：{profile.narrative_perspective}\\n- 节奏类型：{profile.pace_type}"
        except Exception:
            pass

        custom_section = f"额外指令：{custom_instructions}" if custom_instructions else ""
        preserve_section = f"必须保留的元素：{', '.join(preserve_elements)}" if preserve_elements else ""

        formatted_prompt = IMITATE_PROMPT.format(
            style_constraints=style_constraints,
            custom_instructions=custom_section,
            preserve_section=preserve_section,
            original_text=original_text[:15000]
        )

        try:
            response = await self.chat.chat(
                messages=[
                    {"role": "system", "content": "你是网文素材深度重写大师。你擅长在保留原文故事走向和结构的前提下，用全新、富有表现力的语言重写正文，彻底避免机械的替换。"},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.7,
                max_tokens=max(original_wc * 2, 8192)
            )
        except Exception as e:
            return {"error": f"LLM 调用失败: {e}"}

        if not response:
            return {"error": "LLM 返回空内容"}

        # 解析模仿改写结果，提取正文和变更记录
        parts = response.split("---变更记录---")
        sample_text = parts[0].strip()
        changes_text = parts[1].strip() if len(parts) > 1 else ""

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

        return {
            "status": "ok",
            "novel": exact_title,
            "chapter": chapter_data["chapter_number"],
            "chapter_title": chapter_data.get("chapter_title", ""),
            "mode": "imitate",
            "original_word_count": original_wc,
            "draft_word_count": len(sample_text),
            "changes_summary": changes_summary,
            "draft": sample_text
        }

    async def _generate_create(self, exact_title: str, chapter: int, chapter_title: str,
                               original_text: str, prompt: str) -> dict:
        """使用 mimo-v2.5-pro 创造新章节。"""
        # 导入智能搜索的配置
        from webnovel_kb.config import LLM_CHAT_API_KEY, LLM_CHAT_BASE_URL

        # 读取 mimo API 配置，默认与智能搜索模块一致
        mimo_key = os.environ.get("MIMO_API_KEY", LLM_CHAT_API_KEY)
        base_url = os.environ.get("MIMO_BASE_URL", LLM_CHAT_BASE_URL)

        if not base_url:
            mimo_url = "https://token-plan-sgp.xiaomimimo.com/v1/chat/completions"
        elif base_url.endswith("/chat/completions"):
            mimo_url = base_url
        else:
            mimo_url = f"{base_url.rstrip('/')}/chat/completions"

        # 动态拼接风格种子（硬编码经典种子 + 动态传入的章节正文作为第二风格种子）
        dynamic_seed = f"《{exact_title}》第{chapter}章 {chapter_title}\\n{original_text[:15000]}"
        combined_seed = f"{STYLE_SEED_HARDCODED}\\n\\n[新增风格种子]\\n{dynamic_seed}"

        # 四层消息结构
        messages = [
            {"role": "developer", "content": DEVELOPER_MSG},
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "assistant", "content": combined_seed},
            {"role": "user", "content": prompt},
        ]

        payload = {
            "model": "mimo-v2.5-pro",
            "messages": messages,
            "max_completion_tokens": 65536,
            "stream": True,
            "thinking": {"type": "enabled"},
            "frequency_penalty": 0.5,
            "presence_penalty": 0.7,
        }

        headers = {
            "Authorization": f"Bearer {mimo_key}",
            "Content-Type": "application/json"
        }

        draft_content = []
        reasoning_content = []

        logger.info(f"Calling mimo API for create mode, prompt length: {len(prompt)}")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(240.0, connect=10.0), trust_env=False) as client:
                async with client.stream("POST", mimo_url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        error_msg = error_body.decode("utf-8", errors="replace")
                        logger.error(f"mimo API error: {response.status_code} - {error_msg}")
                        return {"error": f"mimo API HTTP 错误 {response.status_code}: {error_msg}"}

                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        payload_str = line[len("data: "):]
                        if payload_str.strip() == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload_str)
                            delta = obj["choices"][0].get("delta", {})
                            reasoning = delta.get("reasoning_content", "")
                            content = delta.get("content", "")
                            if reasoning:
                                reasoning_content.append(reasoning)
                            if content:
                                draft_content.append(content)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as e:
            logger.error(f"mimo API call failed: {e}", exc_info=True)
            return {"error": f"调用 mimo API 异常失败: {e}"}

        draft_text = "".join(draft_content).strip()
        reasoning_text = "".join(reasoning_content).strip()

        if not draft_text:
            return {"error": "mimo API 返回的生成内容为空"}

        return {
            "status": "ok",
            "novel": exact_title,
            "chapter": chapter,
            "mode": "create",
            "draft_word_count": len(draft_text),
            "draft": draft_text,
            "thinking": reasoning_text
        }
