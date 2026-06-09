"""Chapter outline extraction from novels — standalone module.

章纲提取独立模块：串行读取章节 → LLM 提取章纲 → 存入知识库 → 封存状态。
调用 kb 的 read_chapter、save_outline、chat 方法完成全流程。
"""
import asyncio
import logging
import time
from typing import Optional

from webnovel_kb.utils.exceptions import ExtractionError

logger = logging.getLogger("webnovel-kb")

CHAPTER_OUTLINE_PROMPT = """-Goal-
你是资深网文编辑。将本章正文压缩为一段连贯的叙事摘要，用作后续创作参考。摘要要覆盖本章所有关键情节转折点，保持叙事流畅。

-写法要求-
1. 用一段连贯的叙事文字概括全章，不设小标题、不编号、不分段
2. 保持原文的叙事节奏和因果链：谁做了什么→发生了什么→结果如何
3. 保留所有角色名和关键地名、功法名、物品名
4. 保留所有推动剧情的关键事件，省略无关的环境描写和心理独白
5. 章末如有钩子或悬念，自然融入叙事末尾
6. 输出长度控制在原文的 15%-25%

-示例（仅供参考格式，非内容模板）-
正文：某章节
章纲：路胜穿越到古代世界成为九连城富家公子路大公子，在酒坊桂花坊喝酒听曲，撞见几个壮汉谈论水鬼作祟。他花一两银子从光头汉子手中买下一块自称水鬼骨头的绿色玉石。刚出酒坊，玉石在手心融化为一团暗绿色粘液，传出惨叫后炸开成绿烟——玉石变成普通鹅卵石。路胜心头发毛，隐约意识到这个世界绝非普通的古代中国。回府路上，城门守备闲聊提到一道人已出手击杀水鬼，死亡过程与他手中玉石的异象完全吻合。

-注意-
- 只输出章纲正文，不要加任何前缀、后缀或说明文字
- 角色名必须来自原文，不要编造"""


class ChapterOutlineExtractor:
    """章纲提取器 —— 独立模块，调用 kb 内部方法完成读取→提取→存储→封存全流程。

    设计原则：
    - 串行处理，一次一章，避免竞态和并发写
    - LLM 调用失败自动重试一次
    - 读取失败或 LLM 返回空则跳过该章并记录
    - 批量完成后统一封存（save_state）
    """

    MAX_RETRIES = 1

    def __init__(self, chat, kb):
        """初始化提取器。

        Args:
            chat: RemoteChatClient 实例
            kb: WebNovelKnowledgeBase 实例，用于 read_chapter / save_outline / _save_state
        """
        self.chat = chat
        self.kb = kb

    async def extract_single(self, novel_title: str, chapter: int,
                       save: bool = True) -> dict:
        """提取单章章纲。

        Args:
            novel_title: 书名（模糊匹配）
            chapter: 章节号（1-based）
            save: 是否自动存入知识库，默认 True

        Returns:
            dict: {novel, chapter, chapter_title, outline, word_count, saved, error?}
        """
        chapter_data = self.kb.read_chapter(novel_title, chapter)
        if isinstance(chapter_data, dict) and "error" in chapter_data:
            return {
                "novel": novel_title,
                "chapter": chapter,
                "error": chapter_data["error"]
            }

        content = chapter_data.get("content", "")
        if not content:
            return {
                "novel": chapter_data.get("novel", novel_title),
                "chapter": chapter,
                "error": "章节内容为空"
            }

        exact_title = chapter_data["novel"]
        chapter_title = chapter_data.get("chapter_title", "")
        word_count = chapter_data.get("word_count", 0)

        outline = await self._call_llm_for_outline(content, chapter_title, word_count)
        if not outline:
            return {
                "novel": exact_title,
                "chapter": chapter,
                "chapter_title": chapter_title,
                "error": "LLM 提取失败：返回空内容"
            }

        result = {
            "novel": exact_title,
            "chapter": chapter,
            "chapter_title": chapter_title,
            "outline": outline,
            "word_count": word_count,
            "saved": False
        }

        if save:
            try:
                save_result = await self.kb.save_outline(
                    exact_title,
                    [{"chapter": chapter, "content": outline, "outline_type": "章纲"}],
                    overwrite=True
                )
                if save_result.get("error_count", 0) > 0:
                    result["save_error"] = save_result.get("errors")
                else:
                    result["saved"] = True
            except Exception as e:
                logger.warning(f"章纲保存失败 第{chapter}章: {e}")
                result["save_error"] = str(e)

        return result

    async def extract_batch(self, novel_title: str, start_chapter: int,
                      end_chapter: int, progress_callback=None,
                      is_cancelled=None) -> dict:
        """串行批量提取章纲。一次一章，提取后自动存入。

        Args:
            novel_title: 书名
            start_chapter: 起始章节号
            end_chapter: 结束章节号（含）
            progress_callback: 可选，callable(current, total)
            is_cancelled: 可选，callable() -> bool，返回 True 则中止提取

        Returns:
            dict: 封存结果，{novel, total_chapters, success_count, error_count, ...}
        """
        if start_chapter > end_chapter:
            raise ExtractionError(f"起始章节({start_chapter})不能大于结束章节({end_chapter})")

        total = end_chapter - start_chapter + 1
        logger.info(f"开始串行提取章纲: {novel_title} 第{start_chapter}-{end_chapter}章 共{total}章")

        results = []
        start_time = time.time()
        cancelled = False

        for i, ch in enumerate(range(start_chapter, end_chapter + 1)):
            if is_cancelled and is_cancelled():
                logger.info(f"章纲提取任务已取消，已完成 {i}/{total} 章")
                cancelled = True
                break

            logger.info(f"提取章纲 [{i+1}/{total}] 第{ch}章...")
            result = await self.extract_single(novel_title, ch, save=True)
            results.append(result)

            if result.get("error"):
                error_msg = result["error"]
                logger.warning(f"第{ch}章提取失败: {error_msg[:100]}")

            if progress_callback:
                progress_callback(i + 1, total)

            if i < total - 1:
                await asyncio.sleep(0.3)

        elapsed = time.time() - start_time
        success = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]

        exact_title = None
        for r in success:
            exact_title = r.get("novel")
            if exact_title:
                break
        if not exact_title:
            exact_title = self.kb.resolve_novel_title(novel_title)

        self.kb._save_state()

        logger.info(
            f"章纲批量提取完成: {novel_title} 成功{len(success)}/{total} "
            f"失败{len(failed)} 耗时{elapsed:.1f}秒"
        )

        return {
            "novel": exact_title,
            "total_chapters": total,
            "success_count": len(success),
            "error_count": len(failed),
            "chapters_extracted": [r["chapter"] for r in success],
            "elapsed_seconds": round(elapsed, 1),
            "cancelled": cancelled,
            "errors": [
                {"chapter": r["chapter"], "error": r.get("error", "未知")}
                for r in failed
            ] if failed else None
        }

    async def _call_llm_for_outline(self, content: str, chapter_title: str,
                              word_count: int) -> Optional[str]:
        """调用 LLM 提取章纲，失败自动重试一次。

        Args:
            content: 章节正文
            chapter_title: 章节标题
            word_count: 章节字数

        Returns:
            章纲叙事文本，失败返回 None
        """
        if not self.chat:
            logger.error("Chat API 未配置，无法提取章纲")
            return None

        messages = [
            {
                "role": "system",
                "content": "你是资深网文编辑，擅长从章节中提取结构化的章纲。输出严格按格式，精炼准确。"
            },
            {
                "role": "user",
                "content": f"{CHAPTER_OUTLINE_PROMPT}\n\n章节标题：{chapter_title}\n字数：{word_count}\n\n正文：\n{content}"
            }
        ]

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                response = await self.chat.chat(
                    messages, temperature=0.2, max_tokens=4096
                )
                if response and response.strip():
                    return response.strip()
                logger.warning(
                    f"LLM 返回空内容 (attempt {attempt+1}/{self.MAX_RETRIES+1}): "
                    f"{chapter_title}"
                )
            except Exception as e:
                logger.error(
                    f"LLM 调用异常 (attempt {attempt+1}/{self.MAX_RETRIES+1}): "
                    f"{type(e).__name__}: {e}"
                )
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(1.0)

        return None