"""智能搜索引擎——LLM 函数调用模式的多轮搜索。"""
import asyncio
import json
from typing import Optional, Callable, Awaitable

from webnovel_kb.utils.logging_config import get_logger

logger = get_logger("search.smart")


class SmartSearchEngine:
    """智能搜索引擎——LLM 驱动的多轮工具调用搜索。"""

    def __init__(self, chat, unified_search, knowledge_store,
                 external_search, search_knowledge_fn, resolve_fn):
        """
        Args:
            chat: RemoteChatClient 实例
            unified_search: UnifiedSearch 实例
            knowledge_store: KnowledgeStore 实例
            external_search: ExternalSearch 实例
            search_knowledge_fn: async callable(query, ...) 搜索知识库
            resolve_fn: callable(title) -> str 解析书名
        """
        self.chat = chat
        self.unified_search = unified_search
        self.knowledge_store = knowledge_store
        self.external_search = external_search
        self._search_knowledge = search_knowledge_fn
        self._resolve_title = resolve_fn

    def _build_system_prompt(self, novels: dict) -> str:
        """构建 system prompt。"""
        novel_list = [f"{n.title}({n.author}/{n.genre})" for n in novels.values()]
        novel_info = "\n".join(f"  - {n}" for n in novel_list) if novel_list else "无"
        genre_list = sorted(set(n.genre for n in novels.values()))

        return f"""你是网文写作研究助手。你可以调用搜索工具获取知识库中的原文、情节模式、实体信息，以及外部网络和知乎社区的知识。

当前知识库包含 {len(novels)} 本小说，类型包括 {', '.join(genre_list)}。
可用小说：
{novel_info}

工作流程：
1. 分析用户查询意图
2. 调用合适的工具获取数据（可以并行调用多个工具）
   - 内部知识库：search_text / search_patterns
   - 外部知识：web_search（全网搜索）/ zhihu_search（知乎站内搜索）/ zhihu_zhida（知乎直答，用自然语言问题直接获取答案）
   - ⚠️ zhihu_search 必须使用2-5个简短关键词（如"网文 金手指 套路"），禁止用完整问句或长句，否则命中率极低。若第一轮无结果，缩短关键词再试
   - zhihu_zhida 适合复杂、开放式问题，可以直接用自然语言提问
3. 基于原始数据，用你的知识分析提炼
4. 返回有价值的分析结果

注意：
- 每次搜索返回的结果不会太多，如果第一轮没找到满意结果，可以换个角度再搜
- 引用原文时要标注出处（书名、章节）
- 知乎搜索结果自带赞同数和评论数，可借此判断内容可信度

输出格式要求（非常重要）：
- 纯文字，不要用 Markdown 格式
- 不要用表格、标题、加粗、斜体、代码块
- 短句为主，一句一个意思
- 只用中文标点符号（。，、！？）
- 分段用空行，列举用数字或顿号
- 输出要有干货，精炼直接
- 思考过程聚焦搜索策略，不要写成复述结果"""

    def _build_tools(self, genre_list: list) -> list:
        """构建 LLM 工具定义。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": "在全部小说正文中搜索文本内容。适合查找具体描写、场景、对话、情节等。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词或自然语言描述"},
                            "mode": {
                                "type": "string",
                                "enum": ["hybrid", "semantic", "bm25"],
                                "description": "hybrid=语义+关键词混合(推荐), semantic=模糊概念, bm25=精确关键词"
                            },
                            "novel": {"type": "string", "description": "限定书名，留空搜全部"},
                            "genre": {"type": "string", "description": f"限定类型: {', '.join(genre_list)}"},
                            "n_results": {"type": "integer", "description": "返回几条", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_patterns",
                    "description": "搜索已提取的情节模式——悬念链、伏笔、反转、高潮等叙事手法。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索描述"},
                            "type_filter": {
                                "type": "string",
                                "description": "模式类型: 悬念链/跨距伏笔/反转铺垫/情感爆发点/世界观展开/力量体系引入/角色弧光/高潮设计/节奏控制/对比映衬/身份揭示"
                            },
                            "novel": {"type": "string", "description": "限定书名"},
                            "n_results": {"type": "integer", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "搜索互联网获取网文写作相关的外部知识——套路分析、写作技巧、行业趋势、读者偏好等。当知识库内部搜索不足以回答问题时使用。⚠️请使用简短关键词组合，搜索引擎对短词命中率更高。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "简短关键词组合（2-5个词，空格分隔）。例如'网文 爽点 公式'而非'网文的爽点公式是什么'"},
                            "n_results": {"type": "integer", "description": "返回几条", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "zhihu_search",
                    "description": "知乎站内搜索——查找网文写作相关的知乎讨论、经验分享、套路拆解、行业观点等。⚠️必须使用2-5个简短关键词组合（空格分隔），如'网文 金手指 写作技巧'，禁止使用完整问句或长句。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "简短关键词组合（2-5个词，空格分隔）。例如'网文 金手指 套路'而非'主角获得金手指时的反应有哪些'"},
                            "n_results": {"type": "integer", "description": "返回几条", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "zhihu_zhida",
                    "description": "知乎直答——用自然语言问题直接获取知乎AI的答案，适合复杂、开放式问题。与 zhihu_search 的区别：zhihu_search 需要简短关键词搜索帖子，zhihu_zhida 可以直接用完整问题提问并获得结构化回答。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "自然语言问题，可以写成完整的问句。例如'网文主角获得金手指时的典型反应描写有哪些'"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    async def search(self, query: str, n_results: int = 5,
                     novel_filter: Optional[str] = None,
                     genre_filter: Optional[str] = None,
                     output_format: str = "compact",
                     novels: dict = None) -> dict:
        """智能搜索——LLM 函数调用模式。"""
        if not self.chat:
            return {
                "error": "智能搜索需要配置全能 LLM 模型",
                "hint": "请设置 LLM_CHAT_BASE_URL 和 LLM_CHAT_MODEL 环境变量"
            }

        novels = novels or {}
        genre_list = sorted(set(n.genre for n in novels.values()))
        system_prompt = self._build_system_prompt(novels)
        tools = self._build_tools(genre_list)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        MAX_ROUNDS = 200
        thinking_chain = []

        for round_num in range(MAX_ROUNDS):
            try:
                raw = await self.chat.chat_raw(
                    messages=messages,
                    temperature=0.2,
                    max_tokens=8192,
                    tools=tools,
                    tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"Smart search LLM call failed (round {round_num}): {e}")
                return {
                    "error": f"LLM 调用异常: {type(e).__name__}: {str(e)}",
                    "fallback": await self.unified_search.search(
                        query, mode="hybrid", n_results=n_results,
                        novel_filter=novel_filter, genre_filter=genre_filter,
                        output_format=output_format
                    )
                }

            if not raw:
                return {
                    "error": "LLM 返回空（未知原因）",
                    "fallback": await self.unified_search.search(
                        query, mode="hybrid", n_results=n_results,
                        novel_filter=novel_filter, genre_filter=genre_filter,
                        output_format=output_format
                    )
                }

            if raw.get("_error"):
                status_code = raw.get("status_code", 0)
                api_message = raw.get("message", "未知错误")
                api_detail = raw.get("detail", "")
                retry_after = raw.get("retry_after")

                error_info = f"{api_message}"
                if status_code:
                    error_info = f"HTTP {status_code} — {api_message}"
                if api_detail:
                    error_info += f"\n详情: {api_detail[:200]}"
                if retry_after:
                    error_info += f"\n建议等待 {retry_after} 秒后重试"

                if status_code == 429:
                    error_info = f"LLM API 限流 (429) — 请求过于频繁"
                    if retry_after:
                        error_info += f"，建议等待 {retry_after} 秒后重试"
                    elif api_detail:
                        error_info += f"\n详情: {api_detail[:200]}"
                elif status_code >= 500:
                    error_info = f"LLM API 服务端错误 (HTTP {status_code})"
                    if api_detail:
                        error_info += f"\n详情: {api_detail[:200]}"
                    error_info += "\n建议稍后重试"

                logger.error(f"Smart search LLM error (round {round_num}): {error_info}")
                return {
                    "error": error_info,
                    "fallback": await self.unified_search.search(
                        query, mode="hybrid", n_results=n_results,
                        novel_filter=novel_filter, genre_filter=genre_filter,
                        output_format=output_format
                    )
                }

            choice = raw["choices"][0]
            msg = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "")

            if finish_reason == "stop" and not msg.get("tool_calls"):
                answer = msg.get("content") or msg.get("reasoning_content", "")
                await self._auto_save_insight(query, answer)
                return {
                    "query": query,
                    "思考链": self._filter_thinking_chain(thinking_chain, output_format),
                    "结果": answer
                }

            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                answer = msg.get("content") or msg.get("reasoning_content", "")
                await self._auto_save_insight(query, answer)
                return {
                    "query": query,
                    "思考链": self._filter_thinking_chain(thinking_chain, output_format),
                    "结果": answer
                }

            round_reasoning = msg.get("reasoning_content", "")
            messages.append(msg)

            async def _exec_one_tool(tc):
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError as e:
                    func_args = {}
                    logger.warning(f"  [smart_search] {func_name}: JSON解析失败: {e}")
                result_data = ""
                result_preview = ""
                try:
                    if func_name == "search_text":
                        sub_results = await self.unified_search.search(
                            query=func_args.get("query", query),
                            mode=func_args.get("mode", "hybrid"),
                            n_results=func_args.get("n_results", n_results),
                            novel_filter=func_args.get("novel"),
                            genre_filter=func_args.get("genre"),
                            output_format="compact",
                            max_content_length=300
                        )
                        result_data = json.dumps(sub_results, ensure_ascii=False)
                        if isinstance(sub_results, list) and sub_results:
                            first = sub_results[0]
                            result_preview = (first[:200] + "...") if len(first) > 200 else first
                        else:
                            result_preview = "(无结果)"
                    elif func_name == "search_patterns":
                        resolved_novel = self._resolve_title(func_args.get("novel", "")) if func_args.get("novel") else None
                        sub_results = await self._search_knowledge(
                            query=func_args.get("query", ""),
                            knowledge_type="plot_patterns",
                            n_results=func_args.get("n_results", 5),
                            type_filter=func_args.get("type_filter"),
                            source_novel=resolved_novel,
                            output_format="compact",
                            max_content_length=300
                        )
                        result_data = json.dumps(sub_results, ensure_ascii=False)
                        if isinstance(sub_results, list) and sub_results:
                            first = sub_results[0]
                            result_preview = (first[:200] + "...") if len(first) > 200 else first
                        else:
                            result_preview = "(无结果)"
                    elif func_name == "web_search":
                        sub_results = await self.external_search.global_search(
                            query=func_args.get("query", query),
                            n_results=func_args.get("n_results", 5)
                        )
                        result_data = json.dumps(sub_results, ensure_ascii=False)
                        if isinstance(sub_results, list) and sub_results:
                            first = sub_results[0]
                            if isinstance(first, dict):
                                title = first.get("title", "")
                                snippet = first.get("snippet", "") or first.get("content", "")
                                preview_text = f"{title}: {snippet}"
                                result_preview = (preview_text[:200] + "...") if len(preview_text) > 200 else preview_text
                            else:
                                result_preview = str(first)[:200]
                        else:
                            result_preview = "(无结果)"
                    elif func_name == "zhihu_search":
                        sub_results = await self.external_search.zhihu_search(
                            query=func_args.get("query", query),
                            n_results=func_args.get("n_results", 5)
                        )
                        result_data = json.dumps(sub_results, ensure_ascii=False)
                        if isinstance(sub_results, list) and sub_results:
                            first = sub_results[0]
                            if isinstance(first, dict):
                                title = first.get("title", "")
                                snippet = first.get("snippet", "") or first.get("content", "")
                                preview_text = f"{title}: {snippet}"
                                result_preview = (preview_text[:200] + "...") if len(preview_text) > 200 else preview_text
                            else:
                                result_preview = str(first)[:200]
                        else:
                            result_preview = "(无结果)"
                    elif func_name == "zhihu_zhida":
                        sub_results = await self.external_search.zhihu_zhida(
                            query=func_args.get("query", query)
                        )
                        result_data = json.dumps(sub_results, ensure_ascii=False)
                        if isinstance(sub_results, list) and sub_results:
                            first = sub_results[0]
                            if isinstance(first, dict):
                                answer = first.get("answer", "")
                                result_preview = (answer[:200] + "...") if len(answer) > 200 else answer
                            else:
                                result_preview = str(first)[:200]
                        else:
                            result_preview = "(无结果)"
                    else:
                        result_data = json.dumps({"error": f"未知工具: {func_name}"})
                        result_preview = f"(未知工具: {func_name})"
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    logger.error(f"  [smart_search] {func_name} 执行失败: {error_type}: {error_msg}")
                    result_data = json.dumps({
                        "error": f"{error_type}: {error_msg}",
                        "tool": func_name,
                        "query": func_args.get("query", "")[:100],
                    }, ensure_ascii=False)
                    result_preview = f"(执行失败: {error_msg[:100]})"
                logger.debug(f"  [smart_search] {func_name}({func_args.get('query','')[:60]}) -> {len(result_data)} chars")
                return func_name, func_args, {"role": "tool", "tool_call_id": tc["id"], "content": result_data}, result_preview

            round_summary = []
            tool_results = []
            round_previews = []
            if len(tool_calls) == 1:
                fn, fa, tr, preview = await _exec_one_tool(tool_calls[0])
                round_summary.append(f"{fn}({fa.get('query','')[:60]})")
                tool_results.append(tr)
                round_previews.append(preview)
            else:
                results = await asyncio.gather(
                    *[_exec_one_tool(tc) for tc in tool_calls],
                    return_exceptions=True
                )
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        tc = tool_calls[i]
                        func_name = tc["function"]["name"]
                        error_type = type(result).__name__
                        error_msg = str(result)
                        logger.error(f"  [smart_search] 并行执行 {func_name} 失败: {error_type}: {error_msg}")
                        round_summary.append(f"{func_name}(ERROR: {error_type})")
                        round_previews.append(f"(并行执行失败: {error_msg[:100]})")
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": json.dumps({
                                "error": f"并行执行失败: {error_type}: {error_msg}",
                                "tool": func_name,
                            }, ensure_ascii=False)
                        })
                    else:
                        fn, fa, tr, preview = result
                        round_summary.append(f"{fn}({fa.get('query','')[:60]})")
                        tool_results.append(tr)
                        round_previews.append(preview)

            thinking_chain.append({
                "round": round_num + 1,
                "思考": round_reasoning,
                "调用": round_summary,
                "结果快照": round_previews
            })

            messages.extend(tool_results)

        answer = msg.get("content") or msg.get("reasoning_content", "") or "模型尚未生成最终答案"
        await self._auto_save_insight(query, answer)
        return {
            "query": query,
            "思考链": self._filter_thinking_chain(thinking_chain, output_format),
            "结果": answer
        }

    def _filter_thinking_chain(self, thinking_chain: list, output_format: str) -> list:
        """根据输出格式过滤思考链。"""
        if output_format == "clean":
            return []
        if output_format == "raw":
            return thinking_chain

        compact_chain = []
        for r in thinking_chain:
            compact_round = {
                "round": r.get("round"),
                "思考": r.get("思考"),
                "调用": r.get("调用"),
            }
            previews = r.get("结果快照", [])
            valid_previews = [p for p in previews if p and p != "(无结果)"]
            if valid_previews:
                compact_round["结果快照"] = f"(已精简，共命中 {len(previews)} 条记录)"
            else:
                compact_round["结果快照"] = "(无结果)"
            compact_chain.append(compact_round)
        return compact_chain

    async def _auto_save_insight(self, query: str, answer: str):
        """自动将 smart_search 洞察存入 agent_knowledge。"""
        if not answer or len(answer.strip()) < 50:
            return
        if "未找到" in answer[:20] and len(answer) < 100:
            return
        try:
            import jieba
            tags = [w for w in jieba.cut(query) if len(w) > 1][:5]
            await self.knowledge_store.add(
                content=answer,
                title=f"搜索洞察：{query[:50]}",
                category="query_insight",
                tags=tags,
                source="smart_search auto",
                analyze=False,
                auto_generated=True
            )
        except Exception as e:
            logger.debug(f"Auto-save insight failed: {e}")
