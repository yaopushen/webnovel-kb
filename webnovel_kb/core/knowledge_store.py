"""Agent knowledge store — read/write knowledge layer for agent产出."""
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from webnovel_kb.utils.logging_config import get_logger

logger = get_logger("core.knowledge_store")

VALID_CATEGORIES = [
    "market_analysis", "style_guide", "writing_technique",
    "user_experience", "research", "query_insight"
]

INSIGHT_PROMPT = """你是网文写作知识库的分析助手。对比新知识与现有知识，判断关系类型。

新知识：
标题：{title}
内容：{content}

现有相关知识（top-5）：
{related_items}

请判断新知识与现有知识的关系，返回 JSON 数组，每项包含：
- type: "补充"/"冲突"/"重复"/"延伸"/"独立"
- detail: 一句话说明
- related: 关联的现有知识标题列表（如有）

只返回 JSON 数组，不要其他文字。"""


class KnowledgeStore:
    """Agent 知识层——可读写的创作知识库。"""

    CLEANUP_THRESHOLD = 10  # 每存入 N 条触发整理
    CLEANUP_DEBOUNCE_SEC = 60  # 最小整理间隔
    DEDUP_THRESHOLD = 0.95  # 语义去重阈值
    ARCHIVE_DAYS = 30  # 过期降权天数

    def __init__(self, collection, embedding_fn, chat, state_manager):
        self.collection = collection  # agent_knowledge ChromaDB collection
        self.embedding_fn = embedding_fn
        self.chat = chat
        self.state_manager = state_manager

        # Load knowledge_meta from state
        state_file = state_manager.data_dir / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {}
        self._meta = state.get("knowledge_meta", {
            "total_added": 0,
            "added_since_last_cleanup": 0,
            "last_cleanup_at": None,
            "cleanup_count": 0
        })
        self._last_cleanup_time = 0
        self._cleanup_in_progress = False

    async def add(self, content: str, title: str, category: str = "research",
                  tags: list = None, source: str = "",
                  analyze: bool = True, auto_generated: bool = False) -> dict:
        """存入知识，可选洞察分析。"""
        if category not in VALID_CATEGORIES:
            category = "research"

        tags = tags or []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry_id = f"ak_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}"

        # Index in ChromaDB
        try:
            meta = {
                "entry_id": entry_id,
                "title": title,
                "category": category,
                "tags": json.dumps(tags),
                "source": source,
                "created_at": now,
                "auto_generated": str(auto_generated),
                "archived": "false"
            }
            if self.embedding_fn:
                emb = (await self.embedding_fn([content]))[0]
                self.collection.add(
                    ids=[entry_id],
                    documents=[content],
                    metadatas=[meta],
                    embeddings=[emb]
                )
            else:
                self.collection.add(
                    ids=[entry_id],
                    documents=[content],
                    metadatas=[meta]
                )
        except Exception as e:
            logger.error(f"Failed to index knowledge: {e}")
            return {"error": f"存入失败: {e}"}

        # Update meta
        self._meta["total_added"] += 1
        self._meta["added_since_last_cleanup"] += 1
        self._save_meta()

        result = {
            "status": "ok",
            "id": entry_id,
            "title": title,
            "insights": [],
            "total_knowledge": self._meta["total_added"]
        }

        # Optional insight analysis
        if analyze and self.chat:
            try:
                insights = await self._analyze_insight(content, title)
                result["insights"] = insights
                result["related_count"] = len(insights)
            except Exception as e:
                logger.warning(f"Insight analysis failed: {e}")
                result["note"] = f"洞察分析失败: {e}"
        else:
            result["note"] = "已跳过洞察分析"

        # Trigger cleanup if threshold reached
        await self._maybe_cleanup()

        return result

    async def search(self, query: str, category: str = "",
                     n_results: int = 10, archived: bool = False) -> list:
        """语义搜索 agent 知识。"""
        if self.collection.count() == 0:
            return []

        where = {}
        if category:
            where["category"] = category
        if not archived:
            where["archived"] = "false"

        try:
            query_params = {"n_results": n_results}
            if where:
                query_params["where"] = where

            if self.embedding_fn:
                query_vec = (await self.embedding_fn([query]))[0]
                query_params["query_embeddings"] = [query_vec]
            else:
                query_params["query_texts"] = [query]

            results = self.collection.query(**query_params)
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return []

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        output = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            dist = results["distances"][0][i] if results.get("distances") else 0
            output.append({
                "id": meta.get("entry_id", ""),
                "title": meta.get("title", ""),
                "content": doc[:300],
                "category": meta.get("category", ""),
                "source": meta.get("source", ""),
                "score": round(1 - dist, 4)
            })
        return output

    def list_all(self, category: str = "", limit: int = 100) -> list:
        """列出所有 agent 知识。"""
        if self.collection.count() == 0:
            return []

        where = {}
        if category:
            where["category"] = category

        try:
            get_params = {"include": ["documents", "metadatas"], "limit": limit}
            if where:
                get_params["where"] = where
            results = self.collection.get(**get_params)
        except Exception as e:
            logger.error(f"Knowledge list failed: {e}")
            return []

        if not results or not results.get("ids"):
            return []

        output = []
        for i, entry_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            doc = results["documents"][i] if results.get("documents") else ""
            output.append({
                "id": entry_id,
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                "tags": json.loads(meta.get("tags", "[]")),
                "source": meta.get("source", ""),
                "created_at": meta.get("created_at", ""),
                "content_preview": doc[:100],
                "archived": meta.get("archived", "false") == "true"
            })
        return output

    async def _analyze_insight(self, content: str, title: str) -> list:
        """LLM 对比新知识与现有知识，返回洞察。"""
        # Search for related existing knowledge
        related = await self.search(query=content[:500], n_results=5)

        if not related:
            return [{"type": "独立", "detail": "知识库中暂无相关内容，这是全新领域", "related": []}]

        related_items = "\n".join(
            f"- [{r['title']}]({r['category']}): {r['content'][:100]}"
            for r in related
        )

        prompt = INSIGHT_PROMPT.format(
            title=title,
            content=content[:1000],
            related_items=related_items
        )

        try:
            response = await self.chat.chat(
                messages=[
                    {"role": "system", "content": "你是知识库分析助手，擅长对比分析知识间的关系。只返回JSON。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024
            )
            if response:
                # Parse JSON from response
                response = response.strip()
                if response.startswith("```"):
                    response = response.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse insight JSON: {response[:200] if response else 'None'}")
        except Exception as e:
            logger.warning(f"Insight LLM call failed: {e}")

        return [{"type": "独立", "detail": "洞察分析未返回有效结果", "related": []}]

    async def _maybe_cleanup(self):
        """检查是否需要触发整理，若需要则提交任务到 Worker。"""
        if self._meta["added_since_last_cleanup"] < self.CLEANUP_THRESHOLD:
            return

        now = time.time()
        if now - self._last_cleanup_time < self.CLEANUP_DEBOUNCE_SEC:
            return

        # Submit task to worker
        try:
            from webnovel_kb.worker import submit_task
            from webnovel_kb.config import DATA_DIR
            submit_task(
                queue_dir=DATA_DIR / "task_queue",
                task_type="knowledge_cleanup",
                params={}
            )
            self._last_cleanup_time = now
            self._meta["added_since_last_cleanup"] = 0
            self._save_meta()
            logger.info("Successfully submitted knowledge cleanup task to worker")
        except Exception as e:
            logger.error(f"Failed to submit cleanup task to worker: {e}")

    async def _run_cleanup(self):
        """执行知识整理：去重 + 过期降权。"""
        try:
            logger.info("Starting knowledge cleanup...")
            self._last_cleanup_time = time.time()

            archived_count = 0

            # 1. Archive old query_insight entries
            if self.collection.count() > 0:
                results = self.collection.get(
                    where={"category": "query_insight", "archived": "false"},
                    include=["metadatas"]
                )
                if results and results.get("ids"):
                    cutoff = datetime.now() - timedelta(days=self.ARCHIVE_DAYS)
                    for i, entry_id in enumerate(results["ids"]):
                        meta = results["metadatas"][i]
                        created = meta.get("created_at", "")
                        try:
                            created_dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S")
                            if created_dt < cutoff:
                                self.collection.update(
                                    ids=[entry_id],
                                    metadatas=[{**meta, "archived": "true"}]
                                )
                                archived_count += 1
                        except ValueError:
                            pass

            # Update meta
            self._meta["added_since_last_cleanup"] = 0
            self._meta["last_cleanup_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._meta["cleanup_count"] += 1
            self._save_meta()

            logger.info(f"Knowledge cleanup done: archived {archived_count} old entries")
        except Exception as e:
            logger.error(f"Knowledge cleanup failed: {e}")
        finally:
            self._cleanup_in_progress = False

    def _save_meta(self):
        """保存 knowledge_meta 到 state.json。"""
        state_file = self.state_manager.data_dir / "state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
        else:
            state = {}
        state["knowledge_meta"] = self._meta
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
