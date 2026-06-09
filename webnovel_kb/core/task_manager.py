"""异步任务管理——提取任务的启动、查询、取消。"""
import asyncio
import threading
import uuid
from typing import Dict

from webnovel_kb.utils.logging_config import get_logger
from webnovel_kb.utils.exceptions import ExtractionError
from webnovel_kb.utils.novel_resolver import resolve_novel

logger = get_logger("core.task_manager")


class TaskManager:
    """异步任务管理器。"""

    def __init__(self, novels: dict, query_cache, plot_extractor, entity_extractor,
                 template_extractor, scene_extractor):
        self.novels = novels
        self._query_cache = query_cache
        self.plot_extractor = plot_extractor
        self.entity_extractor = entity_extractor
        self.template_extractor = template_extractor
        self.scene_extractor = scene_extractor
        self._async_tasks: Dict[str, dict] = {}
        self._task_lock = threading.Lock()

    def start_async_extraction(self, novel_title: str, max_chunks: int = 20,
                               extract_type: str = "plot_patterns") -> dict:
        """启动异步提取任务。"""
        exact_title, novel_id = resolve_novel(self.novels, novel_title)
        if not exact_title:
            raise ExtractionError(f"未找到小说: {novel_title}")

        task_id = str(uuid.uuid4())[:8]
        self._async_tasks[task_id] = {
            "status": "running",
            "novel": exact_title,
            "extract_type": extract_type,
            "progress": 0,
            "result": None
        }

        def _run():
            try:
                with self._task_lock:
                    self._async_tasks[task_id]["progress"] = 10

                def progress_cb(current, total):
                    pct = 10 + int(current / total * 80)
                    with self._task_lock:
                        if task_id in self._async_tasks:
                            self._async_tasks[task_id]["progress"] = min(pct, 90)

                if extract_type == "plot_patterns":
                    result = asyncio.run(self.plot_extractor.extract_cross_chunk(
                        exact_title, novel_id, exact_title, max_chunks, progress_cb
                    ))
                elif extract_type == "entities":
                    result = asyncio.run(self.entity_extractor.extract_cross_chunk(
                        exact_title, novel_id, exact_title, max_chunks, progress_cb
                    ))
                else:
                    result = asyncio.run(self._extract_async(exact_title, extract_type, max_chunks, False))

                with self._task_lock:
                    if task_id in self._async_tasks:
                        self._async_tasks[task_id]["status"] = "completed"
                        self._async_tasks[task_id]["progress"] = 100
                        self._async_tasks[task_id]["result"] = result

                self._query_cache.clear()
            except Exception as e:
                with self._task_lock:
                    if task_id in self._async_tasks:
                        self._async_tasks[task_id]["status"] = "error"
                        self._async_tasks[task_id]["error"] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"task_id": task_id, "status": "started", "novel": exact_title}

    async def _extract_async(self, novel_title: str, extract_type: str = "all",
                max_chunks: int = 20, cross_chunk: bool = False) -> dict:
        """提取知识（异步版本）。"""
        exact_title, novel_id = resolve_novel(self.novels, novel_title)
        if not exact_title:
            raise ExtractionError(f"未找到小说: {novel_title}")

        results = {"novel": exact_title}

        if extract_type in ["entities", "all"]:
            if cross_chunk:
                results["entities"] = await self.entity_extractor.extract_cross_chunk(
                    novel_title, novel_id, exact_title, max_chunks
                )
            else:
                results["entities"] = await self.entity_extractor.extract(
                    novel_title, novel_id, exact_title, max_chunks
                )

        if extract_type in ["plot_patterns", "all"]:
            if cross_chunk:
                results["plot_patterns"] = await self.plot_extractor.extract_cross_chunk(
                    novel_title, novel_id, exact_title, max_chunks
                )
            else:
                results["plot_patterns"] = await self.plot_extractor.extract(
                    novel_title, novel_id, exact_title, max_chunks
                )

        if extract_type in ["writing_templates", "all"]:
            results["writing_templates"] = await self.template_extractor.extract(
                novel_title, novel_id, exact_title, max_chunks
            )

        if extract_type in ["scene_patterns", "all"]:
            results["scene_patterns"] = await self.scene_extractor.extract(
                novel_title, novel_id, exact_title, max_chunks
            )

        self._query_cache.clear()
        return results

    def get_task_status(self, task_id: str) -> dict:
        """获取任务状态。"""
        with self._task_lock:
            return self._async_tasks.get(task_id, {"error": f"任务不存在: {task_id}"})

    def cancel_task(self, task_id: str) -> dict:
        """取消正在运行的任务。"""
        with self._task_lock:
            if task_id not in self._async_tasks:
                return {"error": f"任务不存在: {task_id}"}
            task = self._async_tasks[task_id]
            if task["status"] != "running":
                return {"error": f"任务 {task_id} 状态为 {task['status']}，无法取消"}
            task["cancel"] = True
            return {"status": "cancelling", "task_id": task_id}
