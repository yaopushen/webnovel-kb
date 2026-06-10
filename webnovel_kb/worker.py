"""Worker process — background task consumer for long-running operations.

Usage:
    python -m webnovel_kb.worker

The worker polls $WEBNOVEL_KB_DATA/task_queue/ for JSON task files,
executes them, and writes results back.
"""
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

from webnovel_kb.utils.logging_config import get_logger

logger = get_logger("worker")


class Worker:
    """后台任务消费者——文件系统队列。"""

    POLL_INTERVAL = 2  # seconds

    def __init__(self, kb, queue_dir: Path):
        self.kb = kb
        self.queue_dir = queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.running = True

    async def run(self):
        """主循环：轮询队列，执行长时任务。"""
        logger.info(f"Worker started, polling {self.queue_dir}")
        while self.running:
            try:
                task = self._dequeue()
                if task:
                    await self._execute(task)
                else:
                    await asyncio.sleep(self.POLL_INTERVAL)
            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(self.POLL_INTERVAL)

    def _dequeue(self):
        """从队列目录读取最旧的待执行任务。"""
        pending = sorted(
            self.queue_dir.glob("*.pending.json"),
            key=lambda f: f.stat().st_mtime
        )
        if not pending:
            return None

        task_file = pending[0]
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                task = json.load(f)
            task["_file"] = str(task_file)
            # Move to running state
            task_id = task.get("task_id", "unknown")
            running_file = self.queue_dir / f"{task_id}.running.json"
            task_file.rename(running_file)
            task["_file"] = str(running_file)
            return task
        except Exception as e:
            logger.error(f"Failed to read task {task_file}: {e}")
            task_file.unlink(missing_ok=True)
            return None

    async def _execute(self, task):
        """执行任务并写回结果。"""
        task_type = task.get("type", "unknown")
        task_id = task.get("task_id", "unknown")
        task_file = Path(task["_file"])

        logger.info(f"Executing task {task_id} ({task_type})")
        start_time = time.time()

        try:
            if task_type == "extract_outline_batch":
                result = await self._exec_outline_batch(task)
            elif task_type == "knowledge_cleanup":
                result = await self._exec_knowledge_cleanup(task)
            elif task_type == "generate_draft":
                result = await self._exec_generate_draft(task)
            else:
                result = {"error": f"Unknown task type: {task_type}"}

            elapsed = round(time.time() - start_time, 1)
            result["elapsed_seconds"] = elapsed

            # Write completed status
            done_file = self.queue_dir / f"{task_id}.done.json"
            with open(done_file, "w", encoding="utf-8") as f:
                json.dump({
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)

            # Remove running file
            task_file.unlink(missing_ok=True)
            logger.info(f"Task {task_id} completed in {elapsed}s")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            error_file = self.queue_dir / f"{task_id}.error.json"
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump({
                    "task_id": task_id,
                    "status": "error",
                    "error": str(e),
                    "failed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
            task_file.unlink(missing_ok=True)

    async def _exec_outline_batch(self, task):
        """执行批量章纲提取。"""
        params = task.get("params", {})
        novel_title = params.get("novel_title")
        start_chapter = params.get("start_chapter")
        end_chapter = params.get("end_chapter")
        task_id = task.get("task_id")

        if not all([novel_title, start_chapter, end_chapter]):
            return {"error": "Missing required params"}

        def progress_cb(current, total):
            pct = int(current / total * 100)
            self._update_progress(task_id, pct)

        def is_cancelled():
            return self._check_cancel(task_id)

        result = await self.kb.outline_extractor.extract_batch(
            novel_title, start_chapter, end_chapter,
            progress_callback=progress_cb,
            is_cancelled=is_cancelled
        )
        return result

    async def _exec_knowledge_cleanup(self, task):
        """执行知识整理。"""
        await self.kb.knowledge_store._run_cleanup()
        return {"status": "cleanup_done"}

    async def _exec_generate_draft(self, task):
        """执行初稿生成任务。"""
        params = task.get("params", {})
        if hasattr(self.kb, 'draft_generator'):
            return await self.kb.draft_generator.generate(**params)
        return {"error": "DraftGenerator not initialized"}

    def _update_progress(self, task_id, pct):
        """更新任务进度到文件。"""
        progress_file = self.queue_dir / f"{task_id}.progress.json"
        try:
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump({"task_id": task_id, "progress": pct}, f)
        except Exception:
            pass

    def _check_cancel(self, task_id):
        """检查任务是否被取消。"""
        cancel_file = self.queue_dir / f"{task_id}.cancel"
        return cancel_file.exists()


def submit_task(queue_dir: Path, task_type: str, params: dict, task_id: str = None) -> str:
    """提交任务到队列（主进程调用）。"""
    task_id = task_id or str(uuid.uuid4())[:8]
    task_file = queue_dir / f"{task_id}.pending.json"
    queue_dir.mkdir(parents=True, exist_ok=True)
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump({
            "task_id": task_id,
            "type": task_type,
            "params": params,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)
    return task_id


def get_task_status(queue_dir: Path, task_id: str) -> dict:
    """查询任务状态（主进程调用）。"""
    # Check done
    done_file = queue_dir / f"{task_id}.done.json"
    if done_file.exists():
        with open(done_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Check error
    error_file = queue_dir / f"{task_id}.error.json"
    if error_file.exists():
        with open(error_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Check progress
    progress_file = queue_dir / f"{task_id}.progress.json"
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
        return {"task_id": task_id, "status": "running", "progress": progress_data.get("progress", 0)}

    # Check pending
    pending_file = queue_dir / f"{task_id}.pending.json"
    if pending_file.exists():
        return {"task_id": task_id, "status": "pending", "progress": 0}

    # Check running
    running_file = queue_dir / f"{task_id}.running.json"
    if running_file.exists():
        return {"task_id": task_id, "status": "running", "progress": 0}

    return {"task_id": task_id, "status": "not_found"}


def cancel_task(queue_dir: Path, task_id: str) -> dict:
    """取消任务（主进程调用）。"""
    cancel_file = queue_dir / f"{task_id}.cancel"
    cancel_file.touch()
    return {"status": "cancelling", "task_id": task_id}


async def main():
    """Worker 入口。"""
    from webnovel_kb.config import DATA_DIR
    from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase

    logger.info("Initializing Worker...")
    kb = WebNovelKnowledgeBase(data_dir=DATA_DIR)
    queue_dir = DATA_DIR / "task_queue"
    worker = Worker(kb, queue_dir)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
