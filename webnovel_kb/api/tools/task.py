"""任务类工具：manage_task。"""
import asyncio


def register_task_tools(mcp, kb, safe):

    @mcp.tool()
    async def manage_task(task_id: str, action: str = "status") -> dict:
        """Manage background asynchronous tasks (query status or cancel)."""
        if action == "cancel":
            return await asyncio.to_thread(safe, "cancel_task", kb.cancel_task, task_id)
        return await asyncio.to_thread(safe, "get_task_status", kb.get_task_status, task_id)
