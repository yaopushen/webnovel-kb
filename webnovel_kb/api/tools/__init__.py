"""MCP 工具分组注册。"""
from webnovel_kb.api.tools.browse import register_browse_tools
from webnovel_kb.api.tools.search import register_search_tools
from webnovel_kb.api.tools.outline import register_outline_tools
from webnovel_kb.api.tools.analysis import register_analysis_tools
from webnovel_kb.api.tools.knowledge import register_knowledge_tools
from webnovel_kb.api.tools.creation import register_creation_tools
from webnovel_kb.api.tools.task import register_task_tools


def register_all_tools(mcp, kb, safe, safe_async):
    """注册所有 MCP 工具。"""
    register_browse_tools(mcp, kb, safe)
    register_search_tools(mcp, kb, safe_async)
    register_outline_tools(mcp, kb, safe, safe_async)
    register_analysis_tools(mcp, kb, safe_async)
    register_knowledge_tools(mcp, kb, safe_async)
    register_creation_tools(mcp, kb, safe_async)
    register_task_tools(mcp, kb, safe)
