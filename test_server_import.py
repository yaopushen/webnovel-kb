#!/usr/bin/env python3
"""Quick import test for v1.6 on server."""
try:
    from webnovel_kb.utils.exceptions import *
    from webnovel_kb.utils.logging_config import setup_logging, get_logger
    from webnovel_kb.utils.query_cache import QueryCache
    from webnovel_kb.config import *
    from webnovel_kb.core.knowledge_base import WebNovelKnowledgeBase
    from webnovel_kb.api.mcp_tools import MCPTools
    print("ALL IMPORTS OK")
except Exception as e:
    print(f"IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()
