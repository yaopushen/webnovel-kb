"""统一的小说模糊匹配工具函数。"""
from typing import Optional, Tuple


def resolve_novel(novels: dict, title: str) -> Tuple[Optional[str], Optional[str]]:
    """解析小说标题，返回 (exact_title, novel_id)。

    支持精确匹配和模糊匹配（title in n.title）。
    找不到时返回 (None, None)。
    """
    for nid, n in novels.items():
        if n.title == title or title in n.title:
            return n.title, nid
    return None, None


def resolve_novel_title(novels: dict, title: str) -> str:
    """将模糊书名解析为精确书名。找不到时返回原始输入。"""
    exact, _ = resolve_novel(novels, title)
    return exact if exact else title
