"""外部搜索 API 封装——知乎全网搜索、站内搜索、直答。"""
import time
from typing import List

import httpx

from webnovel_kb.utils.logging_config import get_logger

logger = get_logger("search.external")


class ExternalSearch:
    """外部搜索 API 封装。"""

    def __init__(self, access_secret: str, search_url: str,
                 global_search_url: str, zhida_url: str):
        self.access_secret = access_secret
        self.search_url = search_url
        self.global_search_url = global_search_url
        self.zhida_url = zhida_url

    async def global_search(self, query: str, n_results: int = 5) -> list:
        """调用知乎全网搜索 API 获取外部知识。"""
        if not self.access_secret:
            return [{"error": "未配置 ZHIHU_ACCESS_SECRET 环境变量"}]
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                resp = await client.get(
                    self.global_search_url,
                    headers={
                        "Authorization": f"Bearer {self.access_secret}",
                        "X-Request-Timestamp": str(int(time.time())),
                        "Content-Type": "application/json",
                    },
                    params={"Query": query, "Count": min(max(n_results, 1), 20)}
                )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("Code") != 0:
                    return [{"error": f"知乎全网搜索错误: {data.get('Message', '未知')}"}]
                items = (data.get("Data") or {}).get("Items") or []
                results = []
                for item in items:
                    results.append({
                        "title": item.get("Title", ""),
                        "url": item.get("Url", ""),
                        "content": (item.get("ContentText") or "")[:500],
                        "author": item.get("AuthorName", ""),
                        "votes": item.get("VoteUpCount", 0),
                        "comments": item.get("CommentCount", 0),
                        "type": item.get("ContentType", ""),
                    })
                return results
            else:
                logger.error(f"Zhihu global search API error: {resp.status_code} - {resp.text[:200]}")
                return [{"error": f"知乎全网搜索 HTTP {resp.status_code}"}]
        except Exception as e:
            logger.error(f"Zhihu global search failed: {e}")
            return [{"error": f"知乎全网搜索失败: {str(e)}"}]

    async def zhihu_search(self, query: str, n_results: int = 5) -> list:
        """调用知乎站内搜索 API 获取讨论和经验分享。"""
        if not self.access_secret:
            return [{"error": "未配置 ZHIHU_ACCESS_SECRET 环境变量"}]
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                resp = await client.get(
                    self.search_url,
                    headers={
                        "Authorization": f"Bearer {self.access_secret}",
                        "X-Request-Timestamp": str(int(time.time())),
                        "Content-Type": "application/json",
                    },
                    params={"Query": query, "Count": min(max(n_results, 1), 10)}
                )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("Code") != 0:
                    return [{"error": f"知乎API错误: {data.get('Message', '未知')}"}]
                items = (data.get("Data") or {}).get("Items") or []
                results = []
                for item in items:
                    results.append({
                        "title": item.get("Title", ""),
                        "url": item.get("Url", ""),
                        "content": (item.get("ContentText") or "")[:500],
                        "author": item.get("AuthorName", ""),
                        "votes": item.get("VoteUpCount", 0),
                        "comments": item.get("CommentCount", 0),
                        "type": item.get("ContentType", ""),
                    })
                return results
            else:
                logger.error(f"Zhihu search API error: {resp.status_code} - {resp.text[:200]}")
                return [{"error": f"知乎API HTTP {resp.status_code}"}]
        except Exception as e:
            logger.error(f"Zhihu search failed: {e}")
            return [{"error": f"知乎搜索失败: {str(e)}"}]

    async def zhihu_zhida(self, query: str, model: str = "zhida-fast-1p5") -> list:
        """调用知乎直答 API——用自然语言问题直接获取答案。"""
        if not self.access_secret:
            return [{"error": "未配置 ZHIHU_ACCESS_SECRET 环境变量"}]
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.post(
                    self.zhida_url,
                    headers={
                        "Authorization": f"Bearer {self.access_secret}",
                        "X-Request-Timestamp": str(int(time.time())),
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": query}],
                        "stream": False,
                    }
                )
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    answer = msg.get("content", "")
                    reasoning = msg.get("reasoning_content", "")
                    result = {"answer": answer}
                    if reasoning:
                        result["reasoning"] = reasoning[:500]
                    return [result]
                return [{"error": "知乎直答返回空内容"}]
            else:
                try:
                    err_data = resp.json()
                    err_msg = err_data.get("error", {}).get("message", resp.text[:200])
                except Exception:
                    err_msg = resp.text[:200]
                logger.error(f"Zhida API error: {resp.status_code} - {err_msg}")
                return [{"error": f"知乎直答错误: {err_msg}"}]
        except Exception as e:
            logger.error(f"Zhida failed: {e}")
            return [{"error": f"知乎直答失败: {str(e)}"}]
