"""Web-search tool stub.

The actual HTTP request requires an API key (Tavily, Serper, Brave, …).
This module provides a ready-to-subclass interface plus a stub that can be
used in unit tests or wired up to any search back-end.
"""

from __future__ import annotations

import abc
import json
import urllib.parse
import urllib.request
from typing import Any, List, Optional

from completeclaw.tools.base import Tool, ToolResult


class WebSearchTool(Tool):
    """Search the web and return a list of results.

    This is a *stub* implementation – it returns an empty result set.
    Subclass it and override :meth:`_search` to connect to a real API.

    Parameters
    ----------
    api_key:
        Optional API key for the search back-end.
    num_results:
        Maximum number of results to return per query.

    Example::

        class TavilySearch(WebSearchTool):
            def _search(self, query, num_results):
                # call Tavily API ...
                return [{"title": "...", "url": "...", "snippet": "..."}]

        tool = TavilySearch(api_key="tvly-...")
        result = tool.run(query="latest LLM benchmarks")
        print(result.output)
    """

    name = "web_search"
    description = (
        "Search the web for up-to-date information.  "
        "Input: {\"query\": \"<search query>\"}."
    )

    def __init__(self, *, api_key: Optional[str] = None, num_results: int = 5) -> None:
        self._api_key = api_key
        self._num_results = num_results

    def run(self, *, query: str = "", **kwargs: Any) -> ToolResult:  # type: ignore[override]
        if not query:
            return ToolResult(output="", success=False, error="No query provided.")
        try:
            results = self._search(query, self._num_results)
            if not results:
                return ToolResult(output="No results found.", metadata={"query": query})
            lines = []
            for i, item in enumerate(results, 1):
                lines.append(
                    f"{i}. {item.get('title', 'No title')}\n"
                    f"   {item.get('url', '')}\n"
                    f"   {item.get('snippet', '')}"
                )
            return ToolResult(
                output="\n\n".join(lines),
                metadata={"query": query, "num_results": len(results)},
            )
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))

    def _search(self, query: str, num_results: int) -> List[dict]:
        """Override this in subclasses to call a real search API.

        Returns a list of dicts with keys ``title``, ``url``, ``snippet``.
        """
        return []


class TavilySearchTool(WebSearchTool):
    """Web search powered by the Tavily API.

    Requires a Tavily API key (https://tavily.com).
    Falls back to the ``TAVILY_API_KEY`` environment variable.
    """

    name = "web_search"

    def __init__(self, *, api_key: Optional[str] = None, num_results: int = 5) -> None:
        import os

        super().__init__(
            api_key=api_key or os.environ.get("TAVILY_API_KEY"),
            num_results=num_results,
        )

    def _search(self, query: str, num_results: int) -> List[dict]:
        url = "https://api.tavily.com/search"
        payload = json.dumps(
            {"api_key": self._api_key, "query": query, "max_results": num_results}
        ).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            data = json.loads(resp.read())
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
            for r in data.get("results", [])
        ]
