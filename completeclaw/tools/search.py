"""Web search tools."""

from __future__ import annotations

from typing import Any, Optional

from completeclaw.tools.base import Tool, ToolResult


class WebSearchTool(Tool):
    """Stub web search tool – returns a placeholder result.

    Replace or subclass this with a real search implementation
    (e.g. Tavily, SerpAPI, DuckDuckGo).

    Parameters
    ----------
    query:
        The search query string.

    Example::

        tool = WebSearchTool()
        result = tool.run(query="Python AI frameworks 2024")
        print(result.output)
    """

    name = "web_search"
    description = (
        "Searches the web for information. "
        "Input: {'query': '<search query>'}."
    )

    def run(self, *, query: str = "", **kwargs: Any) -> ToolResult:
        return ToolResult(
            output=f"[WebSearch stub] No live results for: {query!r}. "
                   "Configure a real search backend to get actual results.",
            metadata={"query": query},
        )


class TavilySearchTool(Tool):
    """Web search powered by the Tavily API.

    Requires ``TAVILY_API_KEY`` environment variable or explicit *api_key*.

    Parameters
    ----------
    api_key:
        Tavily API key.  Falls back to ``TAVILY_API_KEY`` env var.
    max_results:
        Maximum number of results to return.

    Example::

        tool = TavilySearchTool()
        result = tool.run(query="latest AI news")
        print(result.output)
    """

    name = "tavily_search"
    description = (
        "Searches the web using the Tavily API. "
        "Input: {'query': '<search query>'}."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        max_results: int = 5,
    ) -> None:
        import os

        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._max_results = max_results

    def run(self, *, query: str = "", **kwargs: Any) -> ToolResult:
        try:
            import json
            from urllib.request import Request, urlopen

            payload = json.dumps(
                {"api_key": self._api_key, "query": query, "max_results": self._max_results}
            ).encode()
            req = Request(
                "https://api.tavily.com/search",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

            results = data.get("results", [])
            output = "\n\n".join(
                f"{r.get('title', 'No title')}\n{r.get('url', '')}\n{r.get('content', '')}"
                for r in results
            )
            return ToolResult(output=output or "No results found.", metadata={"query": query})
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))
