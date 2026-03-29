"""HTTP request tool for making web/API calls."""

from __future__ import annotations

import json as _json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from completeclaw.tools.base import Tool, ToolResult


class HttpRequestTool(Tool):
    """Makes HTTP requests using only the Python standard library.

    Supports GET, POST, PUT, PATCH, and DELETE methods with optional
    JSON or plain-text bodies and custom headers.

    Parameters
    ----------
    url:
        Target URL (must include scheme, e.g. ``https://``).
    method:
        HTTP method.  Defaults to ``"GET"``.
    headers:
        Optional dict of extra request headers.
    body:
        Optional request body.  A ``dict`` is serialised as JSON and the
        ``Content-Type`` header is set automatically; a ``str`` is sent as-is.
    timeout:
        Request timeout in seconds.  Defaults to ``10``.

    Example::

        tool = HttpRequestTool()

        # Simple GET
        result = tool.run(url="https://httpbin.org/get")
        print(result.output)

        # POST with JSON body
        result = tool.run(
            url="https://httpbin.org/post",
            method="POST",
            body={"name": "CompleteClaw"},
        )
        print(result.output)
    """

    name = "http_request"
    description = (
        "Makes an HTTP request to a URL and returns the response body. "
        "Input: {'url': '<url>', 'method': 'GET|POST|PUT|PATCH|DELETE', "
        "'headers': {<optional>}, 'body': <optional str or dict>, 'timeout': <seconds>}."
    )

    def run(
        self,
        *,
        url: str = "",
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        timeout: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        if not url:
            return ToolResult(output="", success=False, error="'url' is required.")

        req_headers: Dict[str, str] = dict(headers or {})
        data: Optional[bytes] = None

        if body is not None:
            if isinstance(body, dict):
                data = _json.dumps(body).encode()
                req_headers.setdefault("Content-Type", "application/json")
            else:
                data = str(body).encode()

        req = Request(url, data=data, headers=req_headers, method=method.upper())

        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                status = resp.status
                content_type = resp.headers.get("Content-Type", "")

            # Try to decode as UTF-8; fall back to repr for binary content
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = repr(raw)

            return ToolResult(
                output=text,
                metadata={"status": status, "content_type": content_type, "url": url},
            )
        except HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return ToolResult(
                output=body_text,
                success=False,
                error=f"HTTP {exc.code}: {exc.reason}",
                metadata={"status": exc.code, "url": url},
            )
        except URLError as exc:
            return ToolResult(output="", success=False, error=str(exc.reason))
        except Exception as exc:
            return ToolResult(output="", success=False, error=str(exc))
