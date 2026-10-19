"""最小 HTTP 客户端（标准库 urllib）：JSON POST 与 SSE 流式读取。

不引入 ``requests`` / ``httpx``：核心价值在于让"零依赖"这个承诺成立，
同时把超时、重试、错误体透传这些工程细节显式写出来。
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

USER_AGENT = "ragkb/0.1 (+https://example.internal/ragkb)"


class HttpError(RuntimeError):
    """带状态码与响应体的 HTTP 错误，便于上层决定是否重试。"""

    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {body[:400]}")
        self.status = status
        self.url = url
        self.body = body

    @property
    def retryable(self) -> bool:
        return self.status in (408, 409, 425, 429, 500, 502, 503, 504)


def _request(
    url: str, payload: dict[str, Any] | None, headers: dict[str, str], timeout: float, method: str
) -> urllib.request.addinfourl:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    req.add_header("Accept", "application/json")
    for k, v in headers.items():
        req.add_header(k, v)
    return urllib.request.urlopen(req, timeout=timeout)


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
    retries: int = 2,
    backoff: float = 0.8,
) -> dict[str, Any]:
    """带指数退避的 JSON POST。4xx（除 408/429）不重试。"""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with _request(url, payload, headers or {}, timeout, "POST") as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            err = HttpError(e.code, url, body)
            last = err
            if not err.retryable:
                raise
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            last = e
        if attempt < retries:
            time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"请求失败（已重试 {retries} 次）: {url}: {last}")


def stream_sse(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 90.0,
) -> Iterator[dict[str, Any]]:
    """读取 OpenAI 风格 SSE 流，逐个 yield 解析后的 JSON 对象。"""
    with _request(url, payload, headers or {}, timeout, "POST") as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                return
            try:
                yield json.loads(body)
            except json.JSONDecodeError:
                continue


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}
