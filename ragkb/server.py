"""HTTP 服务：标准库 ``http.server`` 实现，含 SSE 流式问答与内置 Web UI。

为什么不用 FastAPI
------------------
核心目标是"零第三方依赖也能跑起来"。这个服务用 ``ThreadingHTTPServer`` 实现了
真实产品需要的几件事：

* **SSE 流式输出**（``POST /chat``）：首字延迟从"整段生成完"降到"检索完 + 首 token"
* **鉴权与行级权限**：``X-API-Key`` + ``X-User-Roles``，检索阶段就按角色过滤
* **请求体大小限制**、统一错误结构、结构化访问日志、优雅退出
* **单实例复用 RagApp**：模型 client、向量矩阵缓存、SQLite 连接都只在启动时建一次

生产部署建议：置反向代理（Nginx/Caddy）后加 TLS 与限流；把 ``RAGKB_API_KEY``
设为强随机值；多进程部署时每个 worker 各自持有 SQLite 连接（WAL 模式支持多读单写）。

接口一览::

    GET    /health                 健康检查（含语料与组件信息）
    GET    /stats                  语料/组件统计
    GET    /metrics                线上指标（trace 聚合 + 反馈）
    GET    /documents              文档列表
    POST   /documents              上传并入库 {filename, content, acl}
    DELETE /documents/<doc_id>     删除文档
    POST   /ask                    问答（非流式，返回引用与调试信息）
    POST   /chat                   问答（SSE 流式：meta → delta* → done）
    POST   /feedback               用户反馈 {trace_id, rating, comment}
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from . import __version__
from .config import Settings
from .loaders import SUPPORTED_SUFFIXES, load_file
from .obs import append_feedback
from .rag import RagApp

STATIC_DIR = Path(__file__).resolve().parent / "static"
_SAFE_NAME_RE = re.compile(r"[^\w\u4e00-\u9fff.\-]+")


class RagkbHTTPServer(ThreadingHTTPServer):
    """抑制"客户端提前断开"造成的噪声堆栈。

    健康检查探针、浏览器关页签、curl 超时都会在服务端还没读完请求行时就断开连接，
    ``BaseHTTPRequestHandler.handle_one_request`` 于是抛 ``ConnectionResetError``，
    而 ``socketserver`` 的默认行为是把**完整 traceback** 打到 stderr。实测这会刷屏，
    更糟的是它看起来像服务故障，很容易被日志告警误判。

    这类异常对 HTTP 服务是正常现象（客户端走了而已），所以静默；其它异常照旧上报。
    """

    daemon_threads = True
    CLIENT_GONE = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError)

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], self.CLIENT_GONE):
            return
        super().handle_error(request, client_address)


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = f"ragkb/{__version__}"
    protocol_version = "HTTP/1.1"

    # 由 serve() 注入
    app: RagApp
    cfg: Settings
    upload_dir: Path

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        """结构化访问日志（默认实现会打到 stderr 且没有耗时）。"""
        print(
            f"[{time.strftime('%H:%M:%S')}] {self.address_string()} "
            f"{self.command} {self.path} {fmt % args}"
        )

    def _send(self, status: int, payload: Any, *, content_type: str = "application/json") -> None:
        body = _json_bytes(payload) if content_type == "application/json" else payload
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": message, "status": int(status)})

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        limit = int(self.cfg.max_upload_mb * 1024 * 1024)
        if length > limit:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"请求体超过 {self.cfg.max_upload_mb} MB")
        return self.rfile.read(length) if length else b""

    def _json_body(self) -> dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"JSON 解析失败：{e}") from e
        if not isinstance(data, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "请求体必须是 JSON 对象")
        return data

    def _auth(self) -> tuple[str, ...] | None:
        """返回请求角色；``None`` 表示不过滤（仅当显式允许且是管理员 key）。"""
        if self.cfg.api_key:
            supplied = self.headers.get("X-API-Key") or ""
            auth = self.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                supplied = supplied or auth[7:].strip()
            if supplied != self.cfg.api_key:
                raise ApiError(HTTPStatus.UNAUTHORIZED, "缺少或错误的 API Key")
        raw = self.headers.get("X-User-Roles")
        if raw is None:
            return tuple(self.cfg.default_roles)
        if raw.strip().lower() in ("", "none", "all", "*"):
            return None if raw.strip().lower() in ("none", "all", "*") else tuple(self.cfg.default_roles)
        return tuple(p.strip() for p in raw.split(",") if p.strip())

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        t0 = time.perf_counter()
        path = urllib.parse.urlparse(self.path).path
        try:
            handler = self._route(method, path)
            handler(path)
            self.log_message("(%.0fms)", (time.perf_counter() - t0) * 1000)
        except ApiError as e:
            self._error(e.status, e.message)
            self.log_message("-> %s (%.0fms)", e.status, (time.perf_counter() - t0) * 1000)
        except BrokenPipeError:
            pass  # 客户端提前断开（流式场景很常见），不是错误
        except Exception as e:  # pragma: no cover - 兜底
            traceback.print_exc()
            try:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")
            except Exception:
                pass
            self.log_message("-> 500 (%.0fms)", (time.perf_counter() - t0) * 1000)

    def _route(self, method: str, path: str) -> Callable[[str], None]:
        table: dict[tuple[str, str], Callable[[str], None]] = {
            ("GET", "/health"): self.handle_health,
            ("GET", "/stats"): self.handle_stats,
            ("GET", "/metrics"): self.handle_metrics,
            ("GET", "/documents"): self.handle_documents_list,
            ("POST", "/documents"): self.handle_documents_upload,
            ("POST", "/ask"): self.handle_ask,
            ("POST", "/chat"): self.handle_chat_stream,
            ("POST", "/feedback"): self.handle_feedback,
        }
        if (method, path) in table:
            return table[(method, path)]
        if method == "DELETE" and path.startswith("/documents/"):
            return self.handle_document_delete
        if method == "GET" and (path == "/" or path.startswith("/static/") or path == "/favicon.ico"):
            return self.handle_static
        raise ApiError(HTTPStatus.NOT_FOUND, f"未知路由：{method} {path}")

    # ------------------------------------------------------------------
    # 处理器
    # ------------------------------------------------------------------
    def handle_health(self, path: str) -> None:
        stats = self.app.store.stats()
        self._send(
            HTTPStatus.OK,
            {
                "status": "ok",
                "version": __version__,
                "documents": stats["documents"],
                "chunks": stats["chunks"],
                "components": self.app.stats()["components"],
                "warnings": self.app.warnings(),
            },
        )

    def handle_stats(self, path: str) -> None:
        self._auth()
        self._send(HTTPStatus.OK, self.app.stats())

    def handle_metrics(self, path: str) -> None:
        self._auth()
        self._send(HTTPStatus.OK, self.app.metrics())

    def handle_documents_list(self, path: str) -> None:
        roles = self._auth()
        docs = self.app.store.list_documents()
        if roles is not None:
            docs = [d for d in docs if set(d.get("acl") or ["public"]) & set(roles)]
        self._send(HTTPStatus.OK, {"count": len(docs), "documents": docs})

    def handle_documents_upload(self, path: str) -> None:
        roles = self._auth()
        body = self._json_body()
        filename = str(body.get("filename") or "").strip()
        content = body.get("content")
        if not filename or content is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, "需要 filename 与 content 字段")
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"不支持的文件类型 {suffix}；支持：{', '.join(sorted(SUPPORTED_SUFFIXES))}",
            )

        acl = body.get("acl") or list(self.cfg.default_roles)
        if isinstance(acl, str):
            acl = [p.strip() for p in acl.split(",") if p.strip()]
        # 只允许把文档标成自己拥有的角色（或全局默认角色），防止越权上传 internal 内容。
        # default_roles 必须算作允许项，否则"只带 internal 角色"的用户连 public 文档都传不了。
        if roles is not None:
            allowed_tags = set(roles) | set(self.cfg.default_roles)
            illegal = set(acl) - allowed_tags
            if illegal:
                raise ApiError(
                    HTTPStatus.FORBIDDEN, f"无权设置这些权限标签：{sorted(illegal)}"
                )

        safe = _SAFE_NAME_RE.sub("_", Path(filename).name)[:120] or "upload.txt"
        target = self.upload_dir / f"{int(time.time())}-{safe}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_bytes(content.encode("utf-8"))
        else:
            raise ApiError(HTTPStatus.BAD_REQUEST, "content 需要是文本（如需二进制请先 base64 解码后再传文本）")

        res = load_file(target, acl=tuple(acl), title=str(body.get("title") or ""))
        if not res.ok or res.document is None:
            raise ApiError(HTTPStatus.BAD_REQUEST, res.error)
        result = self.app.ingestor.ingest_document(res.document, force=bool(body.get("force")))
        status = HTTPStatus.OK if result.status != "failed" else HTTPStatus.BAD_REQUEST
        self._send(status, result.to_dict())

    def handle_document_delete(self, path: str) -> None:
        self._auth()
        doc_id = urllib.parse.unquote(path[len("/documents/") :])
        if not doc_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "缺少 doc_id")
        ok = self.app.store.delete_document(doc_id)
        if not ok:
            raise ApiError(HTTPStatus.NOT_FOUND, f"文档不存在：{doc_id}")
        self._send(HTTPStatus.OK, {"deleted": doc_id})

    def handle_ask(self, path: str) -> None:
        roles = self._auth()
        body = self._json_body()
        question = str(body.get("question") or "").strip()
        if not question:
            raise ApiError(HTTPStatus.BAD_REQUEST, "缺少 question")
        ans = self.app.ask(
            question,
            roles=roles,
            history=body.get("history") or None,
            top_k=body.get("top_k"),
            route=body.get("route"),
        )
        self._send(HTTPStatus.OK, ans.to_dict())

    def handle_chat_stream(self, path: str) -> None:
        """SSE 流式问答。

        事件顺序：``meta``（检索结果与引用来源）→ ``delta``*（增量文本）→ ``done``
        （最终答案、引用、用量、trace_id）。前端用 fetch + ReadableStream 读取，
        刻意不用 ``EventSource``：EventSource 只能 GET，且断线会自动重连导致重复提问。
        """
        roles = self._auth()
        body = self._json_body()
        question = str(body.get("question") or "").strip()
        if not question:
            raise ApiError(HTTPStatus.BAD_REQUEST, "缺少 question")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        try:
            for event in self.app.stream(
                question,
                roles=roles,
                history=body.get("history") or None,
                top_k=body.get("top_k"),
            ):
                self.wfile.write(b"data: " + _json_bytes(event) + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # 用户关掉页面了
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def handle_feedback(self, path: str) -> None:
        self._auth()
        body = self._json_body()
        if body.get("rating") not in (-1, 0, 1, None):
            raise ApiError(HTTPStatus.BAD_REQUEST, "rating 只能是 -1 / 0 / 1")
        record = {
            "trace_id": body.get("trace_id", ""),
            "question": str(body.get("question", ""))[:500],
            "answer": str(body.get("answer", ""))[:2000],
            "rating": body.get("rating"),
            "comment": str(body.get("comment", ""))[:1000],
            "roles": list(self.headers.get("X-User-Roles", "").split(",")),
        }
        append_feedback(self.cfg.feedback_path, record)
        self._send(HTTPStatus.OK, {"ok": True})

    # ------------------------------------------------------------------
    def handle_static(self, path: str) -> None:
        rel = "index.html" if path in ("/", "/favicon.ico") else path[len("/static/") :]
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, f"静态资源不存在：{rel}")
        ctype = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".svg": "image/svg+xml",
        }.get(target.suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, target.read_bytes(), content_type=ctype)


def build_app(cfg: Settings) -> RagApp:
    return RagApp.build(cfg)


def make_server(
    cfg: Settings, app: RagApp | None = None, *, host: str = "127.0.0.1", port: int = 0
) -> RagkbHTTPServer:
    """构造（但不启动）HTTP 服务。测试用它绑定 0 号端口，避免端口冲突。"""
    app = app or build_app(cfg)
    upload_dir = Path(cfg.data_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    handler = type("BoundHandler", (Handler,), {"app": app, "cfg": cfg, "upload_dir": upload_dir})
    httpd = RagkbHTTPServer((host, port), handler)
    httpd.ragkb_app = app  # type: ignore[attr-defined]
    return httpd


def serve(cfg: Settings, *, host: str | None = None, port: int | None = None) -> int:
    host = host or cfg.api_host
    port = port or cfg.api_port
    httpd = make_server(cfg, host=host, port=port)
    app: RagApp = httpd.ragkb_app  # type: ignore[attr-defined]

    stats = app.store.stats()
    print("=" * 72)
    print(f"  ragkb {__version__}  已启动")
    print(f"  Web UI   http://{host}:{port}/")
    print(f"  语料     {stats['documents']} 篇文档 / {stats['chunks']} 个切块")
    print(f"  组件     {app.stats()['components']}")
    print(f"  鉴权     {'已开启（X-API-Key）' if cfg.api_key else '未开启（仅本机调试用）'}")
    for w in app.warnings():
        print(f"  ⚠  {w}")
    print("  Ctrl+C 退出")
    print("=" * 72)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭…")
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
    return 0
