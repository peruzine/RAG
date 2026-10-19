"""HTTP 接口测试：在真实 socket 上跑（0 号端口），覆盖鉴权、SSE、上传、权限越权。

用真实 HTTP 而不是 mock，是因为流式输出、Content-Length、连接关闭这些细节
恰恰是 mock 测不到的——而它们正是线上最容易出问题的地方。
"""

from __future__ import annotations

import json
import shutil
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from tests.helpers import build_app
from tests.helpers import ROOT  # noqa: F401

from ragkb.server import Handler, make_server


class ApiCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 测试里不需要访问日志，否则 CI 日志会被淹没
        cls._orig_log = Handler.log_message
        Handler.log_message = lambda *a, **k: None  # type: ignore[method-assign]
        cls.app, cls.cfg = build_app("api")
        cls.httpd: ThreadingHTTPServer = make_server(cls.cfg, cls.app, port=0)
        cls.port = cls.httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.app.close()
        Handler.log_message = cls._orig_log  # type: ignore[method-assign]
        shutil.rmtree(cls.cfg.data_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def request(self, path: str, *, method: str = "GET", body=None, headers=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            return resp.status, (json.loads(raw) if "json" in ctype else raw.decode("utf-8"))

    def request_error(self, path: str, *, method: str = "GET", body=None, headers=None):
        try:
            self.request(path, method=method, body=body, headers=headers)
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))
        raise AssertionError("期望抛出 HTTPError，但请求成功了")


class TestBasicEndpoints(ApiCase):
    def test_health_needs_no_auth(self):
        status, data = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")
        self.assertGreater(data["chunks"], 0)
        self.assertIn("components", data)

    def test_index_page_served(self):
        status, html = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("企业知识库问答", html)

    def test_stats_and_metrics(self):
        _, stats = self.request("/stats")
        self.assertIn("chunks", stats)
        _, metrics = self.request("/metrics")
        self.assertIn("traces", metrics)
        self.assertIn("feedback", metrics)

    def test_unknown_route_404(self):
        code, data = self.request_error("/nope")
        self.assertEqual(code, 404)
        self.assertIn("未知路由", data["error"])

    def test_static_path_traversal_blocked(self):
        code, _ = self.request_error("/static/../../config.py")
        self.assertEqual(code, 404)


class TestClientDisconnects(ApiCase):
    """客户端提前断开是 HTTP 服务的日常，不该打 traceback（会被日志告警误判为故障）。"""

    def test_reset_errors_are_suppressed(self):
        import contextlib
        import io

        for exc in (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, TimeoutError):
            try:
                raise exc("client went away")
            except exc:
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    self.httpd.handle_error(None, ("127.0.0.1", 0))
                self.assertEqual(buf.getvalue(), "", f"{exc.__name__} 应该被静默")

    def test_real_errors_still_reported(self):
        import contextlib
        import io

        try:
            raise ValueError("真的有问题")
        except ValueError:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                self.httpd.handle_error(None, ("127.0.0.1", 0))
            self.assertIn("ValueError", buf.getvalue())

    def test_abrupt_disconnect_does_not_kill_server(self):
        """发一半请求就断开，服务必须还能继续正常响应。"""
        import socket

        sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        sock.sendall(b"GET /health HTTP/1.1\r\nHost: x\r\n")  # 故意不发空行结束
        sock.close()
        status, data = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")


class TestAsk(ApiCase):
    def test_ask_returns_citations(self):
        status, data = self.request("/ask", method="POST", body={"question": "一线城市住宿每晚上限是多少？"})
        self.assertEqual(status, 200)
        self.assertFalse(data["refused"])
        self.assertTrue(data["citations"])
        self.assertIn("700", data["text"])
        self.assertIn("retrieved", data)

    def test_ask_requires_question(self):
        code, data = self.request_error("/ask", method="POST", body={})
        self.assertEqual(code, 400)
        self.assertIn("question", data["error"])

    def test_ask_refuses_unknown_topic(self):
        _, data = self.request("/ask", method="POST", body={"question": "公司年会在哪里举办？"})
        self.assertTrue(data["refused"])

    def test_roles_header_enforced(self):
        _, data = self.request(
            "/ask", method="POST", body={"question": "P6 职级带宽上限是多少？"},
            headers={"X-User-Roles": "public"},
        )
        self.assertTrue(data["refused"])

    def test_role_escalation_rejected_on_upload(self):
        code, data = self.request_error(
            "/documents", method="POST",
            body={"filename": "x.md", "content": "# T\n\n机密内容", "acl": ["internal"]},
            headers={"X-User-Roles": "public"},
        )
        self.assertEqual(code, 403)
        self.assertIn("无权设置", data["error"])


class TestChatStream(ApiCase):
    def test_sse_event_sequence(self):
        status, text = self.request(
            "/chat", method="POST", body={"question": "餐补每天多少钱？"}
        )
        self.assertEqual(status, 200)
        events = [
            json.loads(line[5:].strip())
            for line in text.splitlines()
            if line.startswith("data:") and "[DONE]" not in line
        ]
        self.assertEqual(events[0]["type"], "meta")
        self.assertEqual(events[-1]["type"], "done")
        self.assertTrue(any(e["type"] == "delta" for e in events))
        self.assertIn("contexts", events[0])
        self.assertTrue(events[-1]["answer"]["text"])

    def test_sse_requires_question(self):
        code, _ = self.request_error("/chat", method="POST", body={})
        self.assertEqual(code, 400)


class TestDocuments(ApiCase):
    def test_upload_list_delete_roundtrip(self):
        status, data = self.request(
            "/documents", method="POST",
            body={"filename": "新制度.md", "content": "# 备用金制度\n\n单笔备用金上限 5000 元，需主管审批。\n", "acl": ["public"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "added")
        self.assertGreaterEqual(data["n_chunks"], 1)
        doc_id = data["doc_id"]

        _, listing = self.request("/documents")
        self.assertIn(doc_id, [d["doc_id"] for d in listing["documents"]])

        _, answer = self.request("/ask", method="POST", body={"question": "备用金单笔上限是多少？"})
        self.assertFalse(answer["refused"])
        self.assertIn("5000", answer["text"])

        _, deleted = self.request(f"/documents/{doc_id}", method="DELETE")
        self.assertEqual(deleted["deleted"], doc_id)
        _, listing2 = self.request("/documents")
        self.assertNotIn(doc_id, [d["doc_id"] for d in listing2["documents"]])

    def test_upload_unsupported_type(self):
        code, data = self.request_error(
            "/documents", method="POST", body={"filename": "x.exe", "content": "abc"}
        )
        self.assertEqual(code, 400)
        self.assertIn("不支持", data["error"])

    def test_upload_missing_fields(self):
        code, _ = self.request_error("/documents", method="POST", body={"filename": "a.md"})
        self.assertEqual(code, 400)

    def test_delete_unknown_document(self):
        code, _ = self.request_error("/documents/doc-nope", method="DELETE")
        self.assertEqual(code, 404)

    def test_oversized_body_rejected(self):
        original = self.cfg.max_upload_mb
        self.cfg.max_upload_mb = 0.001  # 1KB
        try:
            code, data = self.request_error(
                "/documents", method="POST", body={"filename": "big.md", "content": "x" * 5000}
            )
            self.assertEqual(code, 413)
            self.assertIn("超过", data["error"])
        finally:
            self.cfg.max_upload_mb = original

    def test_documents_list_filtered_by_role(self):
        # 带 internal 角色的用户上传一篇"仅内部可见"的文档
        _, data = self.request(
            "/documents", method="POST",
            body={"filename": "内部制度.md", "content": "# 内部\n\n仅内部可见的内容。", "acl": ["internal"]},
            headers={"X-User-Roles": "internal"},
        )
        doc_id = data["doc_id"]
        _, public_view = self.request("/documents", headers={"X-User-Roles": "public"})
        self.assertNotIn(doc_id, [d["doc_id"] for d in public_view["documents"]])
        _, internal_view = self.request("/documents", headers={"X-User-Roles": "public,internal"})
        self.assertIn(doc_id, [d["doc_id"] for d in internal_view["documents"]])
        self.request(f"/documents/{doc_id}", method="DELETE")


class TestFeedback(ApiCase):
    def test_feedback_written(self):
        status, data = self.request(
            "/feedback", method="POST",
            body={"trace_id": "abc", "question": "q", "answer": "a", "rating": 1, "comment": "好"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        lines = self.cfg.feedback_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(json.loads(lines[-1])["trace_id"], "abc")

    def test_invalid_rating(self):
        code, _ = self.request_error("/feedback", method="POST", body={"rating": 5})
        self.assertEqual(code, 400)


class TestAuth(ApiCase):
    def test_api_key_enforced(self):
        self.cfg.api_key = "s3cret"
        try:
            code, data = self.request_error("/stats")
            self.assertEqual(code, 401)
            self.assertIn("API Key", data["error"])
            status, _ = self.request("/stats", headers={"X-API-Key": "s3cret"})
            self.assertEqual(status, 200)
            status2, _ = self.request("/stats", headers={"Authorization": "Bearer s3cret"})
            self.assertEqual(status2, 200)
            # 健康检查仍然免鉴权（负载均衡探针需要）
            self.assertEqual(self.request("/health")[0], 200)
        finally:
            self.cfg.api_key = ""

    def test_roles_none_means_no_filter(self):
        _, data = self.request(
            "/ask", method="POST", body={"question": "P0 故障响应时间"},
            headers={"X-User-Roles": "none"},
        )
        self.assertFalse(data["refused"])


if __name__ == "__main__":
    unittest.main()
