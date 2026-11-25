from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from threading import Thread
import unittest
from unittest.mock import patch

from context_bundles.provider import ChatClient


class ProviderTests(unittest.TestCase):
    def test_real_http_serialization_against_local_stub(self):
        observed = {}
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                observed["path"] = self.path
                observed["request"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                raw = json.dumps({"model": "local-stub", "choices": [{"message": {
                    "content": '{"name":"notify","arguments":{"id":"C17"}}'}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 8}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            def log_message(self, *args):
                pass
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {"MODEL_API_KEY": ""}):
                client = ChatClient("local-stub", f"http://127.0.0.1:{server.server_port}/v1")
                result = client.complete([{"role": "user", "content": "test"}])
            self.assertEqual(observed["path"], "/v1/chat/completions")
            self.assertEqual(observed["request"]["model"], "local-stub")
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["usage"]["prompt_tokens"], 12)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
