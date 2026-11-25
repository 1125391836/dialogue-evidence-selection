"""Minimal synchronous adapter for a configurable Chat Completions-compatible API.

No SDK or agent framework is needed for the static pilot. Actual tools are never
executed. Requests are not automatically retried, so hidden retries do not
inflate cost or masquerade as independent runs.
"""

import json
import http.client
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


class ChatClient:
    def __init__(self, model: str, base_url: str, temperature: float = 0.0, max_output_tokens: int = 512):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Invalid API base URL")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS for remote endpoints")
        self.key = os.environ.get("MODEL_API_KEY", "")
        if not self.key and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Set MODEL_API_KEY in the process environment")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def complete(self, messages: list[dict]) -> dict:
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature, "max_tokens": self.max_output_tokens}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = urllib.request.Request(self.base_url + "/chat/completions",
                                         data=json.dumps(payload).encode(), headers=headers)
        start = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.load(response)
            output = result["choices"][0]["message"]["content"]
            if not isinstance(output, str):
                raise ValueError("API returned no text content")
            return {"status": "ok", "output": output, "usage": result.get("usage", {}),
                    "returned_model": result.get("model"), "latency_seconds": time.monotonic() - start}
        except urllib.error.HTTPError as exc:
            # Do not record request headers, keys, or raw provider error bodies.
            return {"status": "api_error", "error_type": f"HTTP_{exc.code}", "output": "",
                    "usage": {}, "latency_seconds": time.monotonic() - start}
        except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as exc:
            return {"status": "api_error", "error_type": type(exc).__name__, "output": "",
                    "usage": {}, "latency_seconds": time.monotonic() - start}
