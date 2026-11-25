"""DeepSeek adapter: JSON output, explicit thinking mode, complete usage accounting."""

import http.client
import json
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .credentials import load_deepseek


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect refused", headers, fp)


class DeepSeekClient:
    def __init__(self, config: dict, key: str | None = None):
        base = config["base_url"].rstrip("/")
        parsed = urlparse(base)
        if parsed.scheme != "https" or parsed.hostname != "api.deepseek.com" or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.port not in (None, 443):
            raise ValueError("DeepSeek credentials may only be sent to https://api.deepseek.com")
        self.base = base
        self.config = config.copy()
        self.key = load_deepseek() if key is None else key
        self.opener = urllib.request.build_opener(NoRedirect())

    def payload(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        return {"model": self.config["model"], "messages": messages,
                "temperature": self.config.get("temperature", 0),
                "max_tokens": max_tokens or self.config.get("max_output_tokens", 512),
                "thinking": {"type": self.config.get("thinking", "disabled")},
                "response_format": {"type": "json_object"}, "stream": False}

    def complete(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        payload = self.payload(messages, max_tokens)
        request = urllib.request.Request(self.base + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with self.opener.open(request, timeout=self.config.get("timeout_seconds", 90)) as response:
                result = json.load(response)
            choice = result["choices"][0]
            text = choice["message"].get("content")
            if not isinstance(text, str):
                raise ValueError("Missing text output")
            return {"status": "ok", "output": text, "usage": result.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"), "returned_model": result.get("model"),
                    "provider_id": result.get("id"), "provider_created": result.get("created"),
                    "system_fingerprint": result.get("system_fingerprint"),
                    "latency_seconds": time.monotonic() - start}
        except urllib.error.HTTPError as exc:
            return {"status": "api_error", "output": "", "usage": {},
                    "error_type": f"HTTP_{exc.code}", "latency_seconds": time.monotonic() - start}
        except (OSError, http.client.HTTPException, ValueError, KeyError, IndexError, TypeError) as exc:
            return {"status": "api_error", "output": "", "usage": {},
                    "error_type": type(exc).__name__, "latency_seconds": time.monotonic() - start}


def cost_interval(usage: dict, price: dict) -> dict:
    """Use off-peak/peak bounds instead of inventing the provider's billing timezone."""
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        return {"usd_low": None, "usd_high": None, "kind": "missing_usage"}
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None or miss is None:
        # Cache breakdown unknown: lower bound all hits, upper bound all misses.
        low_in = usage["prompt_tokens"] * price["input_hit_low"]
        high_in = usage["prompt_tokens"] * price["input_miss_high"]
    else:
        low_in = hit * price["input_hit_low"] + miss * price["input_miss_low"]
        high_in = hit * price["input_hit_high"] + miss * price["input_miss_high"]
    return {"usd_low": (low_in + usage["completion_tokens"] * price["output_low"]) / 1e6,
            "usd_high": (high_in + usage["completion_tokens"] * price["output_high"]) / 1e6,
            "kind": "documented_price_interval_not_invoice"}
