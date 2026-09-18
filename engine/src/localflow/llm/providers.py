"""Chat-completion providers behind one small interface.

  bundled : the llama-server the engine manages (default, fully local)
  ollama  : a running Ollama instance
  openai  : any OpenAI-compatible endpoint (OpenAI, Groq, OpenRouter, LM Studio, ...) with your key
  anthropic: Claude, with your key

Each provider exposes `complete(system, user)` and, where the backend supports prompt
caching, `prefill(system, user)` to warm the cache with a partial transcript.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class Completion:
    text: str
    ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    #: True when generation stopped because it hit max_tokens rather than finishing. Such
    #: output is a fragment, never an edit, and the caller must not use it.
    truncated: bool = False


class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.0) -> Completion: ...

    def prefill(self, system: str, user: str) -> None:
        """Warm the prompt cache with a prefix (best effort, may be a no-op)."""


def build_provider(cfg, server_factory=None) -> LLMProvider | None:
    """Create the provider named in PostProcessConfig, or None when clean-up is off.
    `server_factory` returns a started LlamaServer for the bundled provider (the engine owns it)."""
    if not cfg.llm_cleanup:
        return None
    kind = cfg.llm_provider
    if kind == "bundled":
        if server_factory is None:
            raise RuntimeError("bundled provider needs a llama-server factory")
        return BundledProvider(server_factory(), timeout=cfg.llm_timeout_s)
    if kind == "ollama":
        return OllamaProvider(cfg.llm_url or "http://127.0.0.1:11434", cfg.llm_model or "qwen3:4b", cfg.llm_timeout_s)
    if kind == "openai":
        if not cfg.llm_url:
            raise RuntimeError("openai provider needs llm_url (e.g. https://api.openai.com or https://api.groq.com/openai)")
        return OpenAICompatibleProvider(cfg.llm_url, cfg.llm_api_key or None, cfg.llm_model, cfg.llm_timeout_s)
    if kind == "anthropic":
        if not cfg.llm_api_key:
            raise RuntimeError("anthropic provider needs llm_api_key")
        return AnthropicProvider(cfg.llm_api_key, cfg.llm_model or "claude-opus-5", cfg.llm_timeout_s)
    raise ValueError(f"unknown llm_provider {kind!r}")


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class OpenAICompatibleProvider:
    """Works for llama-server, OpenAI, Groq, OpenRouter, LM Studio, vLLM ..."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: float = 20.0,
                 extra_body: dict[str, Any] | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.extra_body = extra_body or {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def complete(self, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.0) -> Completion:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            **self.extra_body,
        }
        t0 = time.perf_counter()
        data = _post_json(f"{self.base_url}/v1/chat/completions", body, self._headers(), self.timeout)
        ms = (time.perf_counter() - t0) * 1000
        choice = (data.get("choices") or [{}])[0]
        text = choice.get("message", {}).get("content", "") or ""
        truncated = choice.get("finish_reason") == "length"
        usage = data.get("usage") or {}
        cached = None
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            cached = details.get("cached_tokens")
        return Completion(text.strip(), ms, usage.get("prompt_tokens"), usage.get("completion_tokens"),
                          cached, truncated)

    def prefill(self, system: str, user: str) -> None:
        """One-token completion: llama-server keeps the evaluated prompt in its slot cache."""
        try:
            self.complete(system, user, max_tokens=1)
        except Exception as e:
            log.debug("prefill failed: %s", e)


class BundledProvider(OpenAICompatibleProvider):
    """llama-server managed by the engine. Same API; extra_body enables prompt caching."""

    name = "bundled"

    def __init__(self, server, model: str = "local", timeout: float = 20.0):
        # cache_prompt: reuse the evaluated prefix between requests (the system prompt, and the
        # transcript so far when pre-filling). enable_thinking=false: Qwen3 must answer directly.
        super().__init__(server.base_url, server.api_key, model, timeout,
                         extra_body={"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}})
        self.server = server


class OllamaProvider:
    name = "ollama"

    def __init__(self, url: str = "http://127.0.0.1:11434", model: str = "qwen3:4b", timeout: float = 20.0):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.0) -> Completion:
        body = {
            "model": self.model, "stream": False, "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        t0 = time.perf_counter()
        data = _post_json(f"{self.url}/api/chat", body, {}, self.timeout)
        text = (data.get("message") or {}).get("content", "") or ""
        return Completion(text.strip(), (time.perf_counter() - t0) * 1000, data.get("prompt_eval_count"),
                          data.get("eval_count"), None, data.get("done_reason") == "length")

    def prefill(self, system: str, user: str) -> None:
        pass


class AnthropicProvider:
    """Claude via the official SDK (bring your own key). The system prompt carries a cache
    breakpoint so repeated dictations only pay for the transcript; effort is kept low because
    this is a copy-edit task where latency matters; server-side refusal fallbacks are enabled
    so a rare policy decline still returns cleaned text."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-opus-5", timeout: float = 20.0):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("the anthropic provider needs the SDK: pip install -e ./engine[cloud]") from e
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=1)

    def complete(self, system: str, user: str, *, max_tokens: int = 512, temperature: float = 0.0) -> Completion:
        t0 = time.perf_counter()
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        ms = (time.perf_counter() - t0) * 1000
        if response.stop_reason == "refusal":
            return Completion("", ms)
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return Completion(text.strip(), ms, getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None),
                          getattr(usage, "cache_read_input_tokens", None), response.stop_reason == "max_tokens")

    def prefill(self, system: str, user: str) -> None:
        pass
