"""Protocol-agnostic models + abstract Endpoint for WhalePod.

All three providers (vLLM / OpenAI-compatible / Anthropic) are reduced to
these unified types so the agent loop and UI never touch protocol details.

Two things live here that used to be duplicated per provider:

  * ``normalize_tools`` — the registry emits OpenAI-shaped dicts, but callers
    may also pass ``ToolDef``. Every provider normalizes through one function
    so a new provider cannot regress on the shape it receives.
  * ``Usage`` — the *measured* token accounting reported by the server,
    including how much of the prompt was served from the prefix cache. This is
    the ground truth WhalePod's whole design is optimizing for, so it is a
    first-class type rather than a raw dict.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import httpx


@dataclass
class Message:
    role: str                       # system | user | assistant | tool
    content: str = ""
    name: Optional[str] = None      # tool result name (vLLM/OpenAI)
    tool_call_id: Optional[str] = None  # for role=tool
    tool_calls: Optional[list] = None   # for role=assistant (tool calls requested)
    # Assistant chain-of-thought. Retained for local rendering/replay ONLY —
    # providers must never echo it back upstream (DeepSeek rejects it, and it
    # would bloat the cached prefix for everyone else).
    reasoning: Optional[str] = None


@dataclass
class ToolDef:
    name: str
    description: str = ""
    parameters: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}})


def normalize_tools(tools) -> list[ToolDef]:
    """Coerce anything tool-shaped into ``list[ToolDef]``.

    Accepts OpenAI wrappers (``{"type":"function","function":{...}}``), bare
    schema dicts, Anthropic-style dicts (``input_schema``), and ``ToolDef``.
    """
    out: list[ToolDef] = []
    for t in tools or []:
        if isinstance(t, ToolDef):
            out.append(t)
            continue
        if isinstance(t, dict):
            fn = t["function"] if isinstance(t.get("function"), dict) else t
            params = (fn.get("parameters") or fn.get("input_schema")
                      or {"type": "object", "properties": {}})
            out.append(ToolDef(name=fn.get("name", ""),
                               description=fn.get("description") or "",
                               parameters=params))
            continue
        out.append(ToolDef(
            name=getattr(t, "name", ""),
            description=getattr(t, "description", "") or "",
            parameters=getattr(t, "parameters", None)
            or {"type": "object", "properties": {}},
        ))
    return out


@dataclass
class ToolCallDelta:
    index: int
    id: Optional[str]
    name: Optional[str]          # may come in a later chunk for some providers
    arguments: str               # accumulated JSON string


# --------------------------------------------------------------- usage ---
@dataclass
class Usage:
    """Server-reported token accounting, normalized across providers.

    ``cached_tokens`` is the part of the prompt that hit the provider's prefix
    cache. It is the only honest measure of whether WhalePod's context layout
    is doing its job, so it is surfaced all the way to the status bar.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0        # prompt tokens served from cache (hit)
    cache_write_tokens: int = 0   # Anthropic: cache_creation_input_tokens
    requests: int = 0
    measured: bool = False        # False => provider told us nothing

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cache_hit_rate(self) -> Optional[float]:
        """Fraction of prompt tokens served from cache, or None if unmeasured."""
        if not self.measured or not self.prompt_tokens:
            return None
        return self.cached_tokens / self.prompt_tokens

    def merge(self, other: "Usage") -> None:
        """Accumulate another usage record into this one (session totals)."""
        if other is None:
            return
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.requests += other.requests or 1
        self.measured = self.measured or other.measured


def _int(d: dict, *keys) -> int:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return 0


def parse_openai_usage(raw: Optional[dict]) -> Optional[Usage]:
    """Normalize an OpenAI/vLLM/DeepSeek ``usage`` object.

    Cache accounting appears in two different shapes in the wild:
      * DeepSeek:      ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``
      * OpenAI/vLLM:   ``prompt_tokens_details.cached_tokens``
    Both are read so the number is right on either backend. OpenRouter also
    reports ``prompt_tokens_details.cache_write_tokens``, which is the same
    quantity Anthropic calls ``cache_creation_input_tokens``.
    """
    if not raw:
        return None
    pd = raw.get("prompt_tokens_details") or {}
    cd = raw.get("completion_tokens_details") or {}
    cached = _int(raw, "prompt_cache_hit_tokens")
    if not cached:
        cached = _int(pd, "cached_tokens")
    return Usage(
        prompt_tokens=_int(raw, "prompt_tokens", "input_tokens"),
        completion_tokens=_int(raw, "completion_tokens", "output_tokens"),
        reasoning_tokens=_int(cd, "reasoning_tokens"),
        cached_tokens=cached,
        cache_write_tokens=_int(pd, "cache_write_tokens"),
        requests=1,
        measured=True,
    )


def parse_anthropic_usage(raw: Optional[dict]) -> Optional[Usage]:
    """Normalize an Anthropic ``usage`` object (explicit cache_control)."""
    if not raw:
        return None
    return Usage(
        prompt_tokens=(_int(raw, "input_tokens")
                       + _int(raw, "cache_read_input_tokens")
                       + _int(raw, "cache_creation_input_tokens")),
        completion_tokens=_int(raw, "output_tokens"),
        cached_tokens=_int(raw, "cache_read_input_tokens"),
        cache_write_tokens=_int(raw, "cache_creation_input_tokens"),
        requests=1,
        measured=True,
    )


@dataclass
class StreamingDelta:
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[list[ToolCallDelta]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None


@dataclass
class ChatRequest:
    model: str
    messages: list[Message]
    stream: bool = True
    tools: Optional[list] = None       # dicts or ToolDef; normalized per provider
    thinking: bool = True
    reasoning_effort: str = "high"     # low|medium|high
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # One-off auxiliary calls (compaction summaries) set this. On providers with
    # *explicit* caching it suppresses the cache_control breakpoint: a prompt
    # that will never be sent again should not be paid for at the cache-write
    # premium. Providers with automatic prefix caching (DeepSeek, vLLM) have no
    # equivalent knob and need none — an auxiliary prompt shares no prefix with
    # the session, so it neither hits nor displaces the session's cache.
    no_cache_write: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class ChatResponse:
    content: str
    reasoning: Optional[str] = None
    tool_calls: Optional[list] = None
    usage: Optional[Usage] = None
    finish_reason: Optional[str] = None


class EndpointError(Exception):
    """Provider-level failure (auth, network, rate limit, cold start).

    ``status_code`` is carried explicitly so retry policy can key on it
    instead of substring-matching the message (a response *body* containing
    "503" used to be enough to trigger a cold-start retry).
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        """True for transient conditions worth backing off on."""
        sc = self.status_code
        if sc is None:
            return False
        return sc == 408 or sc == 409 or sc == 429 or 500 <= sc < 600


class Endpoint(ABC):
    type = "base"

    def __init__(self, base_url: str, api_key: Optional[str],
                 extra_headers: Optional[dict] = None, timeout: float = 120.0,
                 stream_usage: bool = True, extra_body: Optional[dict] = None):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        # Provider-specific request fields merged into every payload. This is
        # not a convenience knob: a prefix cache lives in one server's KV cache,
        # so a router that load-balances across providers turns every request
        # into a cold miss. Measured on OpenRouter — an identical 11.2k-token
        # prefix scored ~0% cached unpinned and 98% pinned to one provider.
        # Config puts the pin here (e.g. {"provider": {"order": ["DeepInfra"],
        # "allow_fallbacks": false}}) rather than in the agent, because holding
        # affinity is a transport concern.
        self.extra_body = dict(extra_body or {})
        self.timeout = timeout
        # Ask the server to report usage on streaming responses. Without this,
        # OpenAI-compatible backends send no usage block at all when streaming,
        # which is exactly the number we care most about.
        self.stream_usage = stream_usage
        self._http: Optional[httpx.AsyncClient] = None
        self._http_loop = None

    # -- to be implemented ----------------------------------------------
    @abstractmethod
    async def stream_chat(self, req: ChatRequest) -> AsyncIterator[StreamingDelta]:
        raise NotImplementedError

    @abstractmethod
    async def chat(self, req: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------
    def _auth_headers(self) -> dict:
        raise NotImplementedError

    def _headers(self) -> dict:
        """Auth headers plus any user-configured extras."""
        h = dict(self._auth_headers())
        h.update(self.extra_headers)
        return h

    def _payload(self, req: ChatRequest, stream: Optional[bool] = None) -> dict:
        raise NotImplementedError

    def _url(self, endpoint: str = "/chat/completions") -> str:
        return f"{self.base_url}{endpoint}"

    def _timeout(self) -> httpx.Timeout:
        """Short connect, long read.

        A high-effort reasoning turn can stream for minutes with sparse tokens;
        a flat timeout either kills those turns or makes a dead host hang.
        """
        return httpx.Timeout(self.timeout, connect=min(15.0, self.timeout),
                             read=max(self.timeout, 300.0))

    def _client(self) -> httpx.AsyncClient:
        """One pooled client per endpoint (keep-alive across the agent loop).

        Recreated if the running event loop changed, so a process that calls
        ``asyncio.run`` more than once doesn't reuse a dead connection pool.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._http is not None and self._http_loop is not loop:
            self._http = None
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=self._timeout(),
                # The local system proxy config is broken on some machines and
                # silently breaks TLS; curl proves direct works. See docs.
                trust_env=False,
            )
            self._http_loop = loop
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            try:
                await self._http.aclose()
            finally:
                self._http = None
                self._http_loop = None

    # -- shared error mapping ------------------------------------------------
    @staticmethod
    def _raise_for_status(status_code: int, body: str) -> None:
        if status_code >= 400:
            raise EndpointError(f"HTTP {status_code}: {body[:500]}",
                                status_code=status_code)
