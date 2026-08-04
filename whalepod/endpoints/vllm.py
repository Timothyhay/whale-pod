"""vLLM / OpenAI-compatible endpoint (POST /v1/chat/completions, SSE stream).

Verified against a HuggingFace Inference Endpoint (vLLM backend):
  - streaming returns `data:` lines ending with `data: [DONE]`
  - tool calling returns standard `tool_calls`
  - with reasoning_effort, streaming deltas carry a separate reasoning field

Two provider quirks are handled deliberately:

  * **Reasoning field name.** The tested vLLM endpoint emits ``reasoning``;
    DeepSeek's own API and vLLM's default reasoning parser emit
    ``reasoning_content``. Both are accepted, so Thinking mode does not
    silently go blank when you point WhalePod at api.deepseek.com.
  * **Reasoning is never echoed back.** DeepSeek requires that the chain of
    thought is *not* replayed in subsequent requests. We keep it locally for
    the UI and strip it from the wire.
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from .base import (
    ChatRequest, ChatResponse, Endpoint, EndpointError,
    Message, StreamingDelta, ToolCallDelta, Usage,
    normalize_tools, parse_openai_usage,
)


def _pick_reasoning(d: dict) -> Optional[str]:
    """Read the reasoning delta under either of its two field names."""
    for key in ("reasoning_content", "reasoning"):
        v = d.get(key)
        if v:
            return v
    return None


class VLLMEndpoint(Endpoint):
    type = "vllm"

    def _auth_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        # HF / vLLM endpoints accept any non-empty bearer; some need none.
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    # -- message / tool encoding -----------------------------------------
    def _encode_message(self, m: Message) -> dict:
        d: dict = {"role": m.role, "content": m.content}
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name and m.role == "tool":
            d["name"] = m.name
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
            # An assistant turn that only requests tools carries no text; some
            # servers reject content="" alongside tool_calls.
            if not m.content:
                d["content"] = None
        # m.reasoning is intentionally NOT sent — see module docstring.
        return d

    def _encode_tool(self, t) -> dict:
        return {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }

    def _payload(self, req: ChatRequest, stream: Optional[bool] = None) -> dict:
        stream = req.stream if stream is None else stream
        p: dict = {
            "model": req.model,
            "messages": [self._encode_message(m) for m in req.messages],
            "stream": stream,
        }
        if req.tools:
            p["tools"] = [self._encode_tool(t) for t in normalize_tools(req.tools)]
        if req.max_tokens:
            p["max_tokens"] = req.max_tokens
        if req.temperature is not None:
            p["temperature"] = req.temperature
        if req.top_p is not None:
            p["top_p"] = req.top_p
        if req.thinking:
            p["reasoning_effort"] = req.reasoning_effort
        if stream and self.stream_usage:
            # Without this, OpenAI-compatible servers omit usage entirely on
            # streamed responses — and usage is where cache-hit counts live.
            p["stream_options"] = {"include_usage": True}
        # Configured provider fields first, per-request overrides last.
        p.update(self.extra_body)
        p.update(req.extra)
        return p

    # -- HTTP --------------------------------------------------------------
    def _url(self, endpoint: str = "/v1/chat/completions") -> str:
        return super()._url(endpoint)

    async def chat(self, req: ChatRequest) -> ChatResponse:
        payload = self._payload(req, stream=False)
        client = self._client()
        try:
            r = await client.post(self._url(), json=payload, headers=self._headers())
        except httpx.HTTPError as e:
            raise EndpointError(f"network error: {e}") from e
        self._raise_for_status(r.status_code, r.text)
        return self._parse_chat_response(r.json())

    def _parse_chat_response(self, data: dict) -> ChatResponse:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        return ChatResponse(
            content=msg.get("content") or "",
            reasoning=_pick_reasoning(msg),
            tool_calls=msg.get("tool_calls"),
            usage=parse_openai_usage(data.get("usage")),
            finish_reason=choice.get("finish_reason"),
        )

    async def stream_chat(self, req: ChatRequest) -> AsyncIterator[StreamingDelta]:
        payload = self._payload(req, stream=True)
        client = self._client()
        try:
            async with client.stream("POST", self._url(), json=payload,
                                     headers=self._headers()) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    self._raise_for_status(resp.status_code, body)
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        return
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    delta = self._parse_stream_chunk(chunk)
                    if delta is not None:
                        yield delta
        except httpx.HTTPError as e:
            raise EndpointError(f"network error: {e}") from e

    def _parse_stream_chunk(self, chunk: dict) -> Optional[StreamingDelta]:
        usage = parse_openai_usage(chunk.get("usage"))
        choices = chunk.get("choices")
        if not choices:
            # The final chunk of a stream_options request carries usage only.
            return StreamingDelta(usage=usage) if usage else None
        choice = choices[0]
        d = choice.get("delta") or {}
        tool_calls = None
        if d.get("tool_calls"):
            tool_calls = []
            for tc in d["tool_calls"]:
                fn = tc.get("function") or {}
                tool_calls.append(ToolCallDelta(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    name=fn.get("name"),
                    arguments=fn.get("arguments") or "",
                ))
        return StreamingDelta(
            content=d.get("content"),
            reasoning=_pick_reasoning(d),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )
