"""Anthropic Messages endpoint (/v1/messages) with protocol conversion.

Converts the unified ChatRequest into the Anthropic block-based format and
normalizes its SSE event stream back into StreamingDelta.

Anthropic specifics handled here:
  - auth via `x-api-key` + `anthropic-version`
  - `system` is a top-level field (string or blocks)
  - message content is a list of blocks: text / thinking / tool_use / tool_result
  - there is no `tool` role: tool results are *user* messages carrying
    `tool_result` blocks, and consecutive same-role messages must be merged
  - streaming reasoning arrives as `thinking_delta`; tool arguments arrive as
    `input_json_delta` fragments that must be stitched onto the id/name from
    the preceding `content_block_start`
  - caching is explicit rather than automatic, so `cache_control` is stamped on
    the stable prefix (tools + system) to get the same effect WhalePod relies
    on from DeepSeek's automatic prefix cache
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from .base import (
    ChatRequest, ChatResponse, Endpoint, EndpointError,
    Message, StreamingDelta, ToolCallDelta, Usage,
    normalize_tools, parse_anthropic_usage,
)

ANTHROPIC_VERSION = "2023-06-01"
_CACHE_CONTROL = {"type": "ephemeral"}


class AnthropicEndpoint(Endpoint):
    type = "anthropic"

    def __init__(self, *a, cache_prefix: bool = True, **kw):
        super().__init__(*a, **kw)
        # Mark the stable prefix (tools + system) as cacheable. This is the
        # explicit equivalent of DeepSeek's automatic prefix caching.
        self.cache_prefix = cache_prefix

    def _auth_headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _url(self, endpoint: str = "/v1/messages") -> str:
        return super()._url(endpoint)

    # -- encoding ---------------------------------------------------------
    def _blocks_for(self, msg: Message) -> list:
        """Content blocks for one unified message (excluding system)."""
        if msg.role == "tool":
            return [{"type": "tool_result",
                     "tool_use_id": msg.tool_call_id or "",
                     "content": msg.content or ""}]
        blocks: list = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"raw": args}
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id") or f"toolu_{tc.get('index', 0)}",
                "name": fn.get("name", ""),
                "input": args,
            })
        return blocks

    def _convert_messages(self, msgs: list[Message]) -> tuple[list, list]:
        """Split into (system_blocks, anthropic_messages).

        A ``tool`` role becomes a ``user`` message; adjacent same-role messages
        are merged, since Anthropic rejects consecutive turns of one role.
        """
        system_blocks: list = []
        out: list = []
        for m in msgs:
            if m.role == "system":
                if m.content:
                    system_blocks.append({"type": "text", "text": m.content})
                continue
            role = "user" if m.role in ("user", "tool") else "assistant"
            blocks = self._blocks_for(m)
            if not blocks:
                # Anthropic rejects empty content; drop the turn entirely.
                continue
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})
        return system_blocks, out

    def _payload(self, req: ChatRequest, stream: Optional[bool] = None) -> dict:
        stream = req.stream if stream is None else stream
        system_blocks, messages = self._convert_messages(req.messages)

        p: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens or 8192,
            "messages": messages,
            "stream": stream,
        }
        if req.temperature is not None:
            p["temperature"] = req.temperature
        if req.top_p is not None:
            p["top_p"] = req.top_p
        if req.thinking:
            budget = max(1024, min((req.max_tokens or 8192) - 1, 8192))
            p["thinking"] = {"type": "enabled", "budget_tokens": budget}
        cache_prefix = self.cache_prefix and not req.no_cache_write
        if req.tools:
            tools = [
                {"name": t.name, "description": t.description,
                 "input_schema": t.parameters}
                for t in normalize_tools(req.tools)
            ]
            if tools and cache_prefix:
                tools[-1] = dict(tools[-1], cache_control=_CACHE_CONTROL)
            p["tools"] = tools
        if system_blocks:
            if cache_prefix:
                system_blocks = list(system_blocks)
                system_blocks[-1] = dict(system_blocks[-1],
                                         cache_control=_CACHE_CONTROL)
                p["system"] = system_blocks
            else:
                p["system"] = (system_blocks if len(system_blocks) > 1
                               else system_blocks[0]["text"])
        p.update(self.extra_body)
        p.update(req.extra)
        return p

    # -- non-streaming ----------------------------------------------------
    async def chat(self, req: ChatRequest) -> ChatResponse:
        payload = self._payload(req, stream=False)
        client = self._client()
        try:
            r = await client.post(self._url(), json=payload, headers=self._headers())
        except httpx.HTTPError as e:
            raise EndpointError(f"network error: {e}") from e
        self._raise_for_status(r.status_code, r.text)
        return self._parse_response(r.json())

    def _parse_response(self, data: dict) -> ChatResponse:
        text_parts, thinking = [], ""
        tool_calls = []
        for block in data.get("content") or []:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking += block.get("thinking", "")
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })
        return ChatResponse(
            content="".join(text_parts),
            reasoning=thinking or None,
            tool_calls=tool_calls or None,
            usage=parse_anthropic_usage(data.get("usage")),
            finish_reason=data.get("stop_reason"),
        )

    # -- streaming ---------------------------------------------------------
    async def stream_chat(self, req: ChatRequest) -> AsyncIterator[StreamingDelta]:
        payload = self._payload(req, stream=True)
        client = self._client()
        # index -> (tool id, tool name), populated by content_block_start so
        # later input_json_delta fragments can be attributed to the right call.
        state: dict = {}
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
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        evt = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for delta in self._parse_event(evt, state):
                        yield delta
        except httpx.HTTPError as e:
            raise EndpointError(f"network error: {e}") from e

    def _parse_event(self, evt: dict,
                     state: Optional[dict] = None) -> list[StreamingDelta]:
        state = {} if state is None else state
        etype = evt.get("type")
        index = evt.get("index", 0)

        if etype == "message_start":
            usage = parse_anthropic_usage(
                (evt.get("message") or {}).get("usage"))
            return [StreamingDelta(usage=usage)] if usage else []

        if etype == "content_block_start":
            block = evt.get("content_block") or {}
            btype = block.get("type")
            if btype == "tool_use":
                state[index] = (block.get("id"), block.get("name"))
                return [StreamingDelta(tool_calls=[ToolCallDelta(
                    index=index, id=block.get("id"),
                    name=block.get("name"), arguments="")])]
            if btype == "text" and block.get("text"):
                return [StreamingDelta(content=block["text"])]
            if btype == "thinking" and block.get("thinking"):
                return [StreamingDelta(reasoning=block["thinking"])]
            return []

        if etype == "content_block_delta":
            delta = evt.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "thinking_delta" or "thinking" in delta:
                return [StreamingDelta(reasoning=delta.get("thinking", ""))]
            if dtype == "text_delta" or "text" in delta:
                return [StreamingDelta(content=delta.get("text", ""))]
            if dtype == "input_json_delta":
                tid, tname = state.get(index, (None, None))
                return [StreamingDelta(tool_calls=[ToolCallDelta(
                    index=index, id=tid, name=tname,
                    arguments=delta.get("partial_json", ""))])]
            return []

        if etype == "message_delta":
            usage = parse_anthropic_usage(evt.get("usage"))
            return [StreamingDelta(
                finish_reason=(evt.get("delta") or {}).get("stop_reason"),
                usage=usage,
            )]

        if etype == "error":
            err = evt.get("error") or {}
            raise EndpointError(
                f"anthropic stream error: {err.get('type')}: {err.get('message')}")
        return []
