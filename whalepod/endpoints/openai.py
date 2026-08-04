"""OpenAI-compatible chat-completions endpoint.

Protocol is the same shape as vLLM (both are /v1/chat/completions with SSE),
so this subclasses VLLMEndpoint and only overrides auth + default model.
Differences: OpenAI uses Bearer from OPENAI_API_KEY; no reasoning_effort by
default (that is a vLLM/DeepSeek extension), so it is only sent when extra.
"""
from __future__ import annotations

from .vllm import VLLMEndpoint


class OpenAIChatEndpoint(VLLMEndpoint):
    type = "openai"

    def _auth_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h
