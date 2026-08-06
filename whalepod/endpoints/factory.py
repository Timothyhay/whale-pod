"""Build a concrete Endpoint from config."""
from __future__ import annotations

from typing import Optional

from .base import Endpoint
from .vllm import VLLMEndpoint
from .openai import OpenAIChatEndpoint
from .anthropic import AnthropicEndpoint

_REGISTRY = {
    "deepseek": VLLMEndpoint,
    "custom": VLLMEndpoint,
    "openai": OpenAIChatEndpoint,
    "anthropic": AnthropicEndpoint,
}


def build_endpoint(endpoint_type: str,
                   base_url: str,
                   api_key: Optional[str] = None,
                   extra_headers: Optional[dict] = None,
                   timeout: float = 120.0,
                   stream_usage: bool = True,
                   extra_body: Optional[dict] = None) -> Endpoint:
    cls = _REGISTRY.get(endpoint_type)
    if cls is None:
        raise ValueError(
            f"unknown endpoint type {endpoint_type!r}; "
            f"expected one of {sorted(_REGISTRY)}"
        )
    if not base_url:
        raise ValueError(
            "no endpoint base_url configured — run `whalepod auth` "
            "or set endpoint.base_url in ~/.whalepod/config.json"
        )
    return cls(base_url=base_url, api_key=api_key,
               extra_headers=extra_headers, timeout=timeout,
               stream_usage=stream_usage, extra_body=extra_body)


__all__ = ["build_endpoint", "Endpoint"]
