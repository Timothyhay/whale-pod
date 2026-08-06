"""Endpoint encoding, protocol conversion and usage parsing (all offline).

Several of these are regressions for bugs that only showed up against a live
server, so they are pinned here rather than left to manual testing:
  * tools arriving as plain dicts crashed every provider (``normalize_tools``)
  * ``role="tool"`` was sent to Anthropic, which has no such role (400)
  * streamed ``input_json_delta`` fragments were dropped, so tool arguments
    arrived as ``{}``
  * chain-of-thought was echoed back upstream
  * streaming responses reported no usage at all without ``include_usage``
"""
import unittest

from whalepod.endpoints.base import (
    ChatRequest, Message, StreamingDelta, ToolCallDelta, ToolDef,
    normalize_tools, parse_anthropic_usage, parse_openai_usage,
)
from whalepod.endpoints.vllm import VLLMEndpoint
from whalepod.endpoints.openai import OpenAIChatEndpoint
from whalepod.endpoints.anthropic import AnthropicEndpoint

OPENAI_TOOL = {"type": "function", "function": {
    "name": "read_file", "description": "r",
    "parameters": {"type": "object", "properties": {}}}}


class TestToolNormalization(unittest.TestCase):
    """The registry emits dicts; the endpoints used to require ToolDef."""

    def test_accepts_openai_wrapper_dict(self):
        [t] = normalize_tools([OPENAI_TOOL])
        self.assertEqual(t.name, "read_file")
        self.assertEqual(t.description, "r")
        self.assertEqual(t.parameters["type"], "object")

    def test_accepts_tooldef_and_bare_schema_and_anthropic_shape(self):
        tools = normalize_tools([
            ToolDef(name="a", parameters={"type": "object"}),
            {"name": "b", "parameters": {"type": "object"}},
            {"name": "c", "input_schema": {"type": "object"}},
        ])
        self.assertEqual([t.name for t in tools], ["a", "b", "c"])
        for t in tools:
            self.assertEqual(t.parameters["type"], "object")

    def test_every_provider_accepts_registry_dicts(self):
        from whalepod.tools.registry import ToolRegistry
        from pathlib import Path
        schemas = ToolRegistry(Path.cwd(), sandbox_mode="readonly").schemas()
        req = ChatRequest(model="m",
                          messages=[Message(role="user", content="hi")],
                          tools=schemas)
        for ep in (VLLMEndpoint("https://x", "k"),
                   OpenAIChatEndpoint("https://x", "k"),
                   AnthropicEndpoint("https://x", "k")):
            payload = ep._payload(req)      # must not raise
            self.assertTrue(payload["tools"])


class TestOpenAIStyleEncoding(unittest.TestCase):
    def test_payload_messages_and_tools(self):
        ep = VLLMEndpoint("https://x", "k")
        req = ChatRequest(model="m",
                          messages=[Message(role="user", content="hi")],
                          tools=[OPENAI_TOOL], thinking=True)
        p = ep._payload(req)
        self.assertEqual(p["model"], "m")
        self.assertEqual(p["messages"][0]["content"], "hi")
        self.assertIn("reasoning_effort", p)
        self.assertEqual(p["tools"][0]["function"]["name"], "read_file")

    def test_instant_mode_no_reasoning(self):
        ep = VLLMEndpoint("https://x", "k")
        req = ChatRequest(model="m", messages=[], thinking=False)
        self.assertNotIn("reasoning_effort", ep._payload(req))

    def test_streaming_requests_usage(self):
        """Without include_usage an OpenAI-compatible server streams no usage."""
        ep = VLLMEndpoint("https://x", "k")
        req = ChatRequest(model="m", messages=[Message(role="user", content="h")])
        p = ep._payload(req, stream=True)
        self.assertTrue(p["stream"])
        self.assertEqual(p["stream_options"], {"include_usage": True})

    def test_non_streaming_has_no_stream_options(self):
        ep = VLLMEndpoint("https://x", "k")
        req = ChatRequest(model="m", messages=[])
        self.assertNotIn("stream_options", ep._payload(req, stream=False))

    def test_reasoning_is_never_sent_upstream(self):
        """Replaying chain-of-thought is rejected by DeepSeek and wastes prefix."""
        ep = VLLMEndpoint("https://x", "k")
        req = ChatRequest(model="m", messages=[
            Message(role="assistant", content="ans",
                    reasoning="secret internal thoughts")])
        blob = repr(ep._payload(req))
        self.assertNotIn("secret internal thoughts", blob)
        self.assertNotIn("reasoning_content", blob)

    def test_tool_result_uses_tool_role_with_id(self):
        ep = VLLMEndpoint("https://x", "k")
        req = ChatRequest(model="m", messages=[
            Message(role="tool", content="out", tool_call_id="tc1",
                    name="read_file")])
        m = ep._payload(req)["messages"][0]
        self.assertEqual(m["role"], "tool")
        self.assertEqual(m["tool_call_id"], "tc1")


class TestAnthropicConversion(unittest.TestCase):
    def _req(self, **kw):
        base = dict(model="claude",
                    messages=[Message(role="system", content="SYS"),
                              Message(role="user", content="hi")],
                    tools=[ToolDef(name="read_file", description="r",
                                   parameters={"type": "object"})],
                    thinking=True, max_tokens=500)
        base.update(kw)
        return ChatRequest(**base)

    def test_system_is_top_level_cacheable_blocks(self):
        """The stable prefix is marked cacheable — Anthropic's caching is
        explicit, unlike DeepSeek's automatic prefix cache."""
        p = AnthropicEndpoint("https://x", "k")._payload(self._req())
        self.assertEqual(p["system"],
                         [{"type": "text", "text": "SYS",
                           "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(p["tools"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(p["messages"][0]["role"], "user")
        self.assertEqual(p["messages"][0]["content"],
                         [{"type": "text", "text": "hi"}])
        self.assertEqual(p["tools"][0]["name"], "read_file")
        self.assertEqual(p["thinking"]["type"], "enabled")

    def test_system_is_a_plain_string_without_prefix_caching(self):
        ep = AnthropicEndpoint("https://x", "k", cache_prefix=False)
        self.assertEqual(ep._payload(self._req())["system"], "SYS")

    def test_tool_result_becomes_a_user_tool_result_block(self):
        """There is no `tool` role in the Messages API; sending one is a 400."""
        ep = AnthropicEndpoint("https://x", "k")
        p = ep._payload(self._req(messages=[
            Message(role="user", content="do it"),
            Message(role="assistant", tool_calls=[
                {"id": "tu1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"path":"a"}'}}]),
            Message(role="tool", content="out", tool_call_id="tu1"),
        ]))
        roles = [m["role"] for m in p["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(p["messages"][2]["content"],
                         [{"type": "tool_result", "tool_use_id": "tu1",
                           "content": "out"}])
        # tool_use input is decoded from the JSON string, not passed through
        self.assertEqual(p["messages"][1]["content"][0]["input"], {"path": "a"})

    def test_consecutive_same_role_messages_are_merged(self):
        ep = AnthropicEndpoint("https://x", "k")
        p = ep._payload(self._req(messages=[
            Message(role="tool", content="r1", tool_call_id="a"),
            Message(role="tool", content="r2", tool_call_id="b"),
        ]))
        self.assertEqual([m["role"] for m in p["messages"]], ["user"])
        self.assertEqual(len(p["messages"][0]["content"]), 2)

    def test_empty_turns_are_dropped(self):
        ep = AnthropicEndpoint("https://x", "k")
        p = ep._payload(self._req(messages=[Message(role="assistant", content=""),
                                            Message(role="user", content="hi")]))
        self.assertEqual([m["role"] for m in p["messages"]], ["user"])

    def test_parse_stream_events(self):
        ep = AnthropicEndpoint("https://x", "k")
        d = ep._parse_event({"type": "content_block_delta",
                             "delta": {"type": "thinking_delta",
                                       "thinking": "so"}})
        self.assertEqual(d[0].reasoning, "so")
        d2 = ep._parse_event({"type": "content_block_delta",
                              "delta": {"type": "text_delta", "text": "hi"}})
        self.assertEqual(d2[0].content, "hi")

    def test_streamed_tool_arguments_are_attributed_and_accumulated(self):
        """input_json_delta fragments carry no id/name — they must be stitched
        onto the preceding content_block_start, or arguments arrive empty."""
        ep = AnthropicEndpoint("https://x", "k")
        state: dict = {}
        ep._parse_event({"type": "content_block_start", "index": 0,
                         "content_block": {"type": "tool_use", "id": "tu1",
                                           "name": "read_file"}}, state)
        pieces = []
        for frag in ('{"path"', ':"a.py"}'):
            [d] = ep._parse_event({"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "input_json_delta",
                                             "partial_json": frag}}, state)
            tc = d.tool_calls[0]
            self.assertEqual((tc.id, tc.name), ("tu1", "read_file"))
            pieces.append(tc.arguments)
        self.assertEqual("".join(pieces), '{"path":"a.py"}')

    def test_parse_tool_use(self):
        ep = AnthropicEndpoint("https://x", "k")
        resp = ep._parse_response({
            "content": [{"type": "tool_use", "id": "tu1", "name": "read_file",
                         "input": {"path": "a.py"}}],
            "stop_reason": "tool_use"})
        self.assertEqual(resp.tool_calls[0]["function"]["name"], "read_file")
        self.assertIn("a.py", resp.tool_calls[0]["function"]["arguments"])


class TestUsageParsing(unittest.TestCase):
    def test_deepseek_cache_fields(self):
        u = parse_openai_usage({"prompt_tokens": 1000, "completion_tokens": 20,
                                "prompt_cache_hit_tokens": 800,
                                "prompt_cache_miss_tokens": 200})
        self.assertEqual(u.cached_tokens, 800)
        self.assertAlmostEqual(u.cache_hit_rate, 0.8)
        self.assertTrue(u.measured)

    def test_openai_vllm_cache_fields(self):
        u = parse_openai_usage({
            "prompt_tokens": 500, "completion_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 250},
            "completion_tokens_details": {"reasoning_tokens": 7}})
        self.assertEqual(u.cached_tokens, 250)
        self.assertEqual(u.reasoning_tokens, 7)
        self.assertAlmostEqual(u.cache_hit_rate, 0.5)

    def test_anthropic_cache_fields_count_toward_the_prompt(self):
        u = parse_anthropic_usage({"input_tokens": 100, "output_tokens": 20,
                                   "cache_read_input_tokens": 700,
                                   "cache_creation_input_tokens": 200})
        self.assertEqual(u.prompt_tokens, 1000)
        self.assertEqual(u.cached_tokens, 700)
        self.assertEqual(u.cache_write_tokens, 200)
        self.assertAlmostEqual(u.cache_hit_rate, 0.7)

    def test_openrouter_cache_write_field(self):
        """OpenRouter reports the Anthropic `cache_creation` quantity here."""
        u = parse_openai_usage({
            "prompt_tokens": 11220, "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 11008,
                                      "cache_write_tokens": 212}})
        self.assertEqual(u.cached_tokens, 11008)
        self.assertEqual(u.cache_write_tokens, 212)
        self.assertAlmostEqual(u.cache_hit_rate, 11008 / 11220)

    def test_missing_usage_is_none_not_zero(self):
        self.assertIsNone(parse_openai_usage(None))
        self.assertIsNone(parse_anthropic_usage({}))

    def test_unmeasured_usage_has_no_hit_rate(self):
        from whalepod.endpoints.base import Usage
        self.assertIsNone(Usage(prompt_tokens=10).cache_hit_rate)


class TestProviderAffinity(unittest.TestCase):
    """`extra_body` must reach the wire on every provider.

    Measured on OpenRouter: the same 11.2k-token prefix scored ~0% cached when
    the router was free to pick a provider per request, and 98% when pinned to
    one. A prefix cache is a single server's KV cache, so losing this field
    silently turns every request into a cold miss — hence a regression test
    rather than a config knob nobody checks.
    """

    PIN = {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}

    def _endpoints(self, **kw):
        return (VLLMEndpoint("https://x", "k", **kw),
                OpenAIChatEndpoint("https://x", "k", **kw),
                AnthropicEndpoint("https://x", "k", **kw))

    def test_extra_body_reaches_payload_on_every_provider(self):
        req = ChatRequest(model="m", messages=[Message(role="user", content="h")])
        for ep in self._endpoints(extra_body=self.PIN):
            p = ep._payload(req)
            self.assertEqual(p["provider"], self.PIN["provider"],
                             f"{type(ep).__name__} dropped extra_body")

    def test_per_request_extra_overrides_configured_extra_body(self):
        req = ChatRequest(model="m", messages=[Message(role="user", content="h")],
                          extra={"provider": {"order": ["Fireworks"]}})
        for ep in self._endpoints(extra_body=self.PIN):
            self.assertEqual(ep._payload(req)["provider"],
                             {"order": ["Fireworks"]})

    def test_absent_extra_body_adds_nothing(self):
        req = ChatRequest(model="m", messages=[Message(role="user", content="h")])
        for ep in self._endpoints():
            self.assertNotIn("provider", ep._payload(req))

    def test_extra_body_is_copied_not_aliased(self):
        """A shared dict from config must not be mutated by request encoding."""
        pin = {"provider": {"order": ["DeepInfra"]}}
        ep = VLLMEndpoint("https://x", "k", extra_body=pin)
        ep._payload(ChatRequest(model="m", messages=[]))
        self.assertEqual(pin, {"provider": {"order": ["DeepInfra"]}})
        self.assertIsNot(ep.extra_body, pin)

    def test_factory_forwards_extra_body(self):
        from whalepod.endpoints.factory import build_endpoint
        for typ in ("custom", "openai", "anthropic"):
            ep = build_endpoint(typ, "https://x", api_key="k",
                                extra_body=self.PIN)
            self.assertEqual(ep.extra_body, self.PIN)

    def test_config_round_trips_extra_body(self):
        from whalepod.config import Config, _from_dict
        cfg = _from_dict(Config(), {"endpoint": {"extra_body": self.PIN}})
        self.assertEqual(cfg.endpoint.extra_body, self.PIN)


class TestRetryPolicy(unittest.TestCase):
    def test_retryable_keys_on_status_code_not_message(self):
        """A body *containing* "503" used to be enough to trigger a retry."""
        from whalepod.endpoints.base import EndpointError
        self.assertTrue(EndpointError("boom", status_code=503).retryable)
        self.assertTrue(EndpointError("boom", status_code=429).retryable)
        self.assertFalse(EndpointError("boom", status_code=400).retryable)
        self.assertFalse(EndpointError("HTTP 400: error code 503").retryable)


if __name__ == "__main__":
    unittest.main()
