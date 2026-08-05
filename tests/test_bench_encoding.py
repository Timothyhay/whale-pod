"""The tokenizer's priority order and the bench's DeepSeek V4 wire encoding.

Regressions pinned here:
  * ``estimate_tokens`` used to assume DeepSeek was ``cl100k_base`` (GPT-4's
    vocabulary), silently mis-measuring every context stat and the offline
    cache bench. The resolver must now prefer the official V4 BPE when it is
    available locally, and degrade cleanly when it is not.
  * The offline bench's prefix measurement used to run over a hand-rolled
    ``json.dumps`` paste of the payload. Prefix caching keys on the byte stream
    the *server* fast-tokenizes, so the measurement now runs over the official
    ``encode_messages`` output (BOS + turn delimiters + tools on the system
    message). These tests pin that the encoder is applied and is prefix-stable.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

from whalepod.core import tokenizer as tk
from whalepod.endpoints.vllm import VLLMEndpoint


def _reset_tokenizer():
    """Point the resolver at a nonexistent file and forget what it learned."""
    os.environ[tk.DSV4_PATH_ENV] = str(Path(tempfile.gettempdir()) / "nope.json")
    tk._dsv4_counter = False
    tk._TRY_TIKTOKEN = True
    tk._enc_cache = None
    tk._HAS_TIKTOKEN = False


class TestTokenizerPriority(unittest.TestCase):
    def setUp(self):
        _reset_tokenizer()

    def test_falls_back_without_a_local_tokenizer(self):
        """No network, no tokenizer file => heuristic still answers."""
        self.assertIsNone(tk.estimate_tokens_dsv4("hello"))
        self.assertGreater(tk.estimate_tokens("hello world"), 0)
        self.assertIn(tk.active_tokenizer_name(),
                      ("tiktoken cl100k_base", "heuristic"))

    def test_cjk_counts_roughly_one_per_char(self):
        text = "你好世界" * 100
        self.assertGreater(tk.estimate_tokens(text), 0)
        self.assertLess(tk.estimate_tokens(text), len(text) * 4)

    def test_empty_and_none_are_zero(self):
        self.assertEqual(tk.estimate_tokens(""), 0)
        self.assertEqual(tk.estimate_tokens(None), 0)

    def test_real_downloaded_tokenizer_produces_bpe_counts(self):
        """The real V4 tokenizer must be hittable when it exists on disk."""
        if not tk.dsv4_tokenizer_path().is_file():
            self.skipTest("tokenizer not downloaded — run bench/fetch_tokenizer.py first")
        tk._dsv4_counter = False
        self.assertIsNotNone(tk.estimate_tokens_dsv4("hello world"))
        self.assertIn("deepseek-v4", tk.active_tokenizer_name())


def _payload(messages, tools=None):
    ep = VLLMEndpoint("https://offline.invalid", "k")
    from whalepod.endpoints.base import Message, ChatRequest
    return ep._payload(ChatRequest(model="m", messages=messages, tools=tools,
                                   stream=False))


class TestDSV4WireEncoding(unittest.TestCase):
    """The offline bench measures the model input bytes, not the JSON payload."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
        from validate import wire_text, common_prefix_len  # noqa: E402
        cls.wire_text_fn = staticmethod(wire_text)
        cls._prefix_len = staticmethod(common_prefix_len)

    def test_wire_text_is_the_v4_encoded_prompt_not_json(self):
        from whalepod.endpoints.base import Message
        tool = {"type": "function",
                "function": {"name": "read_file", "description": "r",
                             "parameters": {"type": "object"}}}
        p = _payload([Message(role="system", content="SYS"),
                      Message(role="user", content="hi")], tools=[tool])
        out = self.wire_text_fn(p)
        self.assertIn("<｜begin▁of▁sentence｜>", out)   # BOS
        self.assertIn("<｜User｜>", out)
        self.assertIn("<｜Assistant｜>", out)
        self.assertIn("read_file", out)          # tool schema on the system msg
        self.assertNotIn('"role": "system"', out)  # not a JSON dump

    def test_same_payload_produces_identical_encoding(self):
        """The encoder is deterministic: same input -> same bytes."""
        from whalepod.endpoints.base import Message
        a = self.wire_text_fn(_payload([Message(role="user", content="hi")]))
        b = self.wire_text_fn(_payload([Message(role="user", content="hi")]))
        self.assertEqual(a, b)
        self.assertGreater(len(a), 0)

    def test_different_first_message_breaks_the_prefix(self):
        from whalepod.endpoints.base import Message
        a = self.wire_text_fn(_payload([Message(role="user", content="aaa")]))
        b = self.wire_text_fn(_payload([Message(role="user", content="bbb")]))
        # BOS + effort prefix are shared, the user content is not.
        self.assertLess(self._prefix_len(a, b), len(a))

    def test_tool_results_are_merged_into_a_user_message(self):
        """V4 has no standalone `tool` role; the encoder merges results."""
        from whalepod.endpoints.base import Message
        msgs = [
            Message(role="user", content="do it"),
            Message(role="assistant", tool_calls=[
                {"id": "tc1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"p":"a"}'}}]),
            Message(role="tool", content="out", tool_call_id="tc1", name="read_file"),
        ]
        out = self.wire_text_fn(_payload(msgs))
        self.assertIn("tool_result", out)
        self.assertIn("out", out)
        self.assertIn("read_file", out)


if __name__ == "__main__":
    unittest.main()
