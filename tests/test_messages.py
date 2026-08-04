"""Tests for the prefix-caching-friendly message manager.

These lock the two-zone model in place. The previous version of this file
asserted a three-zone layout and a ``working_tokens`` field, both of which were
removed: a "volatile" zone appended after history is just the tail of the
history as far as prefix caching is concerned, so it re-sent file contents on
every request while claiming to be free.
"""
import unittest

from whalepod.core.messages import MessageManager
from whalepod.endpoints.base import Message, Usage


def _mm(**kw) -> MessageManager:
    mm = MessageManager(**kw)
    mm.set_system("SYS")
    mm.set_tools([{"type": "function", "function": {"name": "read_file"}}])
    return mm


class TestStablePrefix(unittest.TestCase):
    def test_repo_map_reaches_the_request(self):
        """Regression: the map was rendered and then never sent."""
        mm = _mm()
        mm.set_repo_map("MAP-MARKER ƒ a.py:1 def f()")
        mm.add_user("hello")
        msgs = mm.ordered_messages()
        self.assertEqual(msgs[0].role, "system")
        self.assertIn("MAP-MARKER", msgs[0].content)

    def test_single_system_message_with_prompt_before_map(self):
        mm = _mm()
        mm.set_repo_map("THE-MAP")
        msgs = mm.ordered_messages()
        systems = [m for m in msgs if m.role == "system"]
        self.assertEqual(len(systems), 1)
        self.assertLess(systems[0].content.index("SYS"),
                        systems[0].content.index("THE-MAP"))

    def test_prefix_is_byte_stable_across_turns(self):
        mm = _mm()
        mm.set_repo_map("THE-MAP")
        first = mm.ordered_messages()[0].content
        mm.add_user("q"); mm.add_assistant("a")
        self.assertEqual(mm.ordered_messages()[0].content, first)

    def test_refreshing_the_map_keeps_the_prompt_prefix(self):
        """The map sits after the prompt, so a refresh only invalidates its own
        tokens — the prompt's bytes stay cached."""
        mm = _mm()
        mm.set_repo_map("OLD")
        before = mm.ordered_messages()[0].content
        mm.set_repo_map("NEW")
        after = mm.ordered_messages()[0].content
        common = 0
        for a, b in zip(before, after):
            if a != b:
                break
            common += 1
        self.assertGreaterEqual(common, len("SYS"))

    def test_no_system_message_when_nothing_is_set(self):
        mm = MessageManager()
        mm.add_user("hi")
        self.assertEqual([m.role for m in mm.ordered_messages()], ["user"])


class TestHistory(unittest.TestCase):
    def test_history_is_append_only(self):
        mm = _mm()
        mm.add_user("q1")
        mm.add_assistant("a1")
        mm.add_user("q2")
        self.assertEqual([m.content for m in mm.history], ["q1", "a1", "q2"])

    def test_token_counts_track_appends(self):
        mm = _mm()
        self.assertEqual(mm.estimated_history(), 0)
        mm.add_user("some text here")
        self.assertGreater(mm.estimated_history(), 0)
        self.assertEqual(mm.estimated_history(), sum(mm._hist_tokens))

    def test_clear_history_keeps_prefix(self):
        mm = _mm()
        mm.set_repo_map("THE-MAP")
        mm.add_user("q"); mm.add_assistant("a")
        dropped = mm.clear_history()
        self.assertEqual(dropped, 2)
        self.assertEqual(mm.history, [])
        self.assertEqual(mm.estimated_history(), 0)
        self.assertIn("THE-MAP", mm.ordered_messages()[0].content)


class TestOpenToolCalls(unittest.TestCase):
    """An interrupted turn must not leave a log providers reject with a 400."""

    def _pending(self, mm):
        mm.add_user("do it")
        mm.add_assistant("", tool_calls=[
            {"id": "c1", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}},
            {"id": "c2", "type": "function",
             "function": {"name": "grep", "arguments": "{}"}},
        ])

    def test_detects_unanswered_calls(self):
        mm = _mm()
        self._pending(mm)
        self.assertEqual({c["id"] for c in mm.open_tool_calls()}, {"c1", "c2"})

    def test_close_appends_results_without_editing_history(self):
        mm = _mm()
        self._pending(mm)
        before = list(mm.history)
        n = mm.close_open_tool_calls("(interrupted)")
        self.assertEqual(n, 2)
        self.assertEqual(mm.history[:len(before)], before)   # append-only repair
        self.assertEqual(mm.open_tool_calls(), [])
        self.assertEqual([m.role for m in mm.history[-2:]], ["tool", "tool"])

    def test_answered_calls_are_not_reclosed(self):
        mm = _mm()
        self._pending(mm)
        mm.add_tool_result("c1", "ok", "read_file")
        self.assertEqual([c["id"] for c in mm.open_tool_calls()], ["c2"])
        self.assertEqual(mm.close_open_tool_calls(), 1)
        self.assertEqual(mm.close_open_tool_calls(), 0)


class TestPruning(unittest.TestCase):
    def test_no_pruning_below_the_threshold(self):
        mm = _mm(window=1_000_000)
        mm.add_user("u1"); mm.add_assistant("a1")
        self.assertIsNone(mm.prune_if_needed())
        self.assertEqual(len(mm.history), 2)

    def test_prunes_whole_turns_and_reports_the_cost(self):
        mm = MessageManager(window=400, prune_at=0.5, prune_to=0.25)
        mm.set_system("S")
        for i in range(8):
            mm.add_user(f"user question number {i} " * 4)
            mm.add_assistant(f"assistant answer number {i} " * 4)
        evt = mm.prune_if_needed()
        self.assertIsNotNone(evt)
        self.assertGreater(evt.turns_dropped, 0)
        self.assertGreater(evt.tokens_dropped, 0)
        self.assertIn("cache", evt.describe())
        # A marker is left so the model knows there is a gap.
        self.assertEqual(mm.history[0].role, "user")
        self.assertIn("elided", mm.history[0].content)
        # No message was cut in half.
        for m in mm.history[1:]:
            self.assertTrue(m.content.startswith(("user question",
                                                  "assistant answer")))

    def test_pruning_stops_rather_than_erasing_the_live_turn(self):
        """One turn bigger than the target must not wipe the conversation."""
        mm = MessageManager(window=100, prune_at=0.1, prune_to=0.1)
        mm.set_system("S")
        mm.add_user("x" * 4000)
        mm.prune_if_needed()
        self.assertGreaterEqual(len(mm.history), 1)

    def test_history_total_stays_consistent_after_pruning(self):
        mm = MessageManager(window=400, prune_at=0.5, prune_to=0.25)
        mm.set_system("S")
        for i in range(6):
            mm.add_user(f"q{i} " * 20)
            mm.add_assistant(f"a{i} " * 20)
        mm.prune_if_needed()
        self.assertEqual(mm.estimated_history(), sum(mm._hist_tokens))
        self.assertEqual(len(mm._hist_tokens), len(mm.history))


class TestTelemetry(unittest.TestCase):
    def test_stats_totals(self):
        mm = _mm(window=1_000_000)
        mm.add_user("user text")
        st = mm.stats()
        self.assertGreater(st.stable_tokens, 0)
        self.assertEqual(st.window, 1_000_000)
        self.assertEqual(st.total, st.stable_tokens + st.history_tokens)

    def test_cache_hit_rate_is_none_when_unmeasured(self):
        """Regression: the old status bar computed a hit rate from our own token
        estimate, so it showed a healthy percentage with nothing cached."""
        mm = _mm()
        self.assertIsNone(mm.stats().cache_hit_rate)
        mm.record_usage(None)
        self.assertIsNone(mm.stats().cache_hit_rate)

    def test_cache_hit_rate_comes_from_the_server(self):
        mm = _mm()
        mm.record_usage(Usage(prompt_tokens=1000, completion_tokens=10,
                              cached_tokens=750, requests=1, measured=True))
        self.assertAlmostEqual(mm.stats().cache_hit_rate, 0.75)

    def test_session_usage_accumulates(self):
        mm = _mm()
        for _ in range(3):
            mm.record_usage(Usage(prompt_tokens=100, completion_tokens=5,
                                  cached_tokens=50, requests=1, measured=True))
        st = mm.stats()
        self.assertEqual(st.session_usage.prompt_tokens, 300)
        self.assertEqual(st.session_usage.requests, 3)
        self.assertAlmostEqual(st.session_cache_hit_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
