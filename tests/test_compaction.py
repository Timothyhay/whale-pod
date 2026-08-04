"""Context reduction: where the cut lands, and what stands in the gap.

Two things are being locked down here.

*Where to cut* used to be decided inside ``prune_if_needed``, which bailed out
whenever a single turn was bigger than the target — precisely the case where
reduction was most needed, so the request went out over budget and the provider
rejected it. The cut point is now :meth:`MessageManager.plan_reduction`, testable
on its own, and it may land inside a turn as long as the remaining history does
not start with a tool result.

*What stands in the gap* used to be a marker saying "some conversation was
elided". It is now a summary, when one can be produced. The fallback matters as
much as the feature: a summarizer that times out must degrade to the old
behaviour rather than fail the user's turn.
"""
import asyncio
import unittest

from whalepod.core.compaction import (
    Compactor, MAX_TOOL_RESULT_CHARS, serialize, summary_message,
)
from whalepod.core.messages import MessageManager
from whalepod.endpoints.base import ChatResponse, Message


def _mm(**kw) -> MessageManager:
    mm = MessageManager(**kw)
    mm.set_system("SYS")
    return mm


def _fill(mm, turns: int, size: int = 200) -> None:
    """``turns`` complete user/assistant/tool cycles of roughly equal weight."""
    for i in range(turns):
        mm.add_user(f"question {i} " + "q" * size)
        mm.add_assistant("", tool_calls=[
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}])
        mm.add_tool_result(f"c{i}", f"result {i} " + "r" * size, "read_file")
        mm.add_assistant(f"answer {i} " + "a" * size)


# --------------------------------------------------------------- limits
class TestReductionLimit(unittest.TestCase):
    def test_fraction_governs_a_large_window(self):
        mm = _mm(window=1_000_000, prune_at=0.9, reserve_tokens=16_384)
        self.assertEqual(mm.reduction_limit(), 900_000)

    def test_reserve_governs_a_small_window(self):
        """0.9 of 32k leaves 3.2k — less than one reasoning reply.

        The reserve is an absolute floor precisely because a fraction does not
        know how big a reply is.
        """
        mm = _mm(window=32_000, prune_at=0.9, reserve_tokens=16_384)
        self.assertEqual(mm.reduction_limit(), 32_000 - 16_384)

    def test_reserve_can_be_switched_off(self):
        mm = _mm(window=32_000, prune_at=0.9, reserve_tokens=0)
        self.assertEqual(mm.reduction_limit(), 28_800)


# ------------------------------------------------------------- cut point
class TestCutPoint(unittest.TestCase):
    def test_nothing_to_do_below_the_limit(self):
        mm = _mm(window=1_000_000)
        _fill(mm, 3)
        self.assertFalse(mm.needs_reduction())
        self.assertEqual(mm.plan_reduction(), (0, 0, 0, [], False))

    def test_cuts_whole_turns_and_reports_what_goes(self):
        mm = _mm(window=1_200, prune_at=0.5, prune_to=0.25, reserve_tokens=0)
        _fill(mm, 6)
        idx, turns, tokens, ids, mid = mm.plan_reduction()
        self.assertGreater(idx, 0)
        self.assertFalse(mid)
        self.assertEqual(idx % 4, 0)                 # landed on a turn boundary
        self.assertEqual(turns, idx // 4)
        self.assertEqual(ids, [f"c{i}" for i in range(turns)])
        self.assertGreater(tokens, 0)

    def test_planning_does_not_mutate(self):
        mm = _mm(window=1_200, prune_at=0.5, prune_to=0.25, reserve_tokens=0)
        _fill(mm, 6)
        before = len(mm.history)
        mm.plan_reduction()
        mm.plan_reduction()
        self.assertEqual(len(mm.history), before)

    def test_one_oversized_turn_is_cut_into_rather_than_left_over_budget(self):
        """The old code returned None here and the request failed downstream."""
        mm = _mm(window=600, prune_at=0.5, prune_to=0.25, reserve_tokens=0)
        _fill(mm, 1, size=1_200)
        idx, turns, tokens, ids, mid = mm.plan_reduction()
        self.assertTrue(mid)
        self.assertGreater(idx, 0)
        self.assertLess(idx, len(mm.history))

    def test_a_mid_turn_cut_never_leaves_a_dangling_tool_result(self):
        """A tool result whose assistant call was deleted is a 400 everywhere."""
        for size in (400, 800, 1_600, 3_200):
            mm = _mm(window=600, prune_at=0.5, prune_to=0.25, reserve_tokens=0)
            _fill(mm, 1, size=size)
            idx, *_ = mm.plan_reduction()
            with self.subTest(size=size):
                if idx:
                    self.assertNotEqual(mm.history[idx].role, "tool")

    def test_the_live_turn_is_never_erased(self):
        mm = _mm(window=100, prune_at=0.1, prune_to=0.1, reserve_tokens=0)
        _fill(mm, 1, size=2_000)
        mm.prune_if_needed()
        self.assertTrue(mm.history)


# ------------------------------------------------------------ compaction
class TestCompact(unittest.TestCase):
    def _reduce(self):
        mm = _mm(window=1_200, prune_at=0.5, prune_to=0.25, reserve_tokens=0)
        _fill(mm, 6)
        return mm, mm.plan_reduction()

    def test_summary_replaces_the_cut_slice(self):
        mm, (idx, turns, tokens, ids, mid) = self._reduce()
        kept = list(mm.history[idx:])
        evt = mm.compact(summary_message("## Goal\nship it", turns, tokens),
                         idx, turns, tokens, ids, mid)
        self.assertEqual(mm.history[1:], kept)
        self.assertIn("ship it", mm.history[0].content)
        self.assertEqual(evt.turns_dropped, turns)
        self.assertEqual(evt.dropped_tool_call_ids, ids)
        self.assertGreater(evt.summary_tokens, 0)

    def test_the_summary_is_cheaper_than_what_it_replaced(self):
        mm, (idx, turns, tokens, ids, mid) = self._reduce()
        evt = mm.compact(summary_message("## Goal\nship it", turns, tokens),
                         idx, turns, tokens, ids, mid)
        self.assertLess(evt.summary_tokens, evt.tokens_dropped)
        self.assertFalse(mm.needs_reduction())

    def test_token_accounting_stays_consistent(self):
        mm, (idx, turns, tokens, ids, mid) = self._reduce()
        mm.compact(summary_message("s", turns, tokens), idx, turns, tokens, ids,
                   mid)
        self.assertEqual(mm.estimated_history(),
                         sum(mm._message_tokens(m) for m in mm.history))

    def test_counted_separately_from_blind_prunes(self):
        mm, (idx, turns, tokens, ids, mid) = self._reduce()
        mm.compact(summary_message("s", turns, tokens), idx, turns, tokens, ids,
                   mid)
        st = mm.stats()
        self.assertEqual((st.compactions, st.prunes), (1, 0))

    def test_the_summary_is_handed_to_the_model_not_attributed_to_it(self):
        """An assistant-role summary reads as the model's own prior conclusion,
        which it will then defend instead of re-checking."""
        m = summary_message("## Goal\nx", 3, 9_000)
        self.assertEqual(m.role, "user")
        self.assertIn("re-read", m.content)


# ------------------------------------------------------------ serialize
class TestSerialize(unittest.TestCase):
    def test_roles_and_tool_names_survive(self):
        text = serialize([
            Message(role="system", content="SYS"),
            Message(role="user", content="fix the parser"),
            Message(role="assistant", content="", tool_calls=[
                {"id": "c1", "type": "function",
                 "function": {"name": "grep", "arguments": '{"pattern": "x"}'}}]),
            Message(role="tool", content="a.py:1:x", tool_call_id="c1",
                    name="grep"),
        ])
        self.assertNotIn("SYS", text)          # prefix is not part of the cut
        self.assertIn("fix the parser", text)
        self.assertIn("grep", text)
        self.assertIn("a.py:1:x", text)

    def test_file_contents_are_cut_hard(self):
        """Otherwise summarizing costs about as much as the context it shrinks."""
        big = "line\n" * 20_000
        text = serialize([Message(role="tool", content=big, name="read_file")])
        self.assertLess(len(text), MAX_TOOL_RESULT_CHARS + 500)
        self.assertIn("omitted from the summary input", text)

    def test_reasoning_is_not_summarized(self):
        text = serialize([Message(role="assistant", content="done",
                                  reasoning="SECRET-SCRATCH")])
        self.assertNotIn("SECRET-SCRATCH", text)

    def test_an_over_long_transcript_keeps_its_tail(self):
        msgs = [Message(role="user", content=f"turn {i} " + "x" * 500)
                for i in range(200)]
        text = serialize(msgs, max_chars=5_000)
        self.assertLessEqual(len(text), 5_000 + 100)
        self.assertIn("turn 199", text)
        self.assertIn("earlier transcript omitted", text)


# ------------------------------------------------------------- compactor
class _Endpoint:
    """Records the summarize request; can be told to fail."""

    def __init__(self, reply="## Goal\nship it", error=None):
        self.reply = reply
        self.error = error
        self.requests = []

    async def chat(self, req):
        self.requests.append(req)
        if self.error is not None:
            raise self.error
        return ChatResponse(content=self.reply)


class TestCompactor(unittest.TestCase):
    def summarize(self, endpoint, msgs=None):
        msgs = msgs if msgs is not None else [
            Message(role="user", content="fix the parser")]
        return asyncio.run(Compactor(endpoint, "m").summarize(msgs))

    def test_returns_the_summary_text(self):
        ep = _Endpoint()
        self.assertEqual(self.summarize(ep), "## Goal\nship it")

    def test_the_summary_call_carries_no_tools_and_no_reasoning(self):
        """A summarizer holding the agent's tools is invited to call one, and
        paying high-effort reasoning to compress history is a bad trade."""
        ep = _Endpoint()
        self.summarize(ep)
        req = ep.requests[0]
        self.assertIsNone(req.tools)
        self.assertFalse(req.thinking)
        self.assertFalse(req.stream)
        self.assertEqual(req.reasoning_effort, "low")

    def test_it_does_not_pay_the_cache_write_premium(self):
        """This prompt is never sent again, so writing it to the cache is pure
        cost on providers that charge for the write."""
        ep = _Endpoint()
        self.summarize(ep)
        self.assertTrue(ep.requests[0].no_cache_write)

    def test_it_does_not_reuse_the_agent_prefix(self):
        ep = _Endpoint()
        self.summarize(ep)
        systems = [m for m in ep.requests[0].messages if m.role == "system"]
        self.assertEqual(len(systems), 1)
        self.assertIn("compacting the transcript", systems[0].content)

    def test_a_failure_is_swallowed_so_the_turn_survives(self):
        self.assertEqual(self.summarize(_Endpoint(error=TimeoutError())), "")

    def test_an_empty_reply_is_reported_as_no_summary(self):
        self.assertEqual(self.summarize(_Endpoint(reply="   ")), "")

    def test_an_empty_transcript_is_not_sent_at_all(self):
        ep = _Endpoint()
        self.assertEqual(self.summarize(ep, msgs=[]), "")
        self.assertEqual(ep.requests, [])


if __name__ == "__main__":
    unittest.main()
