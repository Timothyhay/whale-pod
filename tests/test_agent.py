"""Agent loop + context ledger, driven by a scripted fake endpoint.

No network: ``FakeEndpoint`` replays a list of pre-baked turns and records the
payloads it was asked to send, which is what lets us assert on things like
"reasoning was never echoed upstream" and "the confirmation happened before the
write".
"""
import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from whalepod.core.agent import Agent, AgentConfig, ConfirmRequest
from whalepod.core.ledger import ContextLedger, file_identity
from whalepod.core.messages import MessageManager
from whalepod.endpoints.base import (
    ChatRequest, ChatResponse, EndpointError, StreamingDelta, ToolCallDelta,
    Usage,
)
from whalepod.tools.registry import ToolRegistry


def call(cid, name, args="{}"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


class Turn:
    """One scripted assistant response (or an error to raise)."""

    def __init__(self, content="", reasoning="", tool_calls=(), usage=None,
                 error=None, chunks=None):
        self.content = content
        self.reasoning = reasoning
        self.tool_calls = list(tool_calls)
        self.usage = usage
        self.error = error
        self.chunks = chunks


class FakeEndpoint:
    def __init__(self, turns, summary=None):
        self.turns = list(turns)
        self.requests: list[ChatRequest] = []
        # Non-streaming replies, used only by the compaction summarizer. None
        # means "this endpoint cannot summarize", which is how the fallback path
        # gets exercised.
        self.summary = summary
        self.summary_requests: list[ChatRequest] = []

    async def chat(self, req: ChatRequest):
        self.summary_requests.append(req)
        if self.summary is None:
            raise EndpointError("no summarizer here", status_code=503)
        return ChatResponse(content=self.summary)

    async def stream_chat(self, req: ChatRequest):
        self.requests.append(req)
        turn = self.turns.pop(0) if self.turns else Turn(content="(done)")
        if turn.error is not None:
            raise turn.error
        if turn.reasoning:
            yield StreamingDelta(reasoning=turn.reasoning)
        for piece in (turn.chunks if turn.chunks is not None
                      else ([turn.content] if turn.content else [])):
            yield StreamingDelta(content=piece)
        for i, tc in enumerate(turn.tool_calls):
            fn = tc["function"]
            # Arguments arrive in fragments, as they do on the wire.
            yield StreamingDelta(tool_calls=[ToolCallDelta(
                index=i, id=tc["id"], name=fn["name"], arguments="")])
            for frag in (fn["arguments"][:2], fn["arguments"][2:]):
                yield StreamingDelta(tool_calls=[ToolCallDelta(
                    index=i, id=None, name=None, arguments=frag)])
        if turn.usage:
            yield StreamingDelta(usage=turn.usage)

    async def aclose(self):
        pass


class AgentCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self._prev = os.environ.get("WHALEPOD_BACKUP_DIR")
        os.environ["WHALEPOD_BACKUP_DIR"] = str(self.root / ".backups")
        self.notices: list[str] = []

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("WHALEPOD_BACKUP_DIR", None)
        else:
            os.environ["WHALEPOD_BACKUP_DIR"] = self._prev
        self._tmp.cleanup()

    def build(self, turns, sandbox="confirm", confirm=None, summary=None, **cfg):
        self.endpoint = FakeEndpoint(turns, summary=summary)
        self.ledger = ContextLedger()
        self.registry = ToolRegistry(self.root, sandbox_mode=sandbox,
                                     ledger=self.ledger)
        self.mm = MessageManager()
        self.mm.set_system("SYS")
        self.mm.set_tools(self.registry.schemas())
        agent = Agent(self.endpoint, self.registry, self.mm,
                      AgentConfig(confirm_callback=confirm,
                                  notice_sink=self.notices.append, **cfg),
                      ledger=self.ledger)
        return agent

    def run_turn(self, agent, text="do it"):
        return asyncio.run(agent.run_turn(text))

    def write(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")
        return p


# ----------------------------------------------------------- basic loop
class TestBasicLoop(AgentCase):
    def test_plain_answer(self):
        agent = self.build([Turn(content="hello there")])
        self.assertEqual(self.run_turn(agent), "hello there")
        self.assertEqual([m.role for m in self.mm.history],
                         ["user", "assistant"])

    def test_tool_then_answer(self):
        self.write("f.py", "x = 1\n")
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "f.py"}')]),
            Turn(content="it sets x"),
        ])
        self.assertEqual(self.run_turn(agent), "it sets x")
        self.assertEqual([m.role for m in self.mm.history],
                         ["user", "assistant", "tool", "assistant"])
        self.assertIn("x = 1", self.mm.history[2].content)

    def test_streamed_tool_arguments_are_accumulated(self):
        """Fragmented arguments used to arrive as {} — the deltas were dropped."""
        self.write("f.py", "target\n")
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "f.py"}')]),
            Turn(content="ok"),
        ])
        self.run_turn(agent)
        self.assertIn("target", self.mm.history[2].content)

    def test_reasoning_is_kept_locally_but_never_sent_back(self):
        agent = self.build([Turn(content="answer",
                                 reasoning="internal thoughts")])
        self.run_turn(agent)
        self.assertEqual(self.mm.history[-1].reasoning, "internal thoughts")
        # The next request must not carry it upstream.
        agent2_turns = [m for m in self.mm.ordered_messages()]
        self.assertTrue(any(m.reasoning for m in agent2_turns))
        from whalepod.endpoints.vllm import VLLMEndpoint
        payload = VLLMEndpoint("https://x", "k")._payload(
            ChatRequest(model="m", messages=agent2_turns))
        self.assertNotIn("internal thoughts", repr(payload))

    def test_iteration_cap_is_reported_not_silently_hit(self):
        self.write("f.py", "x\n")
        looping = [Turn(tool_calls=[call(f"c{i}", "read_dir", '{"path": "."}')])
                   for i in range(6)]
        agent = self.build(looping, max_iterations=3)
        out = self.run_turn(agent)
        self.assertIn("stopped after 3 tool steps", out)
        self.assertTrue(any("stopped after 3" in n for n in self.notices))

    def test_usage_is_recorded_from_the_stream(self):
        agent = self.build([Turn(content="hi", usage=Usage(
            prompt_tokens=1000, completion_tokens=10, cached_tokens=900,
            requests=1, measured=True))])
        self.run_turn(agent)
        self.assertAlmostEqual(self.mm.stats().cache_hit_rate, 0.9)


# ------------------------------------------------------ tool robustness
class TestToolRobustness(AgentCase):
    def test_unknown_tool_gets_the_tool_list_back(self):
        agent = self.build([Turn(tool_calls=[call("c1", "nope")]),
                            Turn(content="ok")])
        self.run_turn(agent)
        self.assertIn("read_file", self.mm.history[2].content)

    def test_malformed_arguments_are_explained(self):
        agent = self.build([Turn(tool_calls=[call("c1", "read_file", '{"path"')]),
                            Turn(content="ok")])
        self.run_turn(agent)
        self.assertIn("JSON", self.mm.history[2].content)

    def test_every_call_gets_exactly_one_result(self):
        """A provider matches results to calls by id; a missing one is a 400."""
        self.write("a.py", "A\n")
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "a.py"}'),
                             call("c2", "read_file", '{"path": "missing.py"}'),
                             call("c3", "nope")]),
            Turn(content="ok"),
        ])
        self.run_turn(agent)
        results = [m for m in self.mm.history if m.role == "tool"]
        self.assertEqual([m.tool_call_id for m in results], ["c1", "c2", "c3"])

    def test_interrupted_tool_calls_are_repaired_next_turn(self):
        agent = self.build([Turn(content="ok")])
        self.mm.add_user("earlier")
        self.mm.add_assistant("", tool_calls=[call("orphan", "read_file")])
        self.run_turn(agent, "continue")
        self.assertEqual(self.mm.open_tool_calls(), [])
        self.assertTrue(any("repaired 1" in n for n in self.notices))

    def test_reads_in_one_batch_run_concurrently(self):
        for name in ("a.py", "b.py", "c.py"):
            self.write(name, f"# {name}\n")
        order: list[str] = []
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "a.py"}'),
                             call("c2", "read_file", '{"path": "b.py"}'),
                             call("c3", "read_file", '{"path": "c.py"}')]),
            Turn(content="ok"),
        ])
        original = self.registry.dispatch

        def traced(name, args):
            order.append(args.get("path", name))
            return original(name, args)

        self.registry.dispatch = traced
        self.run_turn(agent)
        # Results are appended in call order regardless of completion order.
        contents = [m.content for m in self.mm.history if m.role == "tool"]
        self.assertEqual(len(contents), 3)
        self.assertIn("a.py", contents[0])
        self.assertIn("c.py", contents[2])
        self.assertEqual(sorted(order), ["a.py", "b.py", "c.py"])


# ------------------------------------------------------------ tool events
class TestToolEvents(AgentCase):
    """What the UI is told about tool calls while a turn runs."""

    def observe(self, agent):
        events: list = []
        agent.tool_sink = events.append
        return events

    def test_every_call_is_bracketed_by_a_start_and_an_end(self):
        self.write("f.py", "x = 1\n")
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "f.py"}')]),
            Turn(content="ok"),
        ])
        events = self.observe(agent)
        self.run_turn(agent)
        self.assertEqual([(e.phase, e.name) for e in events],
                         [("start", "read_file"), ("end", "read_file")])
        self.assertEqual(events[0].args, {"path": "f.py"})
        self.assertTrue(events[1].result.ok)

    def test_a_call_that_never_reaches_a_tool_is_still_traced(self):
        """Broken JSON and unknown names are exactly what a user wants to see."""
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path"'),
                             call("c2", "nope", "{}")]),
            Turn(content="ok"),
        ])
        events = self.observe(agent)
        self.run_turn(agent)
        ends = [e for e in events if e.phase == "end"]
        self.assertEqual([e.name for e in ends], ["read_file", "nope"])
        self.assertFalse(any(e.result.ok for e in ends))

    def test_the_call_is_announced_before_the_user_is_asked_to_confirm(self):
        """Otherwise the confirmation panel is the first sign anything happened."""
        self.write("f.py", "x = 1\n")
        seen: list = []

        async def confirm(req):
            seen.append(("confirm", req.tool))
            return "yes"

        agent = self.build([
            Turn(tool_calls=[call("c1", "edit_file",
                                  '{"path": "f.py", "old": "x = 1", '
                                  '"new": "x = 2"}')]),
            Turn(content="done"),
        ], confirm=confirm)
        agent.tool_sink = lambda e: seen.append((e.phase, e.name))
        self.run_turn(agent)
        self.assertEqual(seen, [("start", "edit_file"),
                                ("confirm", "edit_file"),
                                ("end", "edit_file")])

    def test_a_ledger_hit_is_flagged_as_one(self):
        self.write("f.py", "CONTENT\n")
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "f.py"}')]),
            Turn(tool_calls=[call("c2", "read_file", '{"path": "f.py"}')]),
            Turn(content="ok"),
        ])
        events = self.observe(agent)
        self.run_turn(agent)
        ends = [e for e in events if e.phase == "end"]
        self.assertFalse(ends[0].result.meta.get("ledger_hit"))
        self.assertTrue(ends[1].result.meta.get("ledger_hit"))

    def test_a_sink_that_raises_does_not_take_the_turn_down(self):
        self.write("f.py", "x\n")

        def broken(event):
            raise RuntimeError("bad renderer")

        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "f.py"}')]),
            Turn(content="ok"),
        ])
        agent.tool_sink = broken
        self.assertEqual(self.run_turn(agent), "ok")


# ---------------------------------------------------------- confirmation
class TestConfirmation(AgentCase):
    def _edit_turns(self):
        return [Turn(tool_calls=[call("c1", "edit_file",
                                      '{"path": "f.py", "old": "x = 1", '
                                      '"new": "x = 2"}')]),
                Turn(content="done")]

    def test_confirmation_happens_before_the_write(self):
        """Regression: the tool wrote first and diffed afterwards, so the user
        was asked about a change that had already landed."""
        self.write("f.py", "x = 1\n")
        seen = {}

        async def confirm(req: ConfirmRequest):
            seen["disk"] = (self.root / "f.py").read_text(encoding="utf-8")
            seen["diff"] = req.plan.diff
            return "yes"

        agent = self.build(self._edit_turns(), confirm=confirm)
        self.run_turn(agent)
        self.assertEqual(seen["disk"], "x = 1\n")          # not yet written
        self.assertIn("+x = 2", seen["diff"])
        self.assertEqual((self.root / "f.py").read_text(encoding="utf-8"),
                         "x = 2\n")                        # written after "yes"

    def test_declining_actually_prevents_the_write(self):
        self.write("f.py", "x = 1\n")

        async def confirm(req):
            return "no"

        agent = self.build(self._edit_turns(), confirm=confirm)
        self.run_turn(agent)
        self.assertEqual((self.root / "f.py").read_text(encoding="utf-8"),
                         "x = 1\n")
        self.assertIn("declined", self.mm.history[2].content)

    def test_no_confirm_handler_fails_closed(self):
        """Regression: a missing callback returned True, so any embedding
        without a UI got a silently auto-approving agent."""
        self.write("f.py", "x = 1\n")
        agent = self.build(self._edit_turns(), confirm=None)
        self.run_turn(agent)
        self.assertEqual((self.root / "f.py").read_text(encoding="utf-8"),
                         "x = 1\n")
        self.assertTrue(any("no confirmation handler" in n for n in self.notices))

    def test_always_skips_later_confirmations_for_that_tool(self):
        self.write("f.py", "1\n2\n")
        calls = []

        async def confirm(req):
            calls.append(req.tool)
            return "always"

        agent = self.build([
            Turn(tool_calls=[call("c1", "edit_file",
                                  '{"path": "f.py", "old": "1", "new": "one"}')]),
            Turn(tool_calls=[call("c2", "edit_file",
                                  '{"path": "f.py", "old": "2", "new": "two"}')]),
            Turn(content="done"),
        ], confirm=confirm)
        self.run_turn(agent)
        self.assertEqual(len(calls), 1)
        self.assertEqual((self.root / "f.py").read_text(encoding="utf-8"),
                         "one\ntwo\n")

    def test_auto_approve_mode_never_asks(self):
        self.write("f.py", "x = 1\n")
        asked = []

        async def confirm(req):
            asked.append(req.tool)
            return "no"

        agent = self.build(self._edit_turns(), sandbox="yes", confirm=confirm)
        self.run_turn(agent)
        self.assertEqual(asked, [])
        self.assertEqual((self.root / "f.py").read_text(encoding="utf-8"),
                         "x = 2\n")

    def test_unplannable_write_is_not_put_to_the_user(self):
        """A plan that cannot be built is the model's problem, not the user's."""
        asked = []

        async def confirm(req):
            asked.append(req.tool)
            return "yes"

        agent = self.build([
            Turn(tool_calls=[call("c1", "edit_file",
                                  '{"path": "missing.py", "old": "a", '
                                  '"new": "b"}')]),
            Turn(content="ok"),
        ], confirm=confirm)
        self.run_turn(agent)
        self.assertEqual(asked, [])
        self.assertIn("not found", self.mm.history[2].content)


# ---------------------------------------------------------------- retries
class TestRetries(AgentCase):
    def test_retries_a_transient_failure_before_any_output(self):
        agent = self.build([Turn(error=EndpointError("cold start",
                                                     status_code=503)),
                            Turn(content="second try")],
                           retry_delay=0.0)
        self.assertEqual(self.run_turn(agent), "second try")
        self.assertTrue(any("retrying" in n for n in self.notices))

    def test_does_not_retry_a_client_error(self):
        agent = self.build([Turn(error=EndpointError("bad request",
                                                     status_code=400))],
                           retry_delay=0.0)
        with self.assertRaises(EndpointError):
            self.run_turn(agent)

    def test_does_not_retry_after_partial_output(self):
        """Retrying mid-stream would duplicate text already on screen."""
        err = EndpointError("dropped", status_code=503)

        class Flaky(FakeEndpoint):
            async def stream_chat(self, req):
                self.requests.append(req)
                yield StreamingDelta(content="partial ")
                raise err

        self.endpoint = Flaky([])
        self.ledger = ContextLedger()
        self.registry = ToolRegistry(self.root, sandbox_mode="confirm",
                                     ledger=self.ledger)
        self.mm = MessageManager()
        agent = Agent(self.endpoint, self.registry, self.mm,
                      AgentConfig(retry_delay=0.0,
                                  notice_sink=self.notices.append),
                      ledger=self.ledger)
        with self.assertRaises(EndpointError):
            self.run_turn(agent)
        self.assertEqual(len(self.endpoint.requests), 1)


# ----------------------------------------------------------------- ledger
class TestLedger(AgentCase):
    def test_second_identical_read_is_a_pointer_not_the_file(self):
        self.write("big.py", "UNIQUE-CONTENT\n" * 5)
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "big.py"}')]),
            Turn(tool_calls=[call("c2", "read_file", '{"path": "big.py"}')]),
            Turn(content="ok"),
        ])
        self.run_turn(agent)
        results = [m.content for m in self.mm.history if m.role == "tool"]
        self.assertIn("UNIQUE-CONTENT", results[0])
        self.assertNotIn("UNIQUE-CONTENT", results[1])
        self.assertIn("already in this conversation", results[1])
        self.assertEqual(self.ledger.hits, 1)

    def test_a_changed_file_is_read_again(self):
        p = self.write("f.py", "VERSION-ONE\n")
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "f.py"}')]),
            Turn(content="first"),
        ])
        self.run_turn(agent)
        p.write_text("VERSION-TWO\n", encoding="utf-8", newline="")
        os.utime(p, (0, 0))          # force a different identity
        agent.endpoint.turns = [
            Turn(tool_calls=[call("c2", "read_file", '{"path": "f.py"}')]),
            Turn(content="second"),
        ]
        self.run_turn(agent, "again")
        results = [m.content for m in self.mm.history if m.role == "tool"]
        self.assertIn("VERSION-TWO", results[-1])

    def test_our_own_edit_invalidates_the_ledger(self):
        self.write("f.py", "OLD-LINE\n")

        async def confirm(req):
            return "yes"

        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "f.py"}')]),
            Turn(tool_calls=[call("c2", "edit_file",
                                  '{"path": "f.py", "old": "OLD-LINE", '
                                  '"new": "NEW-LINE"}')]),
            Turn(tool_calls=[call("c3", "read_file", '{"path": "f.py"}')]),
            Turn(content="ok"),
        ], confirm=confirm)
        self.run_turn(agent)
        results = [m.content for m in self.mm.history if m.role == "tool"]
        self.assertIn("NEW-LINE", results[-1])
        self.assertNotIn("already in this conversation", results[-1])

    def test_path_normalisation_does_not_defeat_the_ledger(self):
        self.write("pkg/f.py", "CONTENT\n")
        led = ContextLedger()
        ident = file_identity(self.root / "pkg/f.py")
        led.note_read("./pkg/f.py", 0, 0, ident, tokens=10)
        self.assertIsNotNone(led.find_current("pkg/f.py", 0, 0, ident))

    def test_a_wider_range_is_not_considered_covered(self):
        led = ContextLedger()
        led.note_read("f.py", 10, 20, (1, 2))
        self.assertIsNotNone(led.find_current("f.py", 12, 18, (1, 2)))
        self.assertIsNone(led.find_current("f.py", 5, 30, (1, 2)))
        self.assertIsNone(led.find_current("f.py", 0, 0, (1, 2)))

    def test_invalidate_only_touches_that_path(self):
        led = ContextLedger()
        led.note_read("a.py", 0, 0, (1, 2))
        led.note_read("b.py", 0, 0, (1, 2))
        led.invalidate("a.py")
        self.assertIsNone(led.find_current("a.py", 0, 0, (1, 2)))
        self.assertIsNotNone(led.find_current("b.py", 0, 0, (1, 2)))

    def test_pruning_retracts_the_ledger_entries_it_elided(self):
        """The ledger says "it is still above you" — a prune can make that false.

        Without this the model asks for a file, is told to scroll up, and finds
        the elision marker where the file used to be. Found by the bench.
        """
        self.write("big.py", "UNIQUE-CONTENT\n" * 400)
        # compaction=False pins this to the blind-prune path; the compacting one
        # has to retract just the same, and is covered below.
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "big.py"}')]),
            Turn(content="first"),
        ], compaction=False)
        self.run_turn(agent)
        self.assertEqual(len(self.ledger.entries), 1)

        # Shrink the window so the next turn must prune, then re-read.
        self.mm.window = 900
        agent.endpoint.turns = [
            Turn(tool_calls=[call("c2", "read_file", '{"path": "big.py"}')]),
            Turn(content="second"),
        ]
        self.run_turn(agent, "read it again")
        self.assertTrue(self.mm.prune_events, "expected a prune")
        results = [m.content for m in self.mm.history if m.role == "tool"]
        self.assertIn("UNIQUE-CONTENT", results[-1],
                      "an elided file must be genuinely re-readable")
        self.assertNotIn("already in this conversation", results[-1])

    def test_forget_messages_keeps_entries_still_in_the_window(self):
        led = ContextLedger()
        led.note_read("a.py", 0, 0, (1, 2), message_id="c1")
        led.note_read("b.py", 0, 0, (1, 2), message_id="c2")
        self.assertEqual(led.forget_messages(["c1"]), 1)
        self.assertIsNone(led.find_current("a.py", 0, 0, (1, 2)))
        self.assertIsNotNone(led.find_current("b.py", 0, 0, (1, 2)))
        self.assertEqual(led.forget_messages([]), 0)
        self.assertEqual(led.forget_messages([""]), 0)


# ------------------------------------------------------ context reduction
class TestContextReduction(AgentCase):
    """The agent's choice between summarizing the cut and just making it."""

    def _overflow(self, agent, text="read it again"):
        """Force the next turn to need reduction, then run it."""
        self.mm.window = 900
        self.mm.reserve_tokens = 0
        agent.endpoint.turns = [Turn(content="second")]
        self.run_turn(agent, text)

    def _seed(self, **cfg):
        self.write("big.py", "UNIQUE-CONTENT\n" * 400)
        agent = self.build([
            Turn(tool_calls=[call("c1", "read_file", '{"path": "big.py"}')]),
            Turn(content="first"),
        ], **cfg)
        self.run_turn(agent, "read big.py")
        return agent

    def test_compaction_is_preferred_over_a_blind_cut(self):
        agent = self._seed(summary="## Goal\nfix the parser\n## Next\nrun tests")
        self._overflow(agent)
        self.assertTrue(self.mm.compaction_events, "expected a compaction")
        self.assertFalse(self.mm.prune_events, "should not also prune")
        self.assertIn("fix the parser", self.mm.history[0].content)

    def test_the_summarizer_gets_the_slice_that_is_being_cut(self):
        agent = self._seed(summary="## Goal\nx")
        self._overflow(agent)
        sent = agent.endpoint.summary_requests[0].messages[-1].content
        self.assertIn("read big.py", sent)
        self.assertIn("read_file", sent)

    def test_a_failed_summary_falls_back_to_pruning(self):
        """summary=None makes the summarize call raise. The turn must still
        finish: compaction is an improvement on pruning, not a new way to fail.
        """
        agent = self._seed()
        self._overflow(agent)
        self.assertFalse(self.mm.compaction_events)
        self.assertTrue(self.mm.prune_events, "expected the fallback prune")
        self.assertTrue(any("falling back" in n for n in self.notices))

    def test_compaction_also_retracts_the_ledger(self):
        """The cut is the same, so the stale "it is still above you" claim is
        the same. A summary naming the file is not the file."""
        agent = self._seed(summary="## Files\nbig.py:1-400 — the thing")
        self.mm.window = 900
        self.mm.reserve_tokens = 0
        agent.endpoint.turns = [
            Turn(tool_calls=[call("c2", "read_file", '{"path": "big.py"}')]),
            Turn(content="second"),
        ]
        self.run_turn(agent, "read it again")
        self.assertTrue(self.mm.compaction_events)
        results = [m.content for m in self.mm.history if m.role == "tool"]
        self.assertIn("UNIQUE-CONTENT", results[-1])
        self.assertNotIn("already in this conversation", results[-1])

    def test_reduction_is_skipped_entirely_when_there_is_room(self):
        agent = self._seed(summary="## Goal\nx")
        agent.endpoint.turns = [Turn(content="second")]
        self.run_turn(agent, "again")
        self.assertEqual(agent.endpoint.summary_requests, [])
        self.assertFalse(self.mm.prune_events)

    def test_the_request_that_follows_a_reduction_is_inside_the_window(self):
        """The point of the reserve: reduce *before* sending, not after a 400."""
        agent = self._seed(summary="## Goal\nx")
        self._overflow(agent)
        sent = agent.endpoint.requests[-1]
        est = sum(self.mm._message_tokens(m) for m in sent.messages)
        self.assertLess(est, self.mm.window)


if __name__ == "__main__":
    unittest.main()
