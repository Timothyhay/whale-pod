"""Agent core loop.

Ties together the endpoint (multi-provider), the prefix-cache-friendly message
manager, the tool registry and the Thinking/Instant switch.

Flow per user turn:

  1. append the user message (append-only, so the cached prefix survives)
  2. stream the assistant response, collecting text + reasoning + tool calls
  3. if there are tool calls:
       - reads run concurrently and are checked against the context ledger, so
         a file already in the window is not shipped a second time
       - writes are *planned* (diff computed, nothing written), confirmed, then
         committed
     append the results and go back to (2)
  4. return the final assistant text

Design notes worth keeping in mind when editing this file:

  * **Confirmation is fail-closed.** A write with no confirmation callback is
    refused. It used to be approved, which meant any embedding of Agent without
    a UI silently got an auto-approving agent.
  * **Tools run off the event loop.** ``registry.dispatch`` is blocking
    (filesystem, subprocess); calling it directly from a coroutine froze
    streaming and the UI for its whole duration.
  * **Retries key on status codes**, not on substrings of the error message,
    and only happen before any output has been shown to the user.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from ..endpoints.base import ChatRequest, EndpointError, StreamingDelta, Usage
from ..tools.base import ToolResult, parse_args, parse_args_checked
from ..tools.plan import WritePlan
from ..tools.registry import ToolRegistry
from .compaction import Compactor, summary_message
from .ledger import ContextLedger, file_identity

READ_ONLY_TOOLS = ("read_file", "read_dir", "grep", "tree_view")


@dataclass
class ConfirmRequest:
    """What the UI needs to ask the user about one write."""
    tool: str
    args: dict
    summary: str
    preview: str
    plan: Optional[WritePlan] = None

    @property
    def is_command(self) -> bool:
        return bool(self.plan and self.plan.command)


@dataclass
class ToolEvent:
    """One observable moment of a tool call, for a UI to draw.

    Reported for *every* call including the ones that never reach a tool — a
    malformed-arguments failure or a hallucinated name is exactly the moment a
    user most wants to see in the trace.
    """
    phase: str                       # "start" | "end"
    name: str
    args: dict = field(default_factory=dict)
    result: Optional[ToolResult] = None
    duration: float = 0.0            # seconds; 0.0 on "start"


@dataclass
class AgentConfig:
    mode: str = "thinking"                       # thinking | instant
    reasoning_effort: str = "high"
    max_iterations: int = 12                     # tool-call loop cap
    max_tokens: Optional[int] = None
    model: str = "deepseek-ai/DeepSeek-V4-Flash-0731"
    # Awaitable[str]: 'yes' | 'no' | 'edit' | 'skip' | 'always'
    confirm_callback: Optional[Callable[[ConfirmRequest], Awaitable[str]]] = None
    notice_sink: Optional[Callable[[str], None]] = None
    max_retries: int = 2
    retry_delay: float = 2.0
    max_parallel_reads: int = 4
    # When the window fills, summarize the turns that are about to be cut rather
    # than just deleting them. Costs one small extra call per reduction; the
    # fallback if it fails is exactly the old behaviour.
    compaction: bool = True
    compaction_max_tokens: int = 2_000


class Agent:
    def __init__(self, endpoint, registry: ToolRegistry, mm,
                 config: Optional[AgentConfig] = None,
                 stream_sink: Optional[Callable[[StreamingDelta], None]] = None,
                 ledger: Optional[ContextLedger] = None,
                 tool_sink: Optional[Callable[[ToolEvent], None]] = None):
        self.endpoint = endpoint
        self.registry = registry
        self.mm = mm
        self.config = config or AgentConfig()
        self.stream_sink = stream_sink
        # Tool calls are the half of a turn the model does not narrate; without a
        # sink the UI could only show the prose and the user had to infer what was
        # read or run from the answer.
        self.tool_sink = tool_sink
        self.ledger = ledger or ContextLedger()
        # The registry invalidates ledger entries when a write lands.
        self.registry.ledger = self.ledger
        self._mode = self.config.mode
        self._always_approve: set[str] = set()
        self.last_usage: Optional[Usage] = None
        # Built on first use so a session that never fills its window never
        # constructs one, and tests can inject their own.
        self._compactor: Optional[Compactor] = None

    # -- mode -------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in ("thinking", "instant"):
            raise ValueError(f"unknown mode {mode!r}")
        self._mode = mode

    def toggle_mode(self) -> str:
        self._mode = "instant" if self._mode == "thinking" else "thinking"
        return self._mode

    # -- plumbing ----------------------------------------------------------
    def _notice(self, text: str) -> None:
        if self.config.notice_sink:
            self.config.notice_sink(text)

    def _request(self) -> ChatRequest:
        return ChatRequest(
            model=self.config.model,
            messages=self.mm.ordered_messages(),
            tools=self.registry.schemas(),
            thinking=self._mode == "thinking",
            reasoning_effort=self.config.reasoning_effort,
            max_tokens=self.config.max_tokens,
        )

    # -- main turn ---------------------------------------------------------
    async def run_turn(self, user_text: str, add_user: bool = True) -> str:
        """Run one user turn (multi-step tool loop). Returns the final text."""
        # Repair *before* appending the new user message: open_tool_calls()
        # scans back only to the last user turn, so adding ours first would hide
        # the dangling calls it is meant to find, and the request would go out
        # with an unanswered tool call — a 400 on every provider.
        closed = self.mm.close_open_tool_calls("(interrupted, not executed)")
        if closed:
            self._notice(f"repaired {closed} unanswered tool call(s) from the "
                         f"previous turn")
        if user_text and add_user:
            self.mm.add_user(user_text)
        self.ledger.turn += 1

        content = ""
        for step in range(self.config.max_iterations):
            await self._reduce_context()

            content, reasoning, tool_calls = await self._stream_with_retry()

            if not tool_calls:
                self.mm.add_assistant(content=content or "",
                                      reasoning=reasoning or None)
                return content

            self.mm.add_assistant(content=content or "", tool_calls=tool_calls,
                                  reasoning=reasoning or None)
            await self._execute_tools(tool_calls)

        # Cap reached: say so in the log and to the user, rather than returning
        # a stale half-finished string as if it were the answer.
        msg = (f"[stopped after {self.config.max_iterations} tool steps without "
               f"a final answer. Say 'continue' to let it keep going.]")
        self.mm.add_note(msg, role="user")
        self._notice(msg)
        return content or msg

    # -- context reduction -------------------------------------------------
    async def _reduce_context(self) -> None:
        """Make room in the window, preferring a summary over a hole.

        Compaction and pruning cut at the same place and cost the same cache
        miss; the difference is whether the session keeps what it learned. So
        compaction is tried first and pruning is the fallback — a summarizer that
        times out degrades the session instead of failing the user's turn.
        """
        if not self.mm.needs_reduction():
            return
        evt = None
        if self.config.compaction:
            evt = await self._compact()
        if evt is None:
            evt = self.mm.prune_if_needed()
        if evt is None:
            return
        # The ledger's whole claim is "that content is still above you". The cut
        # just made some of it untrue, so retract those entries or the next
        # re-read is answered with a pointer into a gap. Found by
        # bench/validate.py; seen firing twice in the live acceptance run.
        forgotten = self.ledger.forget_messages(evt.dropped_tool_call_ids)
        note = evt.describe()
        if forgotten:
            note += f"; {forgotten} loaded file range(s) can be re-read now"
        self._notice(note)

    async def compact_now(self, keep_turns: int = 1):
        """Manual compaction: summarise all but the last ``keep_turns`` turns.

        Unlike :meth:`_compact` this is user-triggered via ``/compact``, so it
        does not wait for the context window to fill.  Everything else — the
        serialiser, the summary prompt, the cache discipline — is shared so
        that manual and automatic compaction produce the same artifact and the
        same cache behaviour.
        """
        idx, turns, tokens, ids, mid = self.mm.plan_manual_reduction(
            keep_turns)
        if idx <= 0:
            return None
        if self._compactor is None:
            self._compactor = Compactor(
                self.endpoint, self.config.model,
                max_tokens=self.config.compaction_max_tokens)
        self._notice(f"compacting {turns} old turn(s) (~{tokens:,} tokens)…")
        summary = await self._compactor.summarize(self.mm.history[:idx])
        if not summary:
            self._notice("compaction produced no summary; falling back to "
                         "dropping the oldest turns")
            return None
        evt = self.mm.compact(summary_message(summary, turns, tokens),
                              idx, turns, tokens, ids, mid)
        if evt:
            forgotten = self.ledger.forget_messages(evt.dropped_tool_call_ids)
            note = evt.describe()
            if forgotten:
                note += f"; {forgotten} loaded file range(s) can be re-read now"
            self._notice(note)
        return evt

    async def _compact(self):
        """Summarize the slice that is about to be cut. None if not possible."""
        idx, turns, tokens, ids, mid = self.mm.plan_reduction()
        if idx <= 0:
            return None
        if self._compactor is None:
            self._compactor = Compactor(
                self.endpoint, self.config.model,
                max_tokens=self.config.compaction_max_tokens)
        self._notice(f"compacting {turns} old turn(s) (~{tokens:,} tokens)…")
        summary = await self._compactor.summarize(self.mm.history[:idx])
        if not summary:
            self._notice("compaction produced no summary; falling back to "
                         "dropping the oldest turns")
            return None
        return self.mm.compact(summary_message(summary, turns, tokens),
                               idx, turns, tokens, ids, mid)

    # -- streaming ---------------------------------------------------------
    async def _stream_with_retry(self) -> tuple[str, Optional[str], list]:
        attempt = 0
        while True:
            try:
                return await self._stream_once()
            except EndpointError as e:
                # Both conditions, not either: the status code says whether a
                # retry could *help*, ``before_output`` says whether it is
                # *safe*. OR-ing them retried 400s forever and re-ran turns whose
                # text was already on the user's screen, duplicating it.
                safe = getattr(e, "before_output", True)
                if not (e.retryable and safe and attempt < self.config.max_retries):
                    raise
                delay = self.config.retry_delay * (2 ** attempt)
                attempt += 1
                self._notice(f"{e} — retrying in {delay:.0f}s "
                             f"({attempt}/{self.config.max_retries})")
                await asyncio.sleep(delay)

    async def _stream_once(self) -> tuple[str, Optional[str], list]:
        """One streaming turn. Returns (content, reasoning, merged tool calls)."""
        content, reasoning_parts, deltas = "", [], []
        usage: Optional[Usage] = None
        produced = False
        try:
            async for d in self.endpoint.stream_chat(self._request()):
                if self.stream_sink:
                    self.stream_sink(d)
                if d.content:
                    content += d.content
                    produced = True
                if d.reasoning:
                    reasoning_parts.append(d.reasoning)
                    produced = True
                if d.tool_calls:
                    deltas.append(d)
                    produced = True
                if d.usage:
                    if usage is None:
                        usage = Usage()
                    usage.merge(d.usage)
        except EndpointError as e:
            # Retrying after partial output would duplicate it on screen.
            e.before_output = not produced      # type: ignore[attr-defined]
            raise

        self.last_usage = usage
        self.mm.record_usage(usage)
        return content, "".join(reasoning_parts) or None, self._merge_calls(deltas)

    def _merge_calls(self, deltas: list[StreamingDelta]) -> list:
        merged: dict[int, dict] = {}
        for d in deltas:
            for tc in d.tool_calls or []:
                slot = merged.setdefault(
                    tc.index, {"id": None, "name": None, "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.name:
                    slot["name"] = tc.name
                slot["arguments"] += tc.arguments or ""
        out = []
        for idx in sorted(merged):
            s = merged[idx]
            out.append({
                "id": s["id"] or f"call_{idx}",
                "type": "function",
                "function": {"name": s["name"] or "",
                             "arguments": s["arguments"] or "{}"},
            })
        return out

    # -- tool execution ----------------------------------------------------
    async def _execute_tools(self, calls: list[dict]) -> None:
        """Run a batch of tool calls, appending one result per call.

        Consecutive read-only calls run concurrently; writes stay strictly
        sequential (each needs its own confirmation, and two writes to one file
        must not race). Results are appended in call order either way, because
        providers match them to the request by position and id.
        """
        i = 0
        while i < len(calls):
            name = (calls[i].get("function") or {}).get("name", "")
            if name in READ_ONLY_TOOLS:
                j = i
                while (j < len(calls)
                       and (calls[j].get("function") or {}).get("name", "")
                       in READ_ONLY_TOOLS
                       and j - i < self.config.max_parallel_reads):
                    j += 1
                batch = calls[i:j]
                results = await asyncio.gather(
                    *(self._observed(c, self._run_read) for c in batch))
                for call, out in zip(batch, results):
                    self._append_result(call, out)
                i = j
                continue
            out = await self._observed(calls[i], self._run_write_or_other)
            self._append_result(calls[i], out)
            i += 1

    async def _observed(self, call: dict, run) -> ToolResult:
        """Run one call, bracketed by tool events.

        A wrapper rather than emitting from inside the runners: those have half a
        dozen early returns each, and an event tied to one of them is an event
        that silently goes missing for the others.
        """
        fn = call.get("function") or {}
        name = fn.get("name") or "tool"
        # Lenient parse: this is for display, and a call whose arguments are
        # broken JSON still has to appear in the trace. The runner does the
        # checked parse and reports the error to the model.
        args = parse_args(fn.get("arguments", ""))
        started = time.monotonic()
        self._emit_tool(ToolEvent("start", name, args))
        try:
            result = await run(call)
        except BaseException as e:
            self._emit_tool(ToolEvent(
                "end", name, args,
                ToolResult(ok=False, error=f"{type(e).__name__}: {e}"),
                time.monotonic() - started))
            raise
        self._emit_tool(ToolEvent("end", name, args, result,
                                  time.monotonic() - started))
        return result

    def _emit_tool(self, event: ToolEvent) -> None:
        if self.tool_sink is None:
            return
        try:
            self.tool_sink(event)
        except Exception:
            # A renderer that throws must not take the turn down with it.
            pass

    def _append_result(self, call: dict, result: ToolResult) -> None:
        name = (call.get("function") or {}).get("name", "tool")
        text = f"[{'ok' if result.ok else 'error'}] {result.to_text()}"
        self.mm.add_tool_result(call.get("id") or "", text, name)

    def _args_or_error(self, call: dict) -> tuple[str, dict, Optional[ToolResult]]:
        fn = call.get("function") or {}
        name = fn.get("name", "")
        args, err = parse_args_checked(fn.get("arguments", ""))
        if err:
            return name, args, ToolResult(
                ok=False, error=f"{name or 'tool'}: {err}")
        if not name:
            return name, args, ToolResult(ok=False, error="tool call had no name")
        if not self.registry.known(name):
            return name, args, ToolResult(
                ok=False,
                error=f"unknown tool {name!r}. Available: "
                      f"{', '.join(self.registry.tool_names())}")
        return name, args, None

    async def _run_read(self, call: dict) -> ToolResult:
        name, args, err = self._args_or_error(call)
        if err:
            return err
        if name == "read_file":
            hit = self._ledger_hit(args)
            if hit is not None:
                return hit
        result = await asyncio.to_thread(self.registry.dispatch, name, args)
        if name == "read_file" and result.ok:
            self._ledger_note(result, call.get("id") or "")
        return result

    async def _run_write_or_other(self, call: dict) -> ToolResult:
        name, args, err = self._args_or_error(call)
        if err:
            return err
        if not self.registry.is_write(name):
            result = await asyncio.to_thread(self.registry.dispatch, name, args)
            if (result.meta or {}).get("repo_map"):
                self._reinstall_repo_map()
            return result

        plan = await asyncio.to_thread(self.registry.plan, name, args)
        if plan is None:
            return ToolResult(ok=False, error=f"{name}: could not be planned")
        if not plan.ok:
            # A plan that cannot be built is not worth asking the user about;
            # the model gets the reason and can correct itself.
            return ToolResult(ok=False, error=plan.error)
        if plan.is_noop():
            return ToolResult(ok=True,
                              output=f"# {name}: no change ({plan.summary()})")

        decision = await self._confirm(name, args, plan)
        if decision == "always":
            self._always_approve.add(name)
            decision = "yes"
        if decision not in ("yes", "edit"):
            return ToolResult(
                ok=False,
                error=(f"the user declined this {name}. Do not retry it; ask "
                       f"what they would prefer."))
        return await asyncio.to_thread(self.registry.commit, plan, True)

    async def _confirm(self, name: str, args: dict, plan: WritePlan) -> str:
        """Ask the UI for a decision. Fail-closed when nobody can be asked."""
        if self.registry.guard.auto_approve() or name in self._always_approve:
            return "yes"
        if self.config.confirm_callback is None:
            # No UI to ask => refuse. Auto-approving here (the old behaviour)
            # let a library embedding write to disk unattended.
            self._notice(f"refused {name}: no confirmation handler is attached "
                         f"(use --yes to approve writes automatically)")
            return "no"
        req = ConfirmRequest(tool=name, args=args, plan=plan,
                             summary=plan.summary(), preview=plan.preview())
        decision = await self.config.confirm_callback(req)
        d = (decision or "").strip().lower()
        return {"y": "yes", "yes": "yes", "e": "edit", "edit": "edit",
                "a": "always", "always": "always",
                "n": "no", "no": "no", "s": "skip", "skip": "skip"}.get(d, "no")

    # -- repo map ----------------------------------------------------------
    def _reinstall_repo_map(self) -> None:
        """Put a refreshed map back into the stable prefix.

        This costs a prefix-cache miss for the map's own tokens, which is why
        the map sits *after* the system prompt: the prompt's tokens stay cached.
        """
        idx = getattr(self.registry, "repo_index", None)
        if idx is None:
            return
        from .prompt import repo_map_section
        self.mm.set_repo_map(repo_map_section(idx.render()))
        self._notice(f"repo map refreshed ({idx.symbol_count} symbols); "
                     f"the map section of the cached prefix is now cold")

    # -- ledger ------------------------------------------------------------
    def _ledger_hit(self, args: dict) -> Optional[ToolResult]:
        """Short-circuit a re-read of content already in the window."""
        path = args.get("path", "")
        if not path:
            return None
        try:
            resolved = self.registry.guard.resolve_within(path)
        except Exception:
            return None
        start = int(args.get("start", 0) or 0)
        end = int(args.get("end", 0) or 0)
        entry = self.ledger.find_current(path, start, end,
                                         file_identity(resolved))
        if entry is None:
            return None
        self.ledger.record_hit(entry)
        return ToolResult(ok=True, output=(
            f"# {entry.label()} is already in this conversation (unchanged "
            f"since it was shown above) — not re-sent. Scroll up to the earlier "
            f"result for its contents."),
            # So the trace can say "already in context" rather than reporting a
            # one-line read, which looks like a truncated file.
            meta={"ledger_hit": True})

    def _ledger_note(self, result: ToolResult, message_id: str = "") -> None:
        info = (result.meta or {}).get("read")
        if not info:
            return
        try:
            resolved = self.registry.guard.resolve_within(info["path"])
        except Exception:
            return
        self.ledger.note_read(
            info["path"], int(info.get("start", 0)), int(info.get("end", 0)),
            file_identity(resolved),
            tokens=max(1, int(info.get("chars", 0)) // 4),
            # Tied to the message that carries it, so a prune can retract it.
            message_id=message_id,
            # Only a complete, untruncated read may answer "the whole file is
            # already above"; a truncated one records the delivered range only.
            complete=bool(info.get("complete")),
        )
