"""Prefix-caching-friendly message manager.

DeepSeek's prefix cache is automatic and keyed on the longest common prefix of
consecutive requests. We cannot "turn it on", only avoid breaking it. There are
exactly **two** zones, because that is all the wire format actually supports:

  Zone 1 — stable prefix
      tool definitions, then one system message holding the system prompt
      followed by the repo-map summary. Byte-identical between turns, so it is
      always a cache hit. The prompt comes *before* the map so refreshing the
      map only invalidates the tail of the prefix, not all of it.

  Zone 2 — append-only history
      every user turn, assistant turn (text + tool calls) and tool result, in
      the order they happened. Only ever appended to.

An earlier design had a third "volatile working set" zone appended after
history. It was deleted: anything appended after history *is* the tail of the
history for caching purposes, and having two tails meant file contents were
re-sent on every request while pretending to be free. File content now enters
history once as a tool result and is tracked by :mod:`whalepod.core.ledger`.

Reducing the history is the one operation that costs a full cache miss, so it is
rare, chunked and reported rather than continuous and silent: nothing happens
until the context crosses :meth:`MessageManager.reduction_limit`, then the oldest
turns are removed in one pass down to ``prune_to``, and an event says what it
cost.

There are two ways to do the removal, and they share the cut point:

  * :meth:`MessageManager.prune_if_needed` deletes the turns and leaves a marker.
    Cheap, lossy, always available.
  * :meth:`MessageManager.compact` replaces them with a summary produced by
    :mod:`whalepod.core.compaction`. Same cut, same cache cost, one extra model
    call, and the session keeps its goal and its findings.

The agent prefers compaction and falls back to pruning, so a summarizer that
fails is a downgrade rather than a broken turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..endpoints.base import Message, Usage  # protocol-agnostic types


@dataclass
class PruneEvent:
    """What one pruning pass dropped — and what it cost."""
    turns_dropped: int
    messages_dropped: int
    tokens_dropped: int
    tokens_after: int
    # tool_call_ids of the dropped tool results. The context ledger keys its
    # "already in the window" claim on these, so it has to hear about them.
    dropped_tool_call_ids: list[str] = field(default_factory=list)
    mid_turn: bool = False

    def describe(self) -> str:
        where = " (cut inside a turn)" if self.mid_turn else ""
        return (f"pruned {self.turns_dropped} old turn(s){where} "
                f"(~{self.tokens_dropped:,} tokens); prefix cache is now cold, "
                f"the next request pays full price")


@dataclass
class ContextStats:
    """Local estimate of context size, plus what the server actually measured.

    ``estimated_*`` fields are our own tokenizer estimate and are always
    available. ``usage`` is the provider's report and is the only trustworthy
    source for cache hit rate — when it is absent we say so instead of
    inventing a number.
    """
    stable_tokens: int = 0
    history_tokens: int = 0
    total: int = 0
    window: int = 1_000_000
    messages: int = 0
    usage: Optional[Usage] = None          # last request, server-measured
    session_usage: Optional[Usage] = None  # cumulative, server-measured
    prunes: int = 0
    compactions: int = 0
    limit: int = 0                         # size at which history is reduced

    @property
    def fill(self) -> float:
        """Fraction of the context window the next request would occupy."""
        return self.total / self.window if self.window else 0.0

    @property
    def cache_hit_rate(self) -> Optional[float]:
        """Measured prefix-cache hit rate of the last request, or None."""
        return self.usage.cache_hit_rate if self.usage else None

    @property
    def session_cache_hit_rate(self) -> Optional[float]:
        return self.session_usage.cache_hit_rate if self.session_usage else None


class MessageManager:
    """Two-zone message store: a stable prefix and append-only history."""

    def __init__(self, window: int = 1_000_000,
                 prune_at: float = 0.9, prune_to: float = 0.5,
                 reserve_tokens: int = 16_384):
        self.window = window
        # Fractions of the window: leave it alone until prune_at, then cut back
        # to prune_to in a single pass. One expensive miss beats many.
        self.prune_at = prune_at
        self.prune_to = prune_to
        # …and an absolute floor on top of the fraction, because a fraction does
        # not know how big a *reply* is. At a 1M window, 0.9 leaves 100k of
        # headroom; at a 32k window it leaves 3.2k, which one reasoning reply
        # plus one tool result will blow through — the request then fails with a
        # context-length error instead of pruning. The trigger is whichever of
        # the two is reached first.
        self.reserve_tokens = max(0, int(reserve_tokens))

        # Zone 1 — stable prefix
        self.system_prompt: str = ""
        self.repo_map_summary: str = ""
        self.tool_schemas: list[dict] = []

        # Zone 2 — append-only history, with a parallel per-message token count
        # so size checks stay O(1) instead of re-estimating the whole log.
        self.history: list[Message] = []
        self._hist_tokens: list[int] = []
        self._hist_total: int = 0

        self.prune_events: list[PruneEvent] = []
        self.compaction_events: list = []
        self.last_usage: Optional[Usage] = None
        self.session_usage: Usage = Usage()

        from .tokenizer import estimate_tokens
        self._estimate = estimate_tokens

    # ------------------------------------------------------------ zone 1
    def set_system(self, text: str) -> None:
        self.system_prompt = text or ""

    def set_tools(self, schemas: list[dict]) -> None:
        self.tool_schemas = list(schemas or [])

    def set_repo_map(self, text: str) -> None:
        """Install the repo map. Kept separate from the prompt so a refresh
        invalidates only the map's own tokens, not the prompt before it."""
        self.repo_map_summary = text or ""

    def system_message(self) -> Optional[Message]:
        """The single system message: prompt first, then the repo map.

        Emitted as one message because providers differ on whether repeated
        system turns are legal, and DeepSeek expects a single one.
        """
        parts = [p for p in (self.system_prompt, self.repo_map_summary) if p]
        if not parts:
            return None
        return Message(role="system", content="\n\n".join(parts))

    # ------------------------------------------------------------ zone 2
    def _append(self, m: Message) -> Message:
        self.history.append(m)
        n = self._message_tokens(m)
        self._hist_tokens.append(n)
        self._hist_total += n
        return m

    def add_user(self, content: str) -> Message:
        return self._append(Message(role="user", content=content))

    def add_assistant(self, content: str = "", tool_calls: Optional[list] = None,
                      reasoning: Optional[str] = None) -> Message:
        return self._append(Message(role="assistant", content=content,
                                    tool_calls=tool_calls, reasoning=reasoning))

    def add_tool_result(self, tool_call_id: str, content: str,
                        name: str) -> Message:
        return self._append(Message(role="tool", content=content,
                                    tool_call_id=tool_call_id, name=name))

    def add_note(self, content: str, role: str = "user") -> Message:
        """Append an out-of-band note (elision marker, interruption, warning).

        Notes are ordinary appended history — there is no separate volatile
        zone for them to live in.
        """
        return self._append(Message(role=role, content=content))

    def clear_history(self) -> int:
        """Drop the conversation, keep the stable prefix. Returns messages dropped.

        The prefix is deliberately untouched: the point of /clear is to free the
        window, and re-sending an identical prefix is a cache *hit*, so there is
        nothing to gain by rebuilding it.
        """
        n = len(self.history)
        self.history.clear()
        self._hist_tokens.clear()
        self._hist_total = 0
        return n

    def open_tool_calls(self) -> list[dict]:
        """Tool calls at the tail of history that have no result yet.

        A turn interrupted between "model asked for tools" and "tools ran"
        leaves the log in a state most providers reject with a 400.
        """
        answered = {m.tool_call_id for m in self.history if m.role == "tool"}
        pending: list[dict] = []
        for m in reversed(self.history):
            if m.role == "assistant" and m.tool_calls:
                for tc in m.tool_calls:
                    if tc.get("id") and tc["id"] not in answered:
                        pending.append(tc)
                break
            if m.role == "user":
                break
        return pending

    def close_open_tool_calls(self, note: str = "interrupted") -> int:
        """Answer any dangling tool call so the log stays valid.

        Append-only repair: we add the missing results rather than editing the
        assistant turn, which keeps the cached prefix intact.
        """
        pending = self.open_tool_calls()
        for tc in pending:
            fn = tc.get("function") or {}
            self.add_tool_result(tc["id"], note, fn.get("name", "tool"))
        return len(pending)

    # ------------------------------------------------------------ flush
    def ordered_messages(self) -> list[Message]:
        """Assemble the request: stable prefix first, then history."""
        msgs: list[Message] = []
        sys_msg = self.system_message()
        if sys_msg is not None:
            msgs.append(sys_msg)
        msgs.extend(self.history)
        return msgs

    def tool_calls_this_turn(self) -> list[dict]:
        return list(self.tool_schemas)

    # ------------------------------------------------------------ pruning
    def reduction_limit(self) -> int:
        """Context size at which the history must be reduced."""
        frac = int(self.window * self.prune_at)
        if self.reserve_tokens:
            return min(frac, self.window - self.reserve_tokens)
        return frac

    def needs_reduction(self) -> bool:
        return self.estimated_total() > self.reduction_limit()

    def plan_manual_reduction(self, keep_turns: int = 1
                              ) -> tuple[int, int, int, list[str], bool]:
        """Cut point for a user-requested compaction, keeping the last N turns.

        Like :meth:`plan_reduction` but does not depend on the window fill
        level — it always compacts everything before the last ``keep_turns``
        turns, regardless of how many tokens are in flight.
        """
        if len(self.history) <= 1:
            return 0, 0, 0, [], False
        user_indices = [i for i, m in enumerate(self.history)
                        if m.role == "user"]
        if len(user_indices) <= keep_turns:
            return 0, 0, 0, [], False
        idx = user_indices[-keep_turns]
        turns = len(user_indices) - keep_turns
        tokens = sum(self._hist_tokens[:idx])
        ids = [m.tool_call_id for m in self.history[:idx]
               if m.role == "tool" and m.tool_call_id]
        return idx, max(turns, 1), tokens, ids, False

    def plan_reduction(self) -> tuple[int, int, int, list[str], bool]:
        """Where to cut, without cutting: ``(index, turns, tokens, ids, mid)``.

        Split out from :meth:`prune_if_needed` so compaction can summarize the
        exact slice that is about to disappear, and so the choice of cut point is
        testable on its own.
        """
        if not self.needs_reduction() or not self.history:
            return 0, 0, 0, [], False
        target = int(self.window * self.prune_to)
        stable = self.estimated_stable()
        idx = turns = 0
        mid_turn = False

        while idx < len(self.history) and stable + sum(
                self._hist_tokens[idx:]) > target:
            end = self._turn_end(idx)
            if end >= len(self.history):
                # The last turn on its own exceeds the target. Cutting inside it
                # is better than the old behaviour, which bailed out and left the
                # context over budget until the provider rejected it — the one
                # case where a long turn was guaranteed to fail. The cut still
                # respects tool-call integrity.
                inner = self._safe_cut_in(idx, target, stable)
                if inner > idx:
                    idx, mid_turn = inner, True
                break
            idx = end
            turns += 1

        if idx <= 0:
            return 0, 0, 0, [], False
        tokens = sum(self._hist_tokens[:idx])
        ids = [m.tool_call_id for m in self.history[:idx]
               if m.role == "tool" and m.tool_call_id]
        return idx, max(turns, 1), tokens, ids, mid_turn

    def _turn_end(self, start: int) -> int:
        """Index one past the turn beginning at ``start`` (next user message)."""
        j = start + 1
        while j < len(self.history) and self.history[j].role != "user":
            j += 1
        return j

    def _safe_cut_in(self, start: int, target: int, stable: int) -> int:
        """Largest cut inside one turn that keeps the message log valid.

        The only hard rule is that the remaining history must not begin with a
        tool result: a result whose assistant tool call has been deleted is an
        unanswered-call error on every provider — the same 400 that
        :meth:`close_open_tool_calls` exists to prevent from the other side.
        Everything else is a judgement call, and the judgement here is to keep as
        much of the recent turn as fits.
        """
        best = start
        for i in range(start + 1, len(self.history)):
            if self.history[i].role == "tool":
                continue
            if stable + sum(self._hist_tokens[i:]) <= target:
                best = i
                break
            best = i
        # Never leave the history empty: an empty log means the live turn was
        # erased, which is worse than being over budget.
        return best if best < len(self.history) else start

    def prune_if_needed(self) -> Optional[PruneEvent]:
        """Drop the oldest turns if the context crossed the reduction limit.

        Returns the :class:`PruneEvent` describing the cost, or None if nothing
        needed to happen (the common case). Cuts land on turn boundaries where
        possible, and never separate an assistant tool call from its results.
        """
        idx, turns, tokens, ids, mid = self.plan_reduction()
        if idx <= 0:
            return None
        self._cut(idx)
        evt = PruneEvent(turns_dropped=turns, messages_dropped=idx,
                         tokens_dropped=tokens,
                         tokens_after=self.estimated_total(),
                         dropped_tool_call_ids=ids, mid_turn=mid)
        self.prune_events.append(evt)
        # Leave a marker so the model knows the gap exists rather than silently
        # losing the earlier conversation.
        self._prepend(Message(
            role="user",
            content=(f"[earlier conversation elided: {turns} turn(s), "
                     f"~{tokens:,} tokens. Re-read files if unsure.]")))
        return evt

    def compact(self, summary: Message, idx: int, turns: int, tokens: int,
                ids: list[str], mid_turn: bool = False):
        """Replace ``history[:idx]`` with one summary message.

        Same cut and the same cache cost as a prune; the difference is what sits
        in the gap. Returns a :class:`~whalepod.core.compaction.CompactionEvent`.
        """
        from .compaction import CompactionEvent
        if idx <= 0:
            return None
        self._cut(idx)
        self._prepend(summary)
        evt = CompactionEvent(
            turns_dropped=turns, messages_dropped=idx, tokens_dropped=tokens,
            tokens_after=self.estimated_total(),
            summary_tokens=self._hist_tokens[0],
            dropped_tool_call_ids=ids, mid_turn=mid_turn)
        self.compaction_events.append(evt)
        return evt

    def _cut(self, idx: int) -> None:
        del self.history[:idx]
        del self._hist_tokens[:idx]
        self._hist_total = sum(self._hist_tokens)

    def _prepend(self, m: Message) -> None:
        self.history.insert(0, m)
        self._hist_tokens.insert(0, self._message_tokens(m))
        self._hist_total = sum(self._hist_tokens)

    # ------------------------------------------------------------ usage
    def record_usage(self, usage: Optional[Usage]) -> None:
        """Store the server's measured accounting for the last request."""
        if usage is None:
            return
        self.last_usage = usage
        self.session_usage.merge(usage)

    # ------------------------------------------------------------ token est
    def _message_tokens(self, m: Message) -> int:
        n = int(self._estimate(m.content or ""))
        for t in m.tool_calls or []:
            n += int(self._estimate(str(t)))
        return n

    def estimated_stable(self) -> int:
        n = 0
        if self.system_prompt:
            n += int(self._estimate(self.system_prompt))
        if self.repo_map_summary:
            n += int(self._estimate(self.repo_map_summary))
        if self.tool_schemas:
            n += int(self._estimate(str(self.tool_schemas)))
        return n

    def estimated_history(self) -> int:
        return self._hist_total

    def estimated_total(self) -> int:
        return self.estimated_stable() + self._hist_total

    def stats(self) -> ContextStats:
        stable = self.estimated_stable()
        return ContextStats(
            stable_tokens=stable,
            history_tokens=self._hist_total,
            total=stable + self._hist_total,
            window=self.window,
            messages=len(self.history),
            usage=self.last_usage,
            session_usage=(self.session_usage
                           if self.session_usage.requests else None),
            prunes=len(self.prune_events),
            compactions=len(self.compaction_events),
            limit=self.reduction_limit(),
        )
