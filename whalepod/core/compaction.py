"""Compaction — replace old turns with a summary instead of a hole.

Pruning (see :mod:`whalepod.core.messages`) frees the window by deleting the
oldest turns and leaving a marker that says "some conversation was elided".
That is honest but lossy in the worst possible way: what gets deleted first is
the *beginning* of the session, which is where the user said what they wanted.
The live acceptance run shows the cost concretely — a prune at request 18
retracted 11 loaded file ranges, so the agent had to rediscover its own findings
while the user's original instruction was gone.

Compaction keeps the same window budget and the same cache cost (rewriting the
head of history is a full prefix miss either way) but pays one small model call
to carry the knowledge across the cut:

    [conversation so far] ──▶ summary ──▶ [summary] + [recent turns kept intact]

Design constraints that shaped this:

  * **The summary call must not touch the main prefix.** It goes out as its own
    request with no tools and a purpose-built system prompt. Reusing the agent's
    prefix would make the summary request a cache *write* competing with the
    session's own prefix, and reusing the agent's tools would invite the
    summarizer to call one.
  * **Structure over prose.** A free-form paragraph reads well and is useless
    for recovery. The requested format names the files touched with line ranges,
    so after the cut the model can re-read precisely what it lost instead of
    re-exploring the repo. This pairs with the ledger retraction: the entries go
    away, the pointers to them survive.
  * **Failure is not fatal.** A timeout, a rate limit or an empty response falls
    back to the blind prune. Compaction is an improvement on pruning, not a new
    single point of failure in the turn loop.
  * **Reasoning is off.** Summarizing does not need a chain of thought, and
    paying high-effort reasoning tokens to compress history is a bad trade.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..endpoints.base import ChatRequest, Message

SUMMARY_SYSTEM = """\
You are compacting the transcript of a coding session so that work can continue \
in a smaller context window. You are not answering the user; you are writing the \
notes your successor will rely on.

Reply with exactly these sections, in this order, and nothing else:

## Goal
What the user asked for, in their terms, including any constraint or preference \
they stated. If the goal changed during the session, give the current one and \
note what changed.

## State
What has actually been done and verified so far, and what is still outstanding. \
Distinguish "done and checked" from "done, not checked" from "attempted and \
failed". Include concrete findings — the cause of a bug, the name of the \
function that matters, a command that works.

## Files
One line per file that was read or changed, as `path:lines — why it matters`. \
Mark changed files with (modified). This list is how the next turn re-reads what \
it needs, so be specific about paths and line ranges.

## Next
The immediate next step, as an instruction.

Rules:
- Preserve exact identifiers, paths, line numbers, commands and error text. \
Never paraphrase an error message.
- Do not invent progress. If something was never verified, say so.
- Omit pleasantries, restatements of tool mechanics, and anything the repository \
itself would tell you.
"""

MAX_SERIALIZED_CHARS = 220_000
MAX_TOOL_RESULT_CHARS = 2_000


@dataclass
class CompactionEvent:
    """What one compaction replaced, and what it cost."""
    turns_dropped: int
    messages_dropped: int
    tokens_dropped: int
    tokens_after: int
    summary_tokens: int = 0
    dropped_tool_call_ids: list[str] = field(default_factory=list)
    mid_turn: bool = False

    def describe(self) -> str:
        where = " (cut inside a turn)" if self.mid_turn else ""
        return (f"compacted {self.turns_dropped} old turn(s) into a "
                f"{self.summary_tokens:,}-token summary{where} "
                f"(~{self.tokens_dropped:,} tokens replaced); prefix cache is "
                f"now cold, the next request pays full price")


def serialize(messages, max_chars: int = MAX_SERIALIZED_CHARS) -> str:
    """Render history as a transcript for the summarizer.

    Tool results are cut hard at :data:`MAX_TOOL_RESULT_CHARS`: the summary needs
    to know *that* a file was read and what was concluded, not to receive the
    file again. Without this the summarize call is nearly as expensive as the
    context it is meant to shrink.

    Reasoning is dropped. It is the model's scratch work, it is never echoed
    upstream, and summarizing it teaches the successor nothing it cannot get from
    the conclusions.
    """
    out: list[str] = []
    for m in messages:
        role = m.role
        if role == "system":
            continue
        if role == "user":
            out.append(f"[User]: {m.content}")
        elif role == "assistant":
            if m.content:
                out.append(f"[Assistant]: {m.content}")
            for tc in m.tool_calls or []:
                fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                args = str(fn.get("arguments", ""))[:400]
                out.append(f"[Assistant calls {fn.get('name', 'tool')}]: {args}")
        elif role == "tool":
            body = m.content or ""
            if len(body) > MAX_TOOL_RESULT_CHARS:
                body = (body[:MAX_TOOL_RESULT_CHARS]
                        + f"\n… [{len(body) - MAX_TOOL_RESULT_CHARS} chars of "
                          f"this result omitted from the summary input]")
            out.append(f"[Result of {m.name or 'tool'}]: {body}")
    text = "\n\n".join(out)
    if len(text) > max_chars:
        # Keep the tail: the summarizer is told the goal is at the top, but a
        # transcript this long has already been compacted once, so its head is
        # itself a summary that the previous pass preserved.
        text = ("[earlier transcript omitted]\n\n" + text[-max_chars:])
    return text


class Compactor:
    """Turns a slice of history into a summary, via one small model call."""

    def __init__(self, endpoint, model: str, max_tokens: int = 2_000,
                 timeout_hint: Optional[float] = None):
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_hint = timeout_hint

    async def summarize(self, messages) -> str:
        """Summary text, or "" if the call could not produce one.

        Errors are swallowed deliberately: the caller's fallback (a blind prune)
        is always available, and a failed compaction must not fail the user's
        turn.
        """
        transcript = serialize(messages)
        if not transcript.strip():
            return ""
        req = ChatRequest(
            model=self.model,
            messages=[
                Message(role="system", content=SUMMARY_SYSTEM),
                Message(role="user",
                        content=f"Transcript to compact:\n\n{transcript}"),
            ],
            stream=False,
            tools=None,
            thinking=False,
            reasoning_effort="low",
            max_tokens=self.max_tokens,
            # This prompt will never be sent again, so on providers with explicit
            # caching it must not be written to the cache at the write premium.
            no_cache_write=True,
        )
        try:
            resp = await self.endpoint.chat(req)
        except Exception:
            return ""
        return (resp.content or "").strip()


def summary_message(summary: str, turns: int, tokens: int) -> Message:
    """The single message that stands in for everything that was cut.

    Sent as a user message rather than an assistant one: it is context handed
    *to* the model, and an assistant message asserting facts it never said reads
    as its own prior conclusion, which it will then defend.
    """
    return Message(role="user", content=(
        f"[Context compacted: {turns} earlier turn(s), ~{tokens:,} tokens, "
        f"replaced by these notes. The messages themselves are gone — re-read "
        f"any file you need rather than assuming it is still above.]\n\n"
        f"{summary}"))
