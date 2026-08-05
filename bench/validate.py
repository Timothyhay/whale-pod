"""Offline validation bench: does WhalePod's context design actually pay off?

Everything here runs with no network. A ``ScriptedEndpoint`` replays a fixed
12-turn session over WhalePod's *own* source tree through the real ``Agent``,
``MessageManager``, ``ContextLedger`` and ``ToolRegistry`` — so what is measured
is the shipping code path, not a model of it.

What is measured, and what the numbers mean
-------------------------------------------
The quantity a prefix cache keys on is the **longest common prefix between
consecutive requests**. Offline we can measure that exactly (it is a property of
the bytes we send), and we report it as *reusable prefix*. We cannot measure
cache *hits* offline — that depends on the server's KV cache — so nothing here
claims to. The live half of the validation (``bench/live_acceptance.py``) reads
the provider's own ``cached_tokens`` and is the only place hit rates are quoted.

Token counts come from ``whalepod.core.tokenizer.estimate_tokens`` (the same
estimator the status bar uses) and are floored to 64-token blocks, which is the
granularity DeepSeek documents for its cache. With tiktoken absent the estimator
is a char heuristic, so absolute token counts are approximate — but every
variant is measured with the same estimator on the same session, so the
*comparison* between designs is sound.

Experiments
-----------
  1. ``variants``   four context designs over the identical session:
                    as-built, ledger disabled, rolling summarization, and the
                    deleted three-zone "volatile working set" layout.
  2. ``prune``      the same session against a window small enough to force
                    pruning: what a prune costs and how the prefix recovers.
  3. ``repomap``    reusable-prefix cost of the repo map across token budgets,
                    including the unbudgeted render the budgeting replaced.
  4. ``deny``       the deny tier under ``--yes``, verified with a tripwire on
                    ``subprocess`` so a regression would have to *spawn* a
                    process to pass. Includes a positive control.

Usage
-----
    python bench/validate.py                # everything, writes bench/results/
    python bench/validate.py --only variants prune
    python bench/validate.py --no-charts
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import charts
except ImportError:                              # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import charts                                # type: ignore

try:
    from . import dsv4_encoding
except ImportError:                              # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dsv4_encoding                          # type: ignore

from whalepod.context.repo_map import build_repo_map
from whalepod.core.agent import Agent, AgentConfig
from whalepod.core.ledger import ContextLedger
from whalepod.core.messages import MessageManager
from whalepod.core.prompt import build_system_prompt, repo_map_section
from whalepod.core.tokenizer import active_tokenizer_name, estimate_tokens
from whalepod.endpoints.base import (
    ChatRequest, ChatResponse, Message, StreamingDelta, ToolCallDelta,
)
from whalepod.endpoints.vllm import VLLMEndpoint
from whalepod.tools.base import ToolResult
from whalepod.tools.registry import ToolRegistry

RESULTS = Path(__file__).resolve().parent / "results"

# Cache granularity DeepSeek documents; a partial block is not reusable.
CACHE_BLOCK = 64

# Measured on OpenRouter -> DeepInfra for deepseek-v4-flash, 2026-08-04, in
# dollars per prompt token. Cache reads are 20% of a fresh prompt token, which
# is what makes prefix stability worth engineering for rather than just tidy.
# Used as a *fallback* only: if a live_acceptance.json is present in results/,
# its server-reported /models pricing is used instead (see _resolve_prices).
PRICE_PROMPT = 0.00000009
PRICE_CACHE_READ = 0.000000018
PRICES_FROM = "hardcoded 2026-08-04 snapshot"


def _resolve_prices() -> tuple[float, float, str]:
    """Prices for the offline cost columns.

    During a large-scale live run the same numbers are priced from the
    provider's /models table; to keep the offline and online benches on the same
    footing, prefer that recorded pricing when it exists, and fall back to the
    constants above (a snapshot) otherwise. Reduces drift without adding network
    to the offline bench.
    """
    src = RESULTS / "live_acceptance.json"
    try:
        if src.is_file():
            meta = json.loads(src.read_text(encoding="utf-8")).get("meta") or {}
            p = meta.get("pricing") or {}
            if p.get("prompt") and p.get("input_cache_read"):
                return (float(p["prompt"]), float(p["input_cache_read"]),
                        f"live /models pricing in {src.name}")
    except (OSError, ValueError):
        pass
    return PRICE_PROMPT, PRICE_CACHE_READ, PRICES_FROM


# --------------------------------------------------------------- session ---
@dataclass
class Turn:
    """One user turn: a user message, batches of tool calls, then an answer."""
    user: str
    batches: list[list[tuple[str, dict]]] = field(default_factory=list)
    answer: str = "Done."


def _read(path: str, **kw) -> tuple[str, dict]:
    return ("read_file", {"path": path, **kw})


# A realistic code-reading session over this repo. Turns 4, 7, 9 and 11 re-read
# a file already in the window — that is what the ledger exists for, and it is
# the single most common way a long session wastes a large context.
SESSION: list[Turn] = [
    Turn("Where is prefix caching handled in this codebase?",
         [[("grep", {"pattern": "prefix cache", "path": "whalepod"})],
          [_read("whalepod/core/messages.py")]],
         "Two zones: a stable prefix (tools + one system message) and "
         "append-only history. See whalepod/core/messages.py:83."),
    Turn("How does the ledger stop the same file arriving twice?",
         [[_read("whalepod/core/ledger.py")]],
         "It records (path, start, end) plus (mtime_ns, size) and answers a "
         "repeat read with a pointer."),
    Turn("Walk me through the agent loop.",
         [[_read("whalepod/core/agent.py")]],
         "run_turn appends the user message, streams, executes tools, repeats."),
    Turn("Remind me what the prune thresholds are in messages.py.",
         [[_read("whalepod/core/messages.py")]],
         "prune_at=0.9, prune_to=0.5 — one expensive miss instead of many."),
    Turn("What tools does the registry expose?",
         [[_read("whalepod/tools/registry.py")]],
         "Reads, writes behind plan/commit, and repo_map_refresh."),
    Turn("Show me the endpoint abstraction.",
         [[_read("whalepod/endpoints/base.py")]],
         "Unified Message/Usage types plus an abstract Endpoint."),
    Turn("In agent.py, where exactly is the ledger consulted?",
         [[_read("whalepod/core/agent.py")]],
         "_run_read calls _ledger_hit before dispatching read_file."),
    Turn("Compare the two provider implementations.",
         [[_read("whalepod/endpoints/vllm.py"),
           _read("whalepod/endpoints/anthropic.py")]],
         "vLLM is OpenAI-shaped; Anthropic needs block conversion."),
    Turn("Does the ledger handle a file being edited under it?",
         [[_read("whalepod/core/ledger.py")]],
         "Yes — invalidate() drops entries for a written path."),
    Turn("How is the repo map budgeted?",
         [[_read("whalepod/context/repo_map.py")]],
         "Weighted max-min fair allocation over top-level components."),
    Turn("Which registry function plans a write?",
         [[_read("whalepod/tools/registry.py")]],
         "ToolRegistry.plan, then commit once approved."),
    Turn("Finally, how is config resolved?",
         [[_read("whalepod/config.py")], [("tree_view", {"path": "whalepod"})]],
         "Flags > env > project config > global config > defaults."),
]


# -------------------------------------------------------------- endpoint ---
class ScriptedEndpoint(VLLMEndpoint):
    """Replays canned responses and records every request payload.

    Subclasses the real vLLM endpoint so requests are encoded by the shipping
    code — including stripping ``reasoning`` from the wire, which is exactly the
    sort of thing an ad-hoc fake would quietly get wrong.
    """

    def __init__(self, script: list[object]):
        super().__init__("https://offline.invalid", "bench")
        self.script = list(script)
        self.step = 0
        self.requests: list[dict] = []

    async def chat(self, req: ChatRequest) -> ChatResponse:   # pragma: no cover
        raise NotImplementedError("bench only streams")

    async def stream_chat(self, req: ChatRequest) -> AsyncIterator[StreamingDelta]:
        self.requests.append(self._payload(req, stream=True))
        if self.step >= len(self.script):
            raise AssertionError(
                f"script exhausted after {self.step} requests — the agent asked "
                f"for more turns than the session defines")
        item = self.script[self.step]
        self.step += 1
        # No usage is emitted: offline we have no measured cache numbers, and
        # inventing them is precisely what this bench must not do.
        if isinstance(item, str):
            yield StreamingDelta(reasoning="(considering)")
            for chunk in _chunks(item, 40):
                yield StreamingDelta(content=chunk)
            yield StreamingDelta(finish_reason="stop")
            return
        for idx, (name, args) in enumerate(item):
            blob = json.dumps(args)
            half = len(blob) // 2
            yield StreamingDelta(tool_calls=[ToolCallDelta(
                index=idx, id=f"call_{self.step}_{idx}", name=name,
                arguments=blob[:half])])
            yield StreamingDelta(tool_calls=[ToolCallDelta(
                index=idx, id=None, name=None, arguments=blob[half:])])
        yield StreamingDelta(finish_reason="tool_calls")


def _chunks(text: str, n: int):
    for i in range(0, len(text), n):
        yield text[i:i + n]


def build_script(session: list[Turn]) -> list[object]:
    out: list[object] = []
    for turn in session:
        for batch in turn.batches:
            out.append(batch)
        out.append(turn.answer)
    return out


# ------------------------------------------------------ prefix accounting ---
def wire_text(payload: dict) -> str:
    """The request as the model's chat template lays it out.

    Prefix caching keys on the byte stream the server *fast-tokenizes*, which for
    DeepSeek V4 is not the JSON payload but its official ``encode_messages``
    output (BOS + ``<|User|>...`` / ``<|Assistant|>...`` turn delimiters, tool
    schemas on the system message, tool results merged into user messages). This
    is computed with the vendored official encoder over the OpenAI-shaped payload
    that was actually sent, so the reusable-prefix and token counts are measured
    on the same bytes the server sees — not an ad-hoc JSON dump.
    """
    messages = [dict(m) for m in payload.get("messages") or []]
    tools = payload.get("tools")
    if tools:
        # The encoder carries tool schemas on the system/developer message.
        if messages and messages[0].get("role") == "system":
            messages[0] = dict(messages[0])
            messages[0]["tools"] = tools
        else:
            messages.insert(0, {"role": "system", "content": "", "tools": tools})
    return dsv4_encoding.encode_messages(
        messages,
        thinking_mode="thinking",
        drop_thinking=True,          # WhalePod never echoes reasoning upstream
        add_default_bos_token=True,
        reasoning_effort="high",     # matches the live bench's default effort
    )


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    # Binary search on the shared prefix length: the strings are ~100k chars and
    # this runs for every request of every variant.
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def blocks(tokens: int) -> int:
    return (tokens // CACHE_BLOCK) * CACHE_BLOCK


@dataclass
class RequestMetrics:
    index: int
    prompt_tokens: int
    reusable_tokens: int
    p_prompt: float = PRICE_PROMPT
    p_cache: float = PRICE_CACHE_READ

    @property
    def fresh_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.reusable_tokens)

    @property
    def reusable_frac(self) -> float:
        return self.reusable_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def cost(self) -> float:
        return self.fresh_tokens * self.p_prompt + self.reusable_tokens * self.p_cache


def measure(payloads: list[dict]) -> list[RequestMetrics]:
    p_prompt, p_cache, _ = _resolve_prices()
    out: list[RequestMetrics] = []
    prev = ""
    for i, p in enumerate(payloads):
        text = wire_text(p)
        total = estimate_tokens(text)
        shared = common_prefix_len(prev, text)
        reusable = blocks(min(estimate_tokens(text[:shared]), total))
        out.append(RequestMetrics(index=i, prompt_tokens=total,
                                  reusable_tokens=reusable,
                                  p_prompt=p_prompt, p_cache=p_cache))
        prev = text
    return out


# --------------------------------------------------------------- variants ---
class NoLedgerAgent(Agent):
    """As-built minus the context ledger: a repeat read re-ships the file."""

    def _ledger_hit(self, args: dict) -> Optional[ToolResult]:
        return None


class RollingSummaryManager(MessageManager):
    """History is periodically collapsed into a summary (a common alternative).

    Every ``every`` user turns, everything before the current turn is replaced
    by one synthetic summary. It genuinely shrinks the context — and it rewrites
    the tail of the prefix each time it fires, so the next request can reuse
    nothing past the system message.
    """

    def __init__(self, *a, every: int = 3, **kw):
        super().__init__(*a, **kw)
        self.every = every
        self.summaries = 0
        self._turns = 0

    def add_user(self, content: str) -> Message:
        self._turns += 1
        m = super().add_user(content)
        if self._turns > self.every:
            self._collapse()
        return m

    def _collapse(self) -> None:
        if len(self.history) <= 1:
            return
        old, last = self.history[:-1], self.history[-1]
        asks = [m.content for m in old if m.role == "user"]
        files = sorted({(m.name or "") + ":" + (m.content or "")[:60]
                        for m in old if m.role == "tool"})
        body = ["[summary of the conversation so far]"]
        body += [f"- asked: {a[:120]}" for a in asks]
        body += [f"- tool output seen: {f[:120]}" for f in files]
        self.history = [Message(role="user", content="\n".join(body)), last]
        self._hist_tokens = [self._message_tokens(m) for m in self.history]
        self._hist_total = sum(self._hist_tokens)
        self.summaries += 1
        self._turns = 1


class ThreeZoneManager(MessageManager):
    """The deleted design: a volatile "working set" appended after history.

    File contents live in zone 3 and are re-rendered on every request, so the
    tail of every request differs from the last one even when nothing changed.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.volatile: dict[str, str] = {}

    def set_volatile(self, path: str, text: str) -> None:
        self.volatile[path] = text

    def ordered_messages(self) -> list[Message]:
        msgs = super().ordered_messages()
        if self.volatile:
            body = "\n\n".join(f"### {p}\n{t}" for p, t in self.volatile.items())
            msgs.append(Message(
                role="user",
                content=f"Working set (current contents)\n{body}"))
        return msgs

    def estimated_history(self) -> int:
        extra = sum(int(self._estimate(t)) for t in self.volatile.values())
        return super().estimated_history() + extra

    def estimated_total(self) -> int:
        return self.estimated_stable() + self.estimated_history()


class ThreeZoneAgent(Agent):
    """Routes read results into the working set instead of into history."""

    def _ledger_hit(self, args: dict) -> Optional[ToolResult]:
        return None                      # the working set *was* the dedup story

    def _append_result(self, call: dict, result: ToolResult) -> None:
        fn = call.get("function") or {}
        if fn.get("name") == "read_file" and result.ok:
            info = (result.meta or {}).get("read") or {}
            path = info.get("path") or "?"
            self.mm.set_volatile(path, result.output or "")
            super()._append_result(call, ToolResult(
                ok=True, output=f"# {path} loaded into the working set"))
            return
        super()._append_result(call, result)


@dataclass
class VariantRun:
    name: str
    note: str
    metrics: list[RequestMetrics]
    ledger_hits: int
    ledger_saved: int
    prunes: int
    summaries: int
    final_context: int

    @property
    def prompt_tokens(self) -> int:
        return sum(m.prompt_tokens for m in self.metrics)

    @property
    def reusable_tokens(self) -> int:
        return sum(m.reusable_tokens for m in self.metrics)

    @property
    def fresh_tokens(self) -> int:
        return sum(m.fresh_tokens for m in self.metrics)

    @property
    def reusable_frac(self) -> float:
        return self.reusable_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def cost(self) -> float:
        return sum(m.cost for m in self.metrics)

    def as_dict(self) -> dict:
        return {
            "name": self.name, "note": self.note,
            "requests": len(self.metrics),
            "prompt_tokens": self.prompt_tokens,
            "reusable_tokens": self.reusable_tokens,
            "fresh_tokens": self.fresh_tokens,
            "reusable_frac": round(self.reusable_frac, 4),
            "prompt_cost_usd": round(self.cost, 6),
            "ledger_hits": self.ledger_hits,
            "ledger_saved_tokens": self.ledger_saved,
            "prunes": self.prunes, "summaries": self.summaries,
            "final_context_tokens": self.final_context,
            "per_request": [
                {"i": m.index, "prompt_tokens": m.prompt_tokens,
                 "reusable_tokens": m.reusable_tokens,
                 "reusable_frac": round(m.reusable_frac, 4)}
                for m in self.metrics
            ],
        }


def make_prefix(mm: MessageManager, registry: ToolRegistry, index) -> None:
    mm.set_tools(registry.schemas())
    mm.set_system(build_system_prompt())
    mm.set_repo_map(repo_map_section(index.render()))


def stable_tokens(index) -> int:
    """Size of the stable prefix (tools + system prompt + repo map).

    Needed to size a window that forces pruning: the prefix is charged against
    the window but can never be pruned, so a window picked from peak *prompt*
    size alone can leave the history budget either untouched or negative.
    """
    mm = MessageManager()
    make_prefix(mm, ToolRegistry(ROOT, sandbox_mode="readonly"), index)
    return mm.estimated_stable()


def prune_forcing_window(index, peak_prompt_tokens: int) -> int:
    """A window this session must actively manage, rounded to 1k.

    Solved from the real thresholds rather than guessed: pruning fires at
    ``prune_at`` (0.9) of the window, so the window has to be small enough that
    the *history* alone crosses it. Aim for the trigger to land at ~70% of peak
    history, which prunes twice over this session and still leaves room for a
    couple of turns afterwards — a window so tight that one turn exceeds the
    target would hit the "single turn bigger than the budget" bail-out and
    measure that instead.
    """
    stable = stable_tokens(index)
    peak_history = max(1, peak_prompt_tokens - stable)
    return max(1000, int((stable + 0.70 * peak_history) / 0.9 / 1000) * 1000)


async def run_variant(name: str, note: str, *, window: int,
                      mm_cls=MessageManager, agent_cls=Agent,
                      index=None, mm_kwargs: Optional[dict] = None) -> VariantRun:
    registry = ToolRegistry(ROOT, sandbox_mode="readonly")
    registry.repo_index = index
    mm = mm_cls(window=window, **(mm_kwargs or {}))
    make_prefix(mm, registry, index)
    endpoint = ScriptedEndpoint(build_script(SESSION))
    ledger = ContextLedger()
    agent = agent_cls(endpoint, registry, mm,
                      AgentConfig(model="bench", max_iterations=6),
                      ledger=ledger)
    for turn in SESSION:
        await agent.run_turn(turn.user)
    return VariantRun(
        name=name, note=note, metrics=measure(endpoint.requests),
        ledger_hits=ledger.hits, ledger_saved=ledger.saved_tokens,
        prunes=len(mm.prune_events),
        summaries=getattr(mm, "summaries", 0),
        final_context=mm.estimated_total(),
    )


VARIANTS = [
    ("as-built", "two zones, append-only, ledger on", MessageManager, Agent, {}),
    ("no-ledger", "same layout, duplicate reads re-shipped",
     MessageManager, NoLedgerAgent, {}),
    ("rolling-summary", "history collapsed into a summary every 3 turns",
     RollingSummaryManager, Agent, {"every": 3}),
    ("three-zone", "deleted design: volatile working set after history",
     ThreeZoneManager, ThreeZoneAgent, {}),
]


async def experiment_variants(index, window: int, label: str) -> dict:
    runs = []
    for name, note, mm_cls, agent_cls, kw in VARIANTS:
        runs.append(await run_variant(name, note, window=window, mm_cls=mm_cls,
                                      agent_cls=agent_cls, index=index,
                                      mm_kwargs=kw))
    return {"window": window, "label": label,
            "runs": [r.as_dict() for r in runs],
            "_runs": runs}


# ----------------------------------------------------------------- pruning ---
async def experiment_prune(index, window: int) -> dict:
    """Same session, window small enough that pruning has to happen."""
    run = await run_variant("as-built (pruning)",
                            f"window={window:,}", window=window, index=index)
    registry = ToolRegistry(ROOT, sandbox_mode="readonly")
    registry.repo_index = index
    mm = MessageManager(window=window)
    make_prefix(mm, registry, index)
    # Re-run capturing prune positions against request index, which is what the
    # chart needs: a prune shows up as the request whose prefix collapses.
    endpoint = ScriptedEndpoint(build_script(SESSION))
    notes: list[tuple[int, str]] = []
    agent = Agent(endpoint, registry, mm,
                  AgentConfig(model="bench", max_iterations=6),
                  ledger=ContextLedger())
    agent.config.notice_sink = lambda text: (
        notes.append((len(endpoint.requests), text))
        if text.startswith("pruned") else None)
    for turn in SESSION:
        await agent.run_turn(turn.user)
    metrics = measure(endpoint.requests)
    return {
        "window": window,
        "prunes": len(mm.prune_events),
        "prune_events": [
            {"at_request": i, "detail": t} for i, t in notes],
        "events": [{"turns_dropped": e.turns_dropped,
                    "messages_dropped": e.messages_dropped,
                    "tokens_dropped": e.tokens_dropped,
                    "tokens_after": e.tokens_after}
                   for e in mm.prune_events],
        "per_request": [
            {"i": m.index, "prompt_tokens": m.prompt_tokens,
             "reusable_tokens": m.reusable_tokens,
             "reusable_frac": round(m.reusable_frac, 4)} for m in metrics],
        "reusable_frac": round(
            sum(m.reusable_tokens for m in metrics)
            / max(1, sum(m.prompt_tokens for m in metrics)), 4),
        "_metrics": metrics,
        "_run": run,
    }


# ---------------------------------------------------------------- repo map ---
def experiment_repomap() -> dict:
    from whalepod.config import Config
    cfg = Config()
    budgets = [2_000, 4_000, 6_000, 8_000, 12_000, 16_000, 24_000]
    index = build_repo_map(ROOT, max_symbols=5000,
                           languages=cfg.repo_map.languages,
                           exclude=["reference"], max_tokens=16_000)
    rows = []
    for b in budgets:
        text = index.render(max_symbols=2000, max_tokens=b)
        rows.append({"budget": b, "tokens": estimate_tokens(text),
                     "lines": text.count("\n") + 1,
                     "truncated": "map truncated" in text})
    # What budgeting replaced: the same index rendered with no token bound, and
    # with the vendored `reference/` tree included as a fresh checkout would.
    unbounded = index.render(max_symbols=2000, max_tokens=10 ** 9)
    with_vendored = build_repo_map(
        ROOT, max_symbols=2000, languages=cfg.repo_map.languages,
        exclude=[], max_tokens=10 ** 9).render(max_symbols=2000,
                                               max_tokens=10 ** 9)
    return {
        "auto_budget_1m_window": cfg.resolved_map_tokens(),
        "files_indexed": len(index.by_file),
        "symbols_indexed": index.symbol_count,
        "treesitter": index.ts_used,
        "curve": rows,
        "unbounded_tokens": estimate_tokens(unbounded),
        "unbounded_with_vendored_tokens": estimate_tokens(with_vendored),
    }


# -------------------------------------------------------------- deny tier ---
class _Tripwire:
    """Stands in for the ``subprocess`` module inside whalepod.tools.edit.

    Patching ``subprocess`` rather than ``run_command_now`` leaves *both* deny
    checks (plan time and commit time) under test: for a denied command to pass
    this experiment it would have to actually reach process spawn.
    """

    def __init__(self):
        self.spawned: list[str] = []

    def run(self, cmd, **kw):
        self.spawned.append(cmd)
        raise AssertionError(f"TRIPWIRE: a command reached subprocess: {cmd!r}")

    class TimeoutExpired(Exception):
        pass


DENIED = [
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda bs=1M",
    "echo hi && rm -rf /",
    "true; shutdown -h now",
    ":(){ :|:& };:",
    "git push --force",
    "diskpart /s script.txt",
    "format c: /q",
    "cat /etc/passwd > /dev/sda",
]


async def experiment_deny() -> dict:
    """--yes must approve edits, not arbitrary destruction.

    Runs each command through the whole agent path (plan -> auto-approve ->
    commit) with the sandbox in ``yes`` mode, which is the mode where a mistake
    would be unattended.
    """
    import whalepod.tools.edit as edit_mod
    trip = _Tripwire()
    real_subprocess = edit_mod.subprocess
    edit_mod.subprocess = trip                        # type: ignore[assignment]
    results = []
    control: dict = {}
    try:
        for cmd in DENIED:
            registry = ToolRegistry(ROOT, sandbox_mode="yes")
            endpoint = ScriptedEndpoint([[("run_command", {"cmd": cmd})], "ok"])
            mm = MessageManager(window=200_000)
            mm.set_system(build_system_prompt())
            agent = Agent(endpoint, registry, mm, AgentConfig(model="bench"),
                          ledger=ContextLedger())
            await agent.run_turn(f"please run: {cmd}")
            tool_msgs = [m.content for m in mm.history if m.role == "tool"]
            out = tool_msgs[0] if tool_msgs else ""
            results.append({
                "command": cmd,
                "refused": out.startswith("[error]"),
                "reason": out[:160],
                "plan_time_denied": registry.guard.is_denied(cmd),
            })
        # Positive control: a harmless command must reach the tripwire, else
        # "nothing spawned" would prove only that the harness is broken.
        registry = ToolRegistry(ROOT, sandbox_mode="yes")
        endpoint = ScriptedEndpoint([[("run_command", {"cmd": "echo hello"})], "ok"])
        mm = MessageManager(window=200_000)
        mm.set_system(build_system_prompt())
        agent = Agent(endpoint, registry, mm, AgentConfig(model="bench"),
                      ledger=ContextLedger())
        try:
            await agent.run_turn("please run: echo hello")
        except AssertionError:
            pass                    # the tripwire firing *is* the pass condition
        control = {"command": "echo hello",
                   "reached_subprocess": "echo hello" in trip.spawned}
    finally:
        edit_mod.subprocess = real_subprocess         # type: ignore[assignment]
    return {
        "commands_tested": len(DENIED),
        "all_refused": all(r["refused"] for r in results),
        "spawned_denied_commands": [c for c in trip.spawned if c in DENIED],
        "positive_control": control,
        "results": results,
    }


# ----------------------------------------------------------------- report ---
def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def report_variants(data: dict) -> str:
    runs: list[VariantRun] = data["_runs"]
    base = next(r for r in runs if r.name == "as-built")
    lines = [f"### Context design variants (window {data['window']:,}, "
             f"{len(base.metrics)} requests)", ""]
    head = (f"{'variant':<17} {'prompt tok':>11} {'reusable':>10} "
            f"{'reuse':>7} {'billed tok':>11} {'prompt $':>10} {'cuts':>5} "
            f"{'vs as-built':>12}")
    lines += [head, "-" * len(head)]
    for r in runs:
        delta = ("—" if r is base
                 else f"{(r.cost / base.cost - 1) * 100:+.0f}% cost")
        # "cuts" = prunes + summarizations: the times a design threw context
        # away. Printed so a "this window forces pruning" claim is checkable
        # from the table instead of taken on trust.
        lines.append(f"{r.name:<17} {r.prompt_tokens:>11,} "
                     f"{r.reusable_tokens:>10,} {pct(r.reusable_frac):>7} "
                     f"{r.fresh_tokens:>11,} {r.cost:>10.5f} "
                     f"{r.prunes + r.summaries:>5} {delta:>12}")
    lines += ["", "Reusable prefix per request (share of that request's prompt):", ""]
    # as-built last: in an overlaid plot the last series wins a contested cell,
    # and it is the line the reader is here to see.
    ordered = sorted(runs, key=lambda r: r.name == "as-built")
    lines.append(charts.ascii_series(
        {r.name: [m.reusable_frac for m in r.metrics] for r in ordered},
        height=11, ymax=1.0))
    lines += ["", "Session prompt cost, $ (cached tokens billed at 20%):", ""]
    lines.append(charts.ascii_bars([(r.name, r.cost) for r in runs],
                                   fmt=lambda v: f"${v:.5f}"))
    return "\n".join(lines)


def report_prune(data: dict) -> str:
    metrics: list[RequestMetrics] = data["_metrics"]
    lines = [f"### Pruning at a {data['window']:,}-token window", ""]
    lines.append(f"prunes: {data['prunes']}")
    for e in data["events"]:
        lines.append(f"  dropped {e['turns_dropped']} turn(s) / "
                     f"{e['messages_dropped']} message(s), "
                     f"~{e['tokens_dropped']:,} tokens -> "
                     f"{e['tokens_after']:,} in context")
    lines += ["", "Reusable prefix per request (▼ = prune):", ""]
    lines.append(charts.ascii_series(
        {"as-built @ small window": [m.reusable_frac for m in metrics]},
        height=11, ymax=1.0,
        xlabels="".join("▼" if any(i == p["at_request"] - 1
                                   for p in data["prune_events"]) else " "
                        for i in range(len(metrics)))))
    lines.append(f"\nsession reusable prefix: {pct(data['reusable_frac'])}")
    return "\n".join(lines)


def report_repomap(data: dict) -> str:
    lines = ["### Repo-map token budget", ""]
    lines.append(f"{data['files_indexed']} files, {data['symbols_indexed']} "
                 f"symbols indexed (tree-sitter: {data['treesitter']})")
    lines.append(f"auto budget for a 1M window: "
                 f"{data['auto_budget_1m_window']:,} tokens")
    lines.append(f"unbudgeted render of the same index: "
                 f"{data['unbounded_tokens']:,} tokens")
    lines.append(f"unbudgeted, vendored reference/ included (a fresh checkout "
                 f"before excludes): {data['unbounded_with_vendored_tokens']:,} "
                 f"tokens")
    lines += ["", "Rendered size vs configured budget:", ""]
    lines.append(charts.ascii_bars(
        [(f"{r['budget']:>6,}", r["tokens"]) for r in data["curve"]],
        fmt=lambda v: f"{v:,.0f} tok"))
    return "\n".join(lines)


def report_deny(data: dict) -> str:
    lines = ["### Deny tier under --yes", ""]
    ctrl = data.get("positive_control") or {}
    lines.append(f"commands tested: {data['commands_tested']}")
    lines.append(f"all refused: {data['all_refused']}")
    lines.append(f"denied commands that reached subprocess: "
                 f"{len(data['spawned_denied_commands'])}")
    lines.append(f"positive control (`{ctrl.get('command')}`) reached "
                 f"subprocess: {ctrl.get('reached_subprocess')}")
    lines.append("")
    for r in data["results"]:
        mark = "refused" if r["refused"] else "!! RAN"
        lines.append(f"  [{mark}] {r['command']}")
    return "\n".join(lines)


# ----------------------------------------------------------------- charts ---
def write_charts(out: dict) -> list[Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if "variants" in out:
        data = out["variants"]
        runs: list[VariantRun] = data["_runs"]
        # as-built drawn last so it sits on top where the lines cross, matching
        # the ASCII plot in the text report.
        ordered = sorted(runs, key=lambda r: r.name == "as-built")
        written.append(charts.svg_lines(
            RESULTS / "reusable_prefix.svg",
            "Reusable prefix per request, by context design",
            {r.name: [m.reusable_frac for m in r.metrics] for r in ordered},
            xlabel="request # (12-turn session over WhalePod's own source)",
            ylabel="share of prompt reusable", ymax=1.0,
            subtitle=f"window {data['window']:,} tokens · offline, measured on "
                     f"the bytes actually sent",
            xtick_labels=[str(i + 1) for i in range(len(runs[0].metrics))]))
        written.append(charts.svg_stacked(
            RESULTS / "prompt_tokens_split.svg",
            "Session prompt tokens: reusable prefix vs freshly billed",
            [r.name for r in runs],
            [("reusable prefix", [r.reusable_tokens for r in runs]),
             ("billed fresh", [r.fresh_tokens for r in runs])],
            value_fmt=lambda v: f"{v/1000:,.0f}k", ylabel="prompt tokens",
            subtitle="taller is worse; the dark part is what you pay full price for"))
        written.append(charts.svg_bars(
            RESULTS / "prompt_cost.svg",
            "Session prompt cost (cached tokens billed at 20%)",
            [(r.name, r.cost) for r in runs],
            value_fmt=lambda v: f"${v:.4f}", ylabel="USD",
            highlight="as-built",
            subtitle="deepseek-v4-flash via DeepInfra pricing, prompt side only"))

    if "variants_small" in out:
        data = out["variants_small"]
        runs = data["_runs"]
        ordered = sorted(runs, key=lambda r: r.name == "as-built")
        written.append(charts.svg_lines(
            RESULTS / "reusable_prefix_small_window.svg",
            "Reusable prefix when every design has to manage the window",
            {r.name: [m.reusable_frac for m in r.metrics] for r in ordered},
            xlabel="request #", ylabel="share of prompt reusable", ymax=1.0,
            subtitle=f"window {data['window']:,} tokens · the comparison the 1M "
                     f"run cannot make, because there nothing ever prunes",
            xtick_labels=[str(i + 1) for i in range(len(runs[0].metrics))]))
        written.append(charts.svg_bars(
            RESULTS / "prompt_cost_small_window.svg",
            f"Session prompt cost at a {data['window']:,}-token window",
            [(r.name, r.cost) for r in runs],
            value_fmt=lambda v: f"${v:.4f}", ylabel="USD",
            highlight="as-built",
            subtitle="cached tokens billed at 20%; prompt side only"))

    if "prune" in out:
        data = out["prune"]
        metrics: list[RequestMetrics] = data["_metrics"]
        markers = [(p["at_request"] - 1, "prune") for p in data["prune_events"]]
        written.append(charts.svg_lines(
            RESULTS / "prune_recovery.svg",
            f"Cost of a prune, and how the prefix recovers",
            {"reusable prefix": [m.reusable_frac for m in metrics]},
            xlabel="request #", ylabel="share of prompt reusable", ymax=1.0,
            markers=markers,
            subtitle=f"window forced to {data['window']:,} tokens so pruning "
                     f"must happen ({data['prunes']} prune(s))",
            xtick_labels=[str(i + 1) for i in range(len(metrics))]))

    if "repomap" in out:
        data = out["repomap"]
        rows = [(f"{r['budget']//1000}k", float(r["tokens"]))
                for r in data["curve"]]
        rows.append(("no budget", float(data["unbounded_tokens"])))
        rows.append(("no budget\n+vendored", float(
            data["unbounded_with_vendored_tokens"])))
        written.append(charts.svg_bars(
            RESULTS / "repo_map_budget.svg",
            "Repo map: rendered size vs configured token budget",
            rows, value_fmt=lambda v: f"{v/1000:,.1f}k",
            ylabel="tokens in the stable prefix",
            subtitle="the map is charged to every request in the session"))
    return written


# ------------------------------------------------------------------- main ---
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["variants", "variants-small", "prune", "repomap",
                             "deny"],
                    help="run a subset of experiments")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--window", type=int, default=1_000_000)
    args = ap.parse_args(argv)
    want = set(args.only or ["variants", "variants-small", "prune", "repomap",
                             "deny"])

    from whalepod.config import Config
    cfg = Config()
    index = build_repo_map(ROOT, max_symbols=cfg.repo_map.max_symbols,
                           languages=cfg.repo_map.languages,
                           exclude=["reference"],
                           max_tokens=cfg.resolved_map_tokens())

    out: dict = {"meta": {
        "root": str(ROOT), "cache_block": CACHE_BLOCK,
        "price_prompt_usd_per_token": PRICE_PROMPT,
        "price_cache_read_usd_per_token": PRICE_CACHE_READ,
        "prices_from": _resolve_prices()[2],
        "tokenizer": active_tokenizer_name(),
        "turns": len(SESSION),
    }}
    sections: list[str] = []

    if "variants" in want:
        out["variants"] = asyncio.run(experiment_variants(
            index, args.window, "1M window (nothing prunes)"))
        sections.append(report_variants(out["variants"]))

    if "variants-small" in want:
        # A window that all four designs must actively manage, so the
        # comparison is not rigged in favour of "never prune".
        base = out.get("variants") or asyncio.run(experiment_variants(
            index, args.window, "sizing run"))
        peak = max(m.prompt_tokens for m in base["_runs"][0].metrics)
        window = prune_forcing_window(index, peak)
        out["variants_small"] = asyncio.run(experiment_variants(
            index, window, f"{window:,} window (as-built has to prune)"))
        sections.append(report_variants(out["variants_small"]))
        out["prune_window"] = window

    if "prune" in want:
        window = out.get("prune_window")
        if window is None:
            base = out.get("variants") or asyncio.run(experiment_variants(
                index, args.window, "sizing run"))
            peak = max(m.prompt_tokens for m in base["_runs"][0].metrics)
            window = prune_forcing_window(index, peak)
        out["prune"] = asyncio.run(experiment_prune(index, window))
        sections.append(report_prune(out["prune"]))

    if "repomap" in want:
        out["repomap"] = experiment_repomap()
        sections.append(report_repomap(out["repomap"]))

    if "deny" in want:
        out["deny"] = asyncio.run(experiment_deny())
        sections.append(report_deny(out["deny"]))

    text = "\n\n".join(sections)
    print(text)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "validation.txt").write_text(text + "\n", encoding="utf-8")
    serializable = {k: _strip_private(v) for k, v in out.items()}
    (RESULTS / "validation.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8")
    if not args.no_charts:
        for p in write_charts(out):
            print(f"\nchart: {p.relative_to(ROOT)}")
    print(f"\nwrote {(RESULTS / 'validation.json').relative_to(ROOT)}")
    return 0


def _strip_private(v):
    if isinstance(v, dict):
        return {k: _strip_private(x) for k, x in v.items()
                if not str(k).startswith("_")}
    if isinstance(v, list):
        return [_strip_private(x) for x in v]
    return v


def _has_tiktoken() -> bool:
    try:
        import tiktoken  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
