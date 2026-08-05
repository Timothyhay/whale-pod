"""Live acceptance test: does the context design pay off against a real server?

``bench/validate.py`` measures the *reusable prefix* — a property of the bytes we
send, which is all that can be known offline. It deliberately never claims a
cache hit. This script closes that gap: it drives the real ``Agent`` through a
real multi-turn session against a real provider and records what the **server**
says it cached, request by request.

What it checks
--------------
  1. ``cached_tokens`` climbs as the session grows, and stays high — that is the
     append-only two-zone layout actually working, not just looking tidy.
  2. The measured hit rate tracks the offline *predicted* reusable prefix. If
     the two disagree badly, either our model of the wire format is wrong or the
     provider is not caching what we think it is; both are worth knowing.
  3. A prune collapses the hit rate on exactly the request after it, and the
     rate recovers afterwards. That is the one expensive event the design
     accepts, and it should be visible and rare rather than continuous.
  4. Provider affinity holds. Unpinned, an aggregator load-balances across
     providers and each request lands in a cold KV cache — measured at ~0%
     cached. ``--no-pin`` reproduces that on purpose.

Requesting usage is not optional
--------------------------------
Streaming responses carry no usage block unless ``stream_options.include_usage``
is set (the endpoint does this), and OpenRouter only reports its own ``cost``
when ``usage: {"include": true}`` is in the body (this script adds that). Where
the provider reports cost we use its number; otherwise we price the tokens from
the live ``/models`` pricing table. Nothing here is estimated silently.

Usage
-----
    $env:OPENROUTER_API_KEY = "sk-or-v1-..."      # never stored in the repo
    python bench/live_acceptance.py               # main run + prune run
    python bench/live_acceptance.py --only main
    python bench/live_acceptance.py --only pin-check   # affinity A/B, 2 requests
    python bench/live_acceptance.py --turns 6 --effort medium
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import charts, validate
except ImportError:                              # run as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import charts                                # type: ignore
    import validate                              # type: ignore

import httpx

from whalepod.context.repo_map import build_repo_map
from whalepod.core.agent import Agent, AgentConfig
from whalepod.core.ledger import ContextLedger
from whalepod.core.messages import MessageManager
from whalepod.endpoints.base import ChatRequest, EndpointError, StreamingDelta
from whalepod.endpoints.openai import OpenAIChatEndpoint
from whalepod.tools.registry import ToolRegistry

RESULTS = validate.RESULTS
KEY_ENVS = ("WHALEPOD_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
DEFAULT_BASE = "https://openrouter.ai/api"
DEFAULT_MODEL = "~deepseek/deepseek-v4-flash-latest"
DEFAULT_PROVIDER = "DeepInfra"


def api_base(url: str) -> str:
    """Base URL in the form the endpoint wants: no trailing ``/v1``.

    ``VLLMEndpoint`` appends ``/v1/chat/completions`` itself, so passing
    ``https://openrouter.ai/api/v1`` (which is what the provider's own curl
    examples show) produced ``/api/v1/v1/chat/completions`` and a 404 page.
    Both spellings are accepted here rather than left as a trap.
    """
    u = (url or "").rstrip("/")
    return u[:-3].rstrip("/") if u.endswith("/v1") else u


def api_key() -> str:
    for env in KEY_ENVS:
        v = os.environ.get(env)
        if v:
            return v.strip()
    raise SystemExit(
        f"no API key: set one of {', '.join(KEY_ENVS)}. The key is passed by "
        f"environment on purpose — it must not land in a repo file.")


# ------------------------------------------------------------- endpoint ---
class RecordingEndpoint(OpenAIChatEndpoint):
    """The real endpoint, plus a record of what went out and what came back.

    Subclassed rather than reimplemented so the measurement is of the shipping
    transport — same payload builder, same SSE parsing, same usage
    normalization. Two hooks are enough: ``_payload`` sees every request and
    ``_parse_stream_chunk`` sees the final usage-only chunk.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.records: list[dict] = []
        self._current: Optional[dict] = None

    def _payload(self, req: ChatRequest, stream: Optional[bool] = None) -> dict:
        p = super()._payload(req, stream)
        if self._current is not None and self._current["payload"] is None:
            # Deep-copied: the agent mutates its message list between requests
            # and the prefix comparison needs the bytes as they were sent.
            self._current["payload"] = json.loads(json.dumps(p))
        return p

    def _parse_stream_chunk(self, chunk: dict):
        raw = chunk.get("usage")
        if raw and self._current is not None:
            self._current["raw_usage"] = raw
        return super()._parse_stream_chunk(chunk)

    async def stream_chat(self, req: ChatRequest) -> AsyncIterator[StreamingDelta]:
        self._current = {"payload": None, "raw_usage": None, "t0": None, "t1": None}
        self.records.append(self._current)
        t0 = time.monotonic()
        self._current["t0"] = t0
        try:
            async for d in super().stream_chat(req):
                yield d
        finally:
            self._current["t1"] = time.monotonic()

    @property
    def payloads(self) -> list[dict]:
        return [r["payload"] or {} for r in self.records]


def build_endpoint(model: str, base_url: str, provider: Optional[str],
                   timeout: float) -> RecordingEndpoint:
    extra: dict = {"usage": {"include": True}}
    if provider:
        # The pin is the precondition for everything else measured here: a
        # prefix cache lives in one server's memory, so a router free to pick a
        # provider per request turns every request into a cold miss.
        extra["provider"] = {"order": [provider], "allow_fallbacks": False}
    return RecordingEndpoint(api_base(base_url), api_key(), extra_body=extra,
                             timeout=timeout)


async def fetch_pricing(base_url: str, model: str) -> dict:
    """Live per-token prices for the model, or {} if the list is unavailable.

    Used only to price tokens when the provider does not report a cost itself,
    so a pricing outage degrades one column instead of failing the run.
    """
    want = model.lstrip("~").split(":")[0]
    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as c:
            r = await c.get(f"{api_base(base_url)}/v1/models")
            r.raise_for_status()
            for m in r.json().get("data") or []:
                if str(m.get("id", "")).lstrip("~").split(":")[0] == want:
                    return {k: float(v) for k, v in
                            (m.get("pricing") or {}).items()
                            if _is_number(v)}
    except Exception as e:                       # noqa: BLE001 - advisory only
        print(f"(pricing lookup failed: {e})", file=sys.stderr)
    return {}


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# -------------------------------------------------------------- session ---
@dataclass
class LiveRequest:
    index: int
    turn: int
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    miss_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: Optional[float] = None
    predicted_reusable: int = 0        # offline: our own prefix measurement
    predicted_prompt: int = 0         # offline: our own token estimate
    pruned_before: bool = False
    idle_s: Optional[float] = None    # wall-clock gap since the previous request
    duration_s: Optional[float] = None

    @property
    def hit_rate(self) -> Optional[float]:
        if not self.prompt_tokens:
            return None
        return self.cached_tokens / self.prompt_tokens

    @property
    def predicted_frac(self) -> float:
        return (self.predicted_reusable / self.predicted_prompt
                if self.predicted_prompt else 0.0)

    @property
    def accounting_consistent(self) -> bool:
        """Server's own cache split: hit + miss should reconcile to prompt."""
        if not self.prompt_tokens:
            return True
        return self.cached_tokens + self.miss_tokens == self.prompt_tokens

    def as_dict(self) -> dict:
        return {
            "i": self.index, "turn": self.turn,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "miss_tokens": self.miss_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "hit_rate": round(self.hit_rate, 4) if self.hit_rate is not None else None,
            "predicted_reusable_frac": round(self.predicted_frac, 4),
            "cost_usd": self.cost_usd,
            "pruned_before": self.pruned_before,
            "idle_s": (round(self.idle_s, 1)
                       if self.idle_s is not None else None),
            "duration_s": (round(self.duration_s, 1)
                           if self.duration_s is not None else None),
            "accounting_consistent": self.accounting_consistent,
        }


@dataclass
class LiveRun:
    label: str
    model: str
    provider: Optional[str]
    window: int
    requests: list[LiveRequest] = field(default_factory=list)
    prunes: list[dict] = field(default_factory=list)
    ledger_hits: int = 0
    ledger_saved: int = 0
    errors: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    pricing: dict = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.requests)

    @property
    def cached_tokens(self) -> int:
        return sum(r.cached_tokens for r in self.requests)

    @property
    def hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    @property
    def cost(self) -> float:
        """Total billed for the session — prompt *and* completion tokens."""
        return sum(r.cost_usd or 0.0 for r in self.requests)

    # The headline claim of this project is about the *prompt* side: caching does
    # nothing for output tokens. Quoting the total against a prompt-only "if
    # nothing had been cached" baseline mixes two things and gets the saving
    # wrong, so both numbers are computed here from the same price list rather
    # than by hand afterwards.
    @property
    def prompt_cost(self) -> Optional[float]:
        """Prompt tokens as actually billed (cached ones at the cache rate)."""
        if not self.pricing:
            return None
        p_in = self.pricing.get("prompt", 0.0)
        p_cache = self.pricing.get("input_cache_read", p_in)
        fresh = self.prompt_tokens - self.cached_tokens
        return fresh * p_in + self.cached_tokens * p_cache

    @property
    def prompt_cost_uncached(self) -> Optional[float]:
        """What the same prompt tokens would have cost with no cache at all."""
        if not self.pricing:
            return None
        return self.prompt_tokens * self.pricing.get("prompt", 0.0)

    @property
    def prompt_saving(self) -> Optional[float]:
        full = self.prompt_cost_uncached
        if not full:
            return None
        return 1.0 - (self.prompt_cost or 0.0) / full

    def as_dict(self) -> dict:
        return {
            "label": self.label, "model": self.model, "provider": self.provider,
            "window": self.window, "requests": len(self.requests),
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": sum(r.completion_tokens for r in self.requests),
            "reasoning_tokens": sum(r.reasoning_tokens for r in self.requests),
            "session_hit_rate": round(self.hit_rate, 4),
            "cost_usd": round(self.cost, 6),
            "prompt_cost_usd": (round(self.prompt_cost, 6)
                                if self.prompt_cost is not None else None),
            "prompt_cost_uncached_usd": (round(self.prompt_cost_uncached, 6)
                                         if self.prompt_cost_uncached is not None
                                         else None),
            "prompt_saving": (round(self.prompt_saving, 4)
                              if self.prompt_saving is not None else None),
            "prunes": self.prunes,
            "ledger_hits": self.ledger_hits,
            "ledger_saved_tokens": self.ledger_saved,
            "errors": self.errors,
            "per_request": [r.as_dict() for r in self.requests],
        }


def build_index():
    from whalepod.config import Config
    cfg = Config()
    return build_repo_map(ROOT, max_symbols=cfg.repo_map.max_symbols,
                          languages=cfg.repo_map.languages,
                          exclude=["reference"],
                          max_tokens=cfg.resolved_map_tokens())


async def run_session(label: str, *, index, window: int, turns: int,
                      model: str, base_url: str, provider: Optional[str],
                      effort: str, timeout: float,
                      pricing: dict) -> LiveRun:
    """Drive the real agent through the scripted user turns, live."""
    registry = ToolRegistry(ROOT, sandbox_mode="readonly")
    registry.repo_index = index
    mm = MessageManager(window=window)
    validate.make_prefix(mm, registry, index)
    endpoint = build_endpoint(model, base_url, provider, timeout)
    ledger = ContextLedger()
    run = LiveRun(label=label, model=model, provider=provider, window=window,
                  pricing=pricing)

    # Prune notices are the only ones we need positionally: a prune's cost shows
    # up on the *next* request, so the request index at notice time is the mark.
    prune_marks: list[int] = []

    def notices(text: str) -> None:
        if text.startswith("pruned"):
            prune_marks.append(len(endpoint.records))
            run.prunes.append({"at_request": len(endpoint.records),
                               "detail": text})
        print(f"    · {text}")

    agent = Agent(endpoint, registry, mm,
                  AgentConfig(model=model, reasoning_effort=effort,
                              max_iterations=8, notice_sink=notices),
                  ledger=ledger)

    turn_of_request: dict[int, int] = {}
    for t, turn in enumerate(validate.SESSION[:turns], start=1):
        before = len(endpoint.records)
        print(f"  turn {t}/{turns}: {turn.user}")
        try:
            answer = await agent.run_turn(turn.user)
            run.answers.append(answer or "")
        except EndpointError as e:
            run.errors.append(f"turn {t}: {e}")
            print(f"    !! {e}")
            break
        finally:
            for i in range(before, len(endpoint.records)):
                turn_of_request[i] = t
    await endpoint.aclose()

    # Offline prediction over the same bytes, for the agreement check.
    predicted = validate.measure(endpoint.payloads)
    prev_end: Optional[float] = None
    for i, rec in enumerate(endpoint.records):
        raw = rec.get("raw_usage") or {}
        pd = raw.get("prompt_tokens_details") or {}
        cd = raw.get("completion_tokens_details") or {}
        cached = int(raw.get("prompt_cache_hit_tokens")
                     or pd.get("cached_tokens") or 0)
        miss = int(raw.get("prompt_cache_miss_tokens")
                   or max(0, int(raw.get("prompt_tokens") or 0) - cached)
                   or 0)
        prompt = int(raw.get("prompt_tokens") or 0)
        completion = int(raw.get("completion_tokens") or 0)
        t0, t1 = rec.get("t0"), rec.get("t1")
        idle = (t0 - prev_end) if (t0 is not None and prev_end is not None) else None
        if t0 is not None:
            prev_end = t1 if t1 is not None else t0
        req = LiveRequest(
            index=i, turn=turn_of_request.get(i, 0),
            prompt_tokens=prompt, cached_tokens=cached,
            cache_write_tokens=int(pd.get("cache_write_tokens") or 0),
            miss_tokens=miss,
            completion_tokens=completion,
            reasoning_tokens=int(cd.get("reasoning_tokens") or 0),
            cost_usd=_cost(raw, prompt, cached, completion, pricing),
            predicted_reusable=predicted[i].reusable_tokens if i < len(predicted) else 0,
            predicted_prompt=predicted[i].prompt_tokens if i < len(predicted) else 0,
            pruned_before=(i in prune_marks),
            idle_s=idle,
            duration_s=(t1 - t0) if (t0 is not None and t1 is not None) else None,
        )
        run.requests.append(req)
    run.ledger_hits = ledger.hits
    run.ledger_saved = ledger.saved_tokens
    return run


def _cost(raw: dict, prompt: int, cached: int, completion: int,
          pricing: dict) -> Optional[float]:
    """The provider's own cost if it reported one, else priced from /models."""
    if _is_number(raw.get("cost")):
        return float(raw["cost"])
    if not pricing:
        return None
    p_in = pricing.get("prompt", 0.0)
    p_out = pricing.get("completion", 0.0)
    p_cache = pricing.get("input_cache_read", p_in)
    fresh = max(0, prompt - cached)
    return fresh * p_in + cached * p_cache + completion * p_out


async def pin_check(*, index, model: str, base_url: str, provider: str,
                    effort: str, timeout: float, pricing: dict) -> dict:
    """Two requests each, pinned and unpinned: affinity as an A/B.

    The claim being tested is uncomfortable enough to be worth its own cheap
    experiment — that a *router* setting, nothing to do with our context layout,
    decides whether any of this works at all.

    Two precautions, both learned by getting it wrong first: each arm gets a
    **unique prefix**, and the nonce goes at the very *front* of the system
    message. Without them the arms share a prefix, and whichever arm ran second
    scored ~98% on its first request off the cache the first arm had just
    written — which reads as "affinity does not matter" when in fact the
    measurement was contaminated. The unpinned arm also runs first, so any
    leakage would understate the effect rather than manufacture it.
    """
    run_id = uuid.uuid4().hex[:8]
    out: dict = {"model": model, "provider": provider, "run_id": run_id,
                 "arms": []}
    for arm, pin in (("unpinned", None), ("pinned", provider)):
        registry = ToolRegistry(ROOT, sandbox_mode="readonly")
        registry.repo_index = index
        mm = MessageManager()
        validate.make_prefix(mm, registry, index)
        # Front of the prefix, so the two arms share nothing cacheable.
        mm.set_system(f"[affinity check {run_id}/{arm}]\n\n{mm.system_prompt}")
        endpoint = build_endpoint(model, base_url, pin, timeout)
        agent = Agent(endpoint, registry, mm,
                      AgentConfig(model=model, reasoning_effort=effort,
                                  max_iterations=1))
        try:
            # Two turns: the second one is the one that can hit, because it is
            # the first request with a long prefix already seen by the server.
            for text in ("Reply with the single word: ready.",
                         "Reply with the single word: again."):
                await agent.run_turn(text)
        except EndpointError as e:
            out["arms"].append({"arm": arm, "error": str(e)})
            await endpoint.aclose()
            continue
        await endpoint.aclose()
        reqs = []
        for rec in endpoint.records:
            raw = rec.get("raw_usage") or {}
            pd = raw.get("prompt_tokens_details") or {}
            prompt = int(raw.get("prompt_tokens") or 0)
            cached = int(raw.get("prompt_cache_hit_tokens")
                         or pd.get("cached_tokens") or 0)
            reqs.append({"prompt_tokens": prompt, "cached_tokens": cached,
                         "hit_rate": round(cached / prompt, 4) if prompt else None,
                         "cost_usd": _cost(raw, prompt, cached,
                                           int(raw.get("completion_tokens") or 0),
                                           pricing)})
        last = reqs[-1] if reqs else {}
        out["arms"].append({"arm": arm, "requests": reqs,
                            "second_request_hit_rate": last.get("hit_rate")})
    return out


# --------------------------------------------------------------- report ---
def pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def report_run(run: LiveRun) -> str:
    lines = [f"### {run.label}", "",
             f"model {run.model} · provider "
             f"{run.provider or 'router default (unpinned)'} · window "
             f"{run.window:,} · {len(run.requests)} requests"]
    if run.errors:
        lines += [""] + [f"!! {e}" for e in run.errors]
    lines += ["", "Per request, as reported by the server:", ""]
    head = (f"{'#':>3} {'turn':>4} {'prompt':>8} {'cached':>8} {'hit':>7} "
            f"{'predicted':>9} {'reason tok':>10} {'idle':>7} {'cost $':>9}")
    lines += [head, "-" * len(head)]
    for r in run.requests:
        mark = " <- after prune" if r.pruned_before else ""
        cost = "—" if r.cost_usd is None else f"{r.cost_usd:.6f}"
        idle = "—" if r.idle_s is None else f"{r.idle_s:.0f}s"
        lines.append(f"{r.index + 1:>3} {r.turn:>4} {r.prompt_tokens:>8,} "
                     f"{r.cached_tokens:>8,} {pct(r.hit_rate):>7} "
                     f"{pct(r.predicted_frac):>9} {r.reasoning_tokens:>10,} "
                     f"{idle:>7} {cost:>9}{mark}")
    lines += ["", "Measured cache hit rate vs offline-predicted reusable prefix:", ""]
    lines.append(charts.ascii_series(
        {"predicted (offline)": [r.predicted_frac for r in run.requests],
         "measured (server)": [(r.hit_rate or 0.0) for r in run.requests]},
        height=11, ymax=1.0,
        xlabels="".join("^" if r.pruned_before else " " for r in run.requests)))

    fresh = run.prompt_tokens - run.cached_tokens
    lines += ["",
              f"session prompt tokens: {run.prompt_tokens:,} "
              f"({run.cached_tokens:,} cached, {fresh:,} billed fresh)",
              f"session hit rate: {pct(run.hit_rate)}",
              f"reasoning tokens: "
              f"{sum(r.reasoning_tokens for r in run.requests):,}"]
    # Prompt cost is quoted on its own because caching does nothing for output
    # tokens: comparing a prompt+completion total against a prompt-only
    # "uncached" baseline understates the saving and invites the objection.
    if run.prompt_cost is not None:
        lines.append(f"prompt cost: ${run.prompt_cost:.5f} billed vs "
                     f"${run.prompt_cost_uncached:.5f} with no cache "
                     f"({pct(run.prompt_saving)} saved)")
        lines.append(f"total billed (prompt + completion): ${run.cost:.5f}")
    elif run.cost:
        lines.append(f"total billed (prompt + completion): ${run.cost:.5f} "
                     f"(no price list: cannot split prompt from completion)")
    else:
        lines.append("cost: not reported")
    lines.append(f"ledger: {run.ledger_hits} duplicate read(s) avoided "
                 f"(~{run.ledger_saved:,} tokens never sent)")
    early, late = _early_late(run)
    if early is not None:
        lines.append(f"hit rate, first 3 requests: {pct(early)} -> "
                     f"last 3 requests: {pct(late)}")
    agree = _agreement(run)
    if agree is not None:
        lines.append(f"offline prediction vs server: mean absolute error "
                     f"{agree * 100:.1f} points")
    for p in run.prunes:
        lines.append(f"prune before request {p['at_request'] + 1}: {p['detail']}")
    _append_eviction_note(run, lines)
    _append_consistency_note(run, lines)
    return "\n".join(lines)


def _append_eviction_note(run: LiveRun, lines: list[str]) -> None:
    """Quantify the two plausible causes of a hit-rate collapse:
    the provider evicting the cache (idle time) vs. our own byte prefix
    changing (prune). Requests with a short idle gap and no prune that still
    miss are the provider-side events the offline model cannot predict.
    """
    collapses = [r for r in run.requests
                 if r.hit_rate is not None and r.hit_rate < 0.5
                 and not r.pruned_before and r.idle_s is not None]
    if not collapses:
        return
    worst = sorted(collapses, key=lambda r: r.idle_s, reverse=True)[0]
    lines.append(
        f"hit-rate collapses without a prune: {len(collapses)} request(s), "
        f"largest idle gap before one: {worst.idle_s:.0f}s "
        f"(request #{worst.index + 1}, hit {pct(worst.hit_rate)}) "
        f"— idle time is the observable proxy for provider KV-cache eviction.")


def _append_consistency_note(run: LiveRun, lines: list[str]) -> None:
    """Server accounting should satisfy hit + miss == prompt_tokens. A broken
    split is worth knowing about: it means our hit-rate column is built on
    numbers the provider itself does not reconcile.
    """
    bad = [r for r in run.requests if not r.accounting_consistent]
    if not bad:
        return
    lines.append(
        f"server cache split inconsistent (hit+miss != prompt): "
        f"{len(bad)} request(s), e.g. request #{bad[0].index + 1} "
        f"prompt={bad[0].prompt_tokens} hit={bad[0].cached_tokens} "
        f"miss={bad[0].miss_tokens}")


def _early_late(run: LiveRun):
    rates = [r.hit_rate for r in run.requests if r.hit_rate is not None]
    if len(rates) < 6:
        return None, None
    return (sum(rates[:3]) / 3, sum(rates[-3:]) / 3)


def _agreement(run: LiveRun) -> Optional[float]:
    pairs = [(r.predicted_frac, r.hit_rate) for r in run.requests
             if r.hit_rate is not None]
    if not pairs:
        return None
    return sum(abs(p - m) for p, m in pairs) / len(pairs)


def _summarize(vals: list[float]) -> Optional[dict]:
    """Median / P10 / P90 of a list, or None when empty."""
    if not vals:
        return None
    s = sorted(vals)
    return {name: round(s[int(round(p * (len(s) - 1)))], 4)
            for name, p in {"p10": 0.10, "median": 0.50, "p90": 0.90}.items()}


def aggregate_runs(runs: list[LiveRun]) -> dict:
    """Collapse repeated live sessions into per-request and session aggregates.

    A single 29-request session is a point sample of the provider's cache
    state; running it N times and reporting median/P10/P90 turns "we hit 88%"
    into a distribution. Shared prefix bytes are deterministic, so any spread is
    server-side (eviction, load) rather than our context layout.
    """
    by_index: dict[int, list[float]] = {}
    hit_rates: list[float] = []
    costs: list[float] = []
    prompt_costs: list[float] = []
    prompt_costs_unc: list[float] = []
    for run in runs:
        if not run.requests:
            continue
        hit_rates.append(run.hit_rate)
        costs.append(run.cost)
        if run.prompt_cost is not None:
            prompt_costs.append(run.prompt_cost)
        if run.prompt_cost_uncached is not None:
            prompt_costs_unc.append(run.prompt_cost_uncached)
        for r in run.requests:
            if r.hit_rate is not None:
                by_index.setdefault(r.index, []).append(r.hit_rate)
    return {
        "sessions": len(runs),
        "session_hit_rate": _summarize(hit_rates),
        "session_cost_usd": _summarize(costs),
        "session_prompt_cost_usd": _summarize(prompt_costs),
        "session_prompt_cost_uncached_usd": _summarize(prompt_costs_unc),
        "per_request": [
            {"request": i + 1, "n": len(by_index[i]), **_summarize(by_index[i])}
            for i in sorted(by_index)
        ],
    }


def report_aggregate(data: dict) -> str:
    lines = [f"### Aggregate over {data['sessions']} repeated sessions", ""]
    def row(name, keys, fmt):
        d = data.get(keys)
        if d is None:
            return None
        return f"{name:<28} p10={fmt(d['p10'])}  med={fmt(d['median'])}  p90={fmt(d['p90'])}"
    for line in filter(None, [
            row("session hit rate", "session_hit_rate",
                lambda v: f"{v * 100:.1f}%"),
            row("session prompt cost $", "session_prompt_cost_usd",
                lambda v: f"${v:.4f}"),
            row("session prompt cost, uncached $",
                "session_prompt_cost_uncached_usd", lambda v: f"${v:.4f}"),
            row("session total billed $", "session_cost_usd",
                lambda v: f"${v:.4f}"),
    ]):
        lines.append(line)
    lines += ["", "Per-request hit rate across runs (server-measured):", ""]
    lines.append(charts.ascii_bars(
        [(f"req {p['request']}", (p.get("median") or 0.0) * 100)
         for p in data["per_request"]],
        fmt=lambda v: f"{v:.0f}%", vmax=100.0))
    return "\n".join(lines)


def report_pin(data: dict) -> str:
    lines = ["### Provider affinity A/B", "",
             "Same prefix, same model; the only difference is whether the "
             "router was allowed to choose.", ""]
    head = f"{'arm':<10} {'req':>4} {'prompt':>8} {'cached':>8} {'hit':>7}"
    lines += [head, "-" * len(head)]
    for arm in data["arms"]:
        if arm.get("error"):
            lines.append(f"{arm['arm']:<10} error: {arm['error']}")
            continue
        for i, r in enumerate(arm["requests"], start=1):
            lines.append(f"{arm['arm']:<10} {i:>4} {r['prompt_tokens']:>8,} "
                         f"{r['cached_tokens']:>8,} {pct(r['hit_rate']):>7}")
    return "\n".join(lines)


def write_charts(out: dict) -> list[Path]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for key, fname, title in (
            ("main", "live_hit_rate.svg",
             "Measured prefix-cache hit rate, live"),
            ("prune", "live_prune.svg",
             "Live: what a prune costs, as the server measures it")):
        run: Optional[LiveRun] = out.get(f"_{key}")
        if run is None or not run.requests:
            continue
        markers = [(r.index, "prune") for r in run.requests if r.pruned_before]
        written.append(charts.svg_lines(
            RESULTS / fname, title,
            {"predicted reusable prefix (offline)":
                [r.predicted_frac for r in run.requests],
             "cached_tokens / prompt_tokens (server)":
                [(r.hit_rate or 0.0) for r in run.requests]},
            xlabel="request #", ylabel="share of prompt", ymax=1.0,
            markers=markers,
            subtitle=f"{run.model} pinned to "
                     f"{run.provider or 'no provider'} · window "
                     f"{run.window:,} · session hit rate {pct(run.hit_rate)}",
            xtick_labels=[str(r.index + 1) for r in run.requests]))
    if "_main" in out and out["_main"].requests:
        run = out["_main"]
        fresh = run.prompt_tokens - run.cached_tokens
        written.append(charts.svg_stacked(
            RESULTS / "live_tokens_split.svg",
            "Live session prompt tokens: served from cache vs billed fresh",
            ["as-built, pinned"],
            [("served from cache", [run.cached_tokens]),
             ("billed fresh", [fresh])],
            value_fmt=lambda v: f"{v/1000:,.0f}k", ylabel="prompt tokens",
            subtitle="server-reported, one 12-turn session over this repo"))
    if "pin" in out:
        arms = [a for a in out["pin"]["arms"] if not a.get("error")]
        if arms:
            written.append(charts.svg_bars(
                RESULTS / "live_provider_affinity.svg",
                "Provider affinity decides whether caching happens at all",
                [(a["arm"], (a.get("second_request_hit_rate") or 0.0) * 100)
                 for a in arms],
                value_fmt=lambda v: f"{v:.0f}%",
                ylabel="hit rate on the 2nd request (%)",
                highlight="pinned",
                subtitle="identical prefix; the router setting is the only "
                         "difference"))
    return written


# ----------------------------------------------------------------- main ---
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["main", "prune", "pin-check"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--provider", default=DEFAULT_PROVIDER,
                    help="provider to pin to; empty string disables the pin")
    ap.add_argument("--no-pin", action="store_true",
                    help="run unpinned (expected to destroy the hit rate)")
    ap.add_argument("--turns", type=int, default=len(validate.SESSION))
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each session this many times and report "
                         "median/P10/P90 (recommended for large-scale runs)")
    ap.add_argument("--window", type=int, default=1_000_000)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)
    want = set(args.only or ["main", "prune", "pin-check"])
    provider = None if (args.no_pin or not args.provider) else args.provider

    index = build_index()
    pricing = asyncio.run(fetch_pricing(args.base_url, args.model))
    out: dict = {"meta": {"model": args.model, "base_url": args.base_url,
                          "provider": provider, "effort": args.effort,
                          "turns": args.turns, "pricing": pricing}}
    sections: list[str] = []

    if "main" in want:
        print(f"== live session, window {args.window:,} x{args.repeat} ==")
        runs: list[LiveRun] = []
        for rep in range(max(1, args.repeat)):
            print(f"  ── repetition {rep + 1}/{max(1, args.repeat)}")
            run = asyncio.run(run_session(
                f"Live session ({args.turns} turns, window {args.window:,})",
                index=index, window=args.window, turns=args.turns,
                model=args.model, base_url=args.base_url, provider=provider,
                effort=args.effort, timeout=args.timeout, pricing=pricing))
            runs.append(run)
            if rep == 0:
                out["main"], out["_main"] = run.as_dict(), run
                sections.append(report_run(run))
        if len(runs) > 1:
            out["main_aggregate"] = aggregate_runs(runs)
            sections.append(report_aggregate(out["main_aggregate"]))

    if "prune" in want:
        # Sized from our own estimate of this session's peak, in the same units
        # the manager uses to decide, so the prune is guaranteed rather than
        # hoped for.
        peak = max((r.predicted_prompt for r in out["_main"].requests),
                   default=0) if "_main" in out else 0
        if not peak:
            peak = _offline_peak(index)
        window = validate.prune_forcing_window(index, peak)
        print(f"== live session, window forced to {window:,} x{args.repeat} ==")
        prune_runs: list[LiveRun] = []
        for rep in range(max(1, args.repeat)):
            print(f"  ── repetition {rep + 1}/{max(1, args.repeat)}")
            run = asyncio.run(run_session(
                f"Live session at a {window:,}-token window (pruning forced)",
                index=index, window=window, turns=args.turns, model=args.model,
                base_url=args.base_url, provider=provider, effort=args.effort,
                timeout=args.timeout, pricing=pricing))
            prune_runs.append(run)
            if rep == 0:
                out["prune"], out["_prune"] = run.as_dict(), run
                sections.append(report_run(run))
        if len(prune_runs) > 1:
            out["prune_aggregate"] = aggregate_runs(prune_runs)
            sections.append(report_aggregate(out["prune_aggregate"]))

    if "pin-check" in want and args.provider:
        print("== provider affinity A/B ==")
        out["pin"] = asyncio.run(pin_check(
            index=index, model=args.model, base_url=args.base_url,
            provider=args.provider, effort=args.effort,
            timeout=args.timeout, pricing=pricing))
        sections.append(report_pin(out["pin"]))

    text = "\n\n".join(sections)
    print("\n" + text)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "live_acceptance.txt").write_text(text + "\n", encoding="utf-8")
    (RESULTS / "live_acceptance.json").write_text(
        json.dumps(validate._strip_private(out), indent=2), encoding="utf-8")
    if not args.no_charts:
        for p in write_charts(out):
            print(f"chart: {p.relative_to(ROOT)}")
    print(f"wrote {(RESULTS / 'live_acceptance.json').relative_to(ROOT)}")
    return 0


def _offline_peak(index) -> int:
    """Peak prompt size of the scripted session, from the offline bench.

    Only needed when ``--only prune`` skips the main run and there is no live
    session to size the window from.
    """
    run = asyncio.run(validate.run_variant("sizing", "", window=1_000_000,
                                           index=index))
    return max(m.prompt_tokens for m in run.metrics)


if __name__ == "__main__":
    raise SystemExit(main())
