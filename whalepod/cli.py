"""WhalePod CLI.

Commands:
  whalepod                 interactive REPL
  whalepod ask <prompt>    single-turn answer (non-interactive)
  whalepod auth            configure endpoint + API key
  whalepod repo-map        build / inspect the repo symbol map
  whalepod context-stats   context-zone token estimate
  whalepod tokens <text>   token estimate for text
  whalepod rollback        restore pre-edit snapshots (works across processes)
  whalepod config          show resolved config

Two structural notes about the REPL:

  * The event loop runs in the **main thread**, one loop reused for the whole
    session. Streaming deltas are therefore printed as they arrive, and a write
    confirmation can prompt the user from inside a turn. The previous design ran
    the turn in a worker thread, which meant output only appeared once the turn
    was over and confirmation prompts fought the renderer for stdin.
  * The transcript is printed, not held in a full-screen ``Live`` region, so
    scrollback survives. The transient status occupies one self-erasing line.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .config import Config, load_config, save_global_config
from .ui import render as R

_PROMPT = "\n\033[1;36m❯\033[0m "


def _setup_encoding() -> None:
    """Render Unicode on Windows terminals instead of dying on cp1252."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


# --------------------------------------------------------------- session
class Session:
    """Everything one WhalePod conversation needs, wired together once.

    ``ask`` and the REPL share this so they cannot drift apart — notably they
    now share a single system prompt, which is also what keeps the prefix cache
    warm across both entry points.
    """

    def __init__(self, cfg: Config, root: Optional[Path] = None,
                 auto_yes: bool = False, quiet: bool = False,
                 max_tokens: Optional[int] = None):
        from .core.agent import Agent, AgentConfig
        from .core.ledger import ContextLedger
        from .core.messages import MessageManager
        from .core.prompt import build_system_prompt
        from .endpoints.factory import build_endpoint
        from .tools.registry import ToolRegistry

        cfg.check()                      # fail with an actionable message, early
        self.cfg = cfg
        self.root = Path(root or Path.cwd()).resolve()
        self.quiet = quiet
        self.console = R.make_console() if R.use_rich() else None

        self.endpoint = build_endpoint(
            cfg.endpoint.type, cfg.resolved_base_url(),
            api_key=cfg.resolved_api_key(),
            extra_headers=cfg.endpoint.extra_headers,
            timeout=cfg.endpoint.timeout,
            extra_body=cfg.endpoint.extra_body,
        )
        self.ledger = ContextLedger()
        self._instructions_from = ""     # set by _project_instructions()
        sandbox = "yes" if auto_yes else cfg.sandbox
        self.registry = ToolRegistry(self.root, sandbox_mode=sandbox,
                                     ledger=self.ledger)
        self.mm = MessageManager(window=cfg.context_window,
                                 prune_at=cfg.prune_at, prune_to=cfg.prune_to,
                                 reserve_tokens=cfg.reserve_tokens)
        self.mm.set_system(build_system_prompt(
            self._project_instructions(),
            guidelines=self.registry.guidelines()))
        self.mm.set_tools(self.registry.schemas())
        if self._instructions_from:
            self.notice(f"project instructions loaded from "
                        f"{self._instructions_from}")
        if self.registry.guard.is_readonly():
            self.notice("readonly sandbox: the write tools are not offered this "
                        "session")
        self.install_repo_map()

        self.agent = Agent(
            self.endpoint, self.registry, self.mm,
            AgentConfig(mode=cfg.default_mode, model=cfg.resolved_model(),
                        max_tokens=max_tokens,
                        compaction=cfg.compaction,
                        compaction_max_tokens=cfg.compaction_max_tokens,
                        notice_sink=self.notice),
            ledger=self.ledger,
        )

    # -- setup helpers -----------------------------------------------------
    # Ours first, then the conventions other agents established. Reading
    # AGENTS.md / CLAUDE.md is not vendor courtesy: those files are where a repo
    # already records its build commands, test invocation and house style, and a
    # WhalePod session that ignores them starts by rediscovering all of it.
    # First match wins — a repo that wants to say something different to
    # WhalePod says it in WHALEPOD.md.
    INSTRUCTION_FILES = ("WHALEPOD.md", ".whalepod.md", "AGENTS.md", "CLAUDE.md")

    def _project_instructions(self) -> str:
        """Durable per-project instructions, if the repo provides any."""
        for name in self.INSTRUCTION_FILES:
            p = self.root / name
            if p.is_file():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if text.strip():
                    self._instructions_from = name
                    return text
        return ""

    def install_repo_map(self) -> None:
        """Build the repo map and put it in the stable prefix.

        This is the fix for the map being built and then dropped on the floor:
        it was rendered into a local variable that never reached the request, so
        every session paid to scan the repo and the model never saw the result.
        """
        from .context.repo_map import build_repo_map
        from .core.prompt import repo_map_section
        rm = self.cfg.repo_map
        try:
            idx = build_repo_map(self.root, max_symbols=rm.max_symbols,
                                 languages=rm.languages,
                                 use_treesitter=rm.use_treesitter,
                                 exclude=rm.exclude,
                                 max_tokens=self.cfg.resolved_map_tokens())
        except Exception as e:
            self.notice(f"repo map unavailable: {type(e).__name__}: {e}")
            return
        self.registry.repo_index = idx
        self.mm.set_repo_map(repo_map_section(idx.render()))

    # -- output ------------------------------------------------------------
    def notice(self, text: str) -> None:
        if self.quiet:
            return
        if self.console:
            self.console.print(f"[dim]· {text}[/dim]")
        else:
            click.echo(f"· {text}", err=True)

    def echo(self, text: str, style: str = "") -> None:
        if self.console and style:
            self.console.print(f"[{style}]{text}[/{style}]")
        elif self.console:
            self.console.print(text)
        else:
            click.echo(text)

    async def aclose(self) -> None:
        await self.endpoint.aclose()


# ----------------------------------------------------------- turn runner
class TurnRunner:
    """Streams one turn to the terminal and prints its telemetry."""

    def __init__(self, session: Session, show_reasoning: bool = True):
        self.s = session
        self.show_reasoning = show_reasoning
        self.status = R.StatusLine()
        self.reasoning_tokens = 0
        self.out_tokens = 0
        self._in_reasoning = False
        self._wrote_text = False

    def sink(self, delta) -> None:
        from .core.tokenizer import estimate_tokens
        if delta.reasoning:
            self.reasoning_tokens += estimate_tokens(delta.reasoning)
            self._in_reasoning = True
            self._paint("reasoning")
        if delta.content:
            if self._in_reasoning:
                self._in_reasoning = False
            if not self._wrote_text:
                self.status.clear()
                self._wrote_text = True
            # Written straight through: this is the token-by-token stream, and
            # buffering it until the turn ended was the single biggest reason
            # the old REPL felt dead while the model was talking.
            sys.stdout.write(delta.content)
            sys.stdout.flush()
            self.out_tokens += estimate_tokens(delta.content)
            self._paint("streaming")

    def _paint(self, state: str) -> None:
        if self._wrote_text and state == "streaming":
            return          # don't fight the text we are printing
        bits = [f"{state}…", f"{self.status.elapsed():.0f}s"]
        if self.reasoning_tokens:
            bits.append(f"💭 {self.reasoning_tokens:,}")
        if self.out_tokens:
            bits.append(f"↓ {self.out_tokens:,}")
        self.status.update(" · ".join(bits))

    async def confirm(self, req) -> str:
        """Show the diff that is about to be written and ask."""
        self.status.clear()
        s = self.s
        if s.console:
            from rich.panel import Panel
            body = R.render_diff(req.plan.diff) if req.plan.diff else req.preview
            s.console.print(Panel(body, title=f"confirm · {req.summary}",
                                  border_style="yellow", title_align="left"))
        else:
            click.echo(f"\n--- confirm: {req.summary} ---")
            click.echo(req.preview)
        prompt = "  [y]es  [n]o  [a]lways this tool  > "
        try:
            return (await asyncio.to_thread(input, prompt)).strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("")
            return "no"

    async def run(self, user_text: str, add_user: bool = True) -> str:
        from .endpoints.base import EndpointError
        s = self.s
        s.agent.stream_sink = self.sink
        s.agent.config.confirm_callback = self.confirm
        try:
            text = await s.agent.run_turn(user_text, add_user=add_user)
        except EndpointError as e:
            self.status.clear()
            s.echo(f"endpoint error: {e}", "red")
            return ""
        except KeyboardInterrupt:
            self.status.clear()
            s.echo("(interrupted)", "yellow")
            return ""
        finally:
            self.status.clear()
        if self._wrote_text:
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif text:
            s.echo(text)
        return text

    def report(self) -> None:
        s = self.s
        st = s.mm.stats()
        s.notice(R.usage_line(st.usage))
        if self.reasoning_tokens and not self.show_reasoning:
            s.notice(f"reasoning: ~{self.reasoning_tokens:,} tokens (hidden)")


# ------------------------------------------------------------- commands
@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version")
@click.pass_context
def main(ctx, version):
    _setup_encoding()
    if version:
        click.echo(f"whalepod {__version__}")
        ctx.exit(0)
    if ctx.invoked_subcommand is None:
        repl()


ENDPOINT_TYPES = ["deepseek", "custom", "openai", "anthropic"]


def _do_auth_flow(cfg: Config) -> Config:
    """Interactive endpoint setup. Works in both CLI and REPL contexts."""
    click.echo("WhalePod endpoint setup")
    typ = click.prompt("Endpoint type", default=cfg.endpoint.type,
                       type=click.Choice(ENDPOINT_TYPES))
    base = click.prompt("Base URL (no trailing /v1, e.g. https://api.deepseek.com)",
                        default=cfg.endpoint.base_url or "")
    key = click.prompt("API key (ENTER to keep / use env)", default="",
                       show_default=False, hide_input=True)
    model = click.prompt("Model", default=cfg.endpoint.model or "",
                         show_default=False)
    cfg.endpoint.type = typ
    cfg.endpoint.base_url = base.strip()
    if key:
        cfg.endpoint.api_key = key
    if model:
        cfg.endpoint.model = model.strip()
    return cfg


@main.command("auth")
def auth():
    """Configure endpoint + API key."""
    cfg = load_config()
    _do_auth_flow(cfg)
    p = save_global_config(cfg)
    click.echo(f"Saved to {p}")


@main.command("repo-map")
@click.option("--limit", default=200, help="symbols to print")
@click.option("--max-tokens", default=0, type=int,
              help="token budget for the map (0 = config value)")
def repo_map(limit, max_tokens):
    """Build and print the repo symbol map."""
    from .context.repo_map import build_repo_map
    from .core.tokenizer import estimate_tokens
    cfg = load_config()
    rm = cfg.repo_map
    idx = build_repo_map(Path.cwd(), max_symbols=rm.max_symbols,
                         languages=rm.languages,
                         use_treesitter=rm.use_treesitter,
                         exclude=rm.exclude,
                         max_tokens=cfg.resolved_map_tokens())
    text = idx.render(limit, max_tokens=max_tokens or None)
    click.echo(f"tree-sitter: {idx.ts_used} | files: {len(idx.by_file)} | "
               f"symbols: {idx.symbol_count} | rendered: "
               f"{estimate_tokens(text):,} tok")
    click.echo(text)


@main.command("context-stats")
def context_stats():
    """Show what the stable prefix costs for this repo."""
    from .core.messages import MessageManager
    from .core.prompt import build_system_prompt, repo_map_section
    from .context.repo_map import build_repo_map
    from .tools.registry import ToolRegistry
    cfg = load_config()
    mm = MessageManager(window=cfg.context_window,
                        prune_at=cfg.prune_at, prune_to=cfg.prune_to,
                        reserve_tokens=cfg.reserve_tokens)
    # Not "readonly": that mode drops the write tools, which would undercount
    # the prefix this command exists to measure.
    reg = ToolRegistry(Path.cwd(), sandbox_mode="confirm")
    idx = build_repo_map(Path.cwd(), max_symbols=cfg.repo_map.max_symbols,
                         languages=cfg.repo_map.languages,
                         use_treesitter=cfg.repo_map.use_treesitter,
                         exclude=cfg.repo_map.exclude,
                         max_tokens=cfg.resolved_map_tokens())
    mm.set_system(build_system_prompt(guidelines=reg.guidelines()))
    mm.set_tools(reg.schemas())
    mm.set_repo_map(repo_map_section(idx.render()))
    st = mm.stats()
    click.echo(f"stable prefix : {st.stable_tokens:>9,} tok  "
               f"(system prompt + tools + repo map)")
    click.echo(f"history       : {st.history_tokens:>9,} tok")
    click.echo(f"total         : {st.total:>9,} tok / {st.window:,} "
               f"({st.fill * 100:.1f}%)")
    click.echo(f"reduce at     : {st.limit:>9,} tok  "
               f"(whichever comes first: {st.window:,}×{mm.prune_at:g} or "
               f"{st.window:,}−{mm.reserve_tokens:,} reserved)")
    click.echo("cache hit rate: measured per request at runtime "
               "(see /stats in the REPL)")


@main.command("tokens")
@click.argument("text")
def tokens(text):
    """Rough token estimate for a string."""
    from .core.tokenizer import estimate_tokens
    click.echo(f"~{estimate_tokens(text)} tokens")


@main.command("config")
def config():
    """Show resolved configuration (key redacted)."""
    cfg = load_config()
    click.echo(json.dumps({
        "endpoint": {k: ("***" if k == "api_key" and v else v)
                     for k, v in cfg.endpoint.__dict__.items()},
        "mode": cfg.default_mode,
        "sandbox": cfg.sandbox,
        "model": cfg.resolved_model(),
        "context_window": cfg.context_window,
        # Resolved, not raw: max_tokens is usually 0 ("auto"), and the whole
        # point of showing config is to answer "why is my map truncated?".
        "repo_map": {"max_tokens": cfg.resolved_map_tokens(),
                     "max_symbols": cfg.repo_map.max_symbols,
                     "languages": cfg.repo_map.languages,
                     "exclude": cfg.repo_map.exclude},
        "loaded_from": cfg._loaded_from,
    }, indent=2))


@main.command("rollback")
@click.option("--session", default="", help="session id (default: most recent)")
@click.option("--list", "do_list", is_flag=True, help="list recorded sessions")
def rollback(session, do_list):
    """Restore files written by a previous WhalePod session.

    Reads the on-disk snapshot manifest, so this works from a fresh process —
    it used to construct an empty in-memory manager and always report
    "nothing to roll back".
    """
    from .sandbox.snapshot import SnapshotManager, backup_root
    if do_list:
        found = SnapshotManager.sessions()
        if not found:
            click.echo(f"no snapshot sessions under {backup_root()}")
            return
        for d in reversed(found):
            mgr = SnapshotManager.load(d)
            flag = " (rolled back)" if mgr.rolled_back else ""
            click.echo(f"{d.name}  {len(mgr.snapshots)} file(s)  "
                       f"{mgr.root or '?'}{flag}")
        return
    if session:
        mgr = SnapshotManager.load(backup_root() / session)
        if not mgr.snapshots:
            click.echo(f"no snapshots recorded for session {session!r}")
            return
    else:
        mgr = SnapshotManager.load_latest(root=Path.cwd())
        if mgr is None:
            mgr = SnapshotManager.load_latest()
        if mgr is None:
            click.echo("nothing to roll back")
            return
    click.echo(f"rolling back session {mgr.session_id} "
               f"({len(mgr.snapshots)} file(s))")
    for line in mgr.rollback():
        click.echo(f"  {line}")


# ------------------------------------------------------------- ask ---
@main.command("ask")
@click.argument("prompt")
@click.option("--model", default=None)
@click.option("--mode", type=click.Choice(["thinking", "instant"]), default=None)
@click.option("--max-tokens", type=int, default=None)
@click.option("--yes", is_flag=True, help="auto-approve writes (unattended)")
@click.option("--no-tools", is_flag=True, help="answer without touching the repo")
def ask(prompt, model, mode, max_tokens, yes, no_tools):
    """Single-turn answer (non-interactive)."""
    _setup_encoding()
    cfg = load_config()
    if model:
        cfg.endpoint.model = model
    if mode:
        cfg.default_mode = mode
    if no_tools:
        cfg.sandbox = "readonly"
    try:
        asyncio.run(_run_ask(cfg, prompt, auto_yes=yes, max_tokens=max_tokens))
    except click.ClickException:
        raise
    except Exception as e:
        raise SystemExit(f"error: {type(e).__name__}: {e}")


async def _run_ask(cfg: Config, prompt: str, auto_yes: bool,
                   max_tokens: Optional[int]):
    session = Session(cfg, auto_yes=auto_yes, max_tokens=max_tokens)
    runner = TurnRunner(session)
    try:
        await runner.run(prompt)
        runner.report()
    finally:
        await session.aclose()


# --------------------------------------------------------- repl ------
def repl():
    """Interactive REPL. The event loop lives in this thread."""
    _setup_encoding()
    cfg = load_config()
    try:
        session = Session(cfg)
    except ValueError as e:
        raise SystemExit(f"{e}")

    st = session.mm.stats()
    session.echo(R.render_startup(
        __version__, cfg.resolved_model(), str(session.root),
        st.stable_tokens,
        session.registry.repo_index.symbol_count if session.registry.repo_index else 0))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    first = True
    try:
        while True:
            if not first:
                session.echo(R.render_divider())
            first = False
            try:
                user = input(_PROMPT)
            except (EOFError, KeyboardInterrupt):
                click.echo("")
                break
            user = user.strip()
            if not user:
                first = True   # don't add divider for blank input
                continue
            if user in ("/quit", "/exit", "/q"):
                break
            if _handle_command(session, user):
                continue
            runner = TurnRunner(session)
            try:
                loop.run_until_complete(runner.run(user))
            except KeyboardInterrupt:
                session.echo("(interrupted)", "yellow")
                session.mm.close_open_tool_calls("(interrupted by user)")
            runner.report()
    finally:
        try:
            loop.run_until_complete(session.aclose())
        finally:
            loop.close()
    session.echo("goodbye 🐋", "dim")


def _handle_command(session: Session, user: str) -> bool:
    """Handle a /slash command. Returns True if it was one."""
    if not user.startswith("/"):
        return False
    cmd = user.split()[0]
    if cmd == "/help":
        session.echo(R.render_help())
    elif cmd == "/config":
        _do_auth_flow(session.cfg)
        p = save_global_config(session.cfg)
        session.echo(f"Saved to {p} — restart to apply", "dim")
    elif cmd == "/mode":
        session.echo(f"mode: {session.agent.toggle_mode()}", "yellow")
    elif cmd == "/stats":
        st = session.mm.stats()
        ctx = R.context_line(st, session.cfg.resolved_model(),
                             session.agent.mode, session.cfg.sandbox)
        detail = [R.usage_line(st.usage)]
        if st.session_usage:
            detail.append(f"session: {R.usage_line(st.session_usage)}")
        if st.compactions:
            detail.append(f"context compacted {st.compactions} time(s) this "
                          f"session")
        if st.prunes:
            detail.append(f"context pruned (no summary) {st.prunes} time(s) "
                          f"this session")
        session.echo(R.render_stats(ctx, detail))
    elif cmd == "/context":
        session.echo(session.ledger.stats_line(), "dim")
        session.echo(session.ledger.summary(), "dim")
    elif cmd == "/refresh":
        session.install_repo_map()
        idx = session.registry.repo_index
        session.echo(f"repo map: {idx.symbol_count if idx else 0} symbols", "dim")
    elif cmd == "/rollback":
        for line in session.registry.rollback():
            session.echo(f"  {line}", "dim")
    elif cmd == "/clear":
        n = session.mm.clear_history()
        session.ledger.entries.clear()
        session.echo(f"cleared {n} message(s); repo map and prompt kept "
                     f"(they are still a cache hit)", "dim")
    elif cmd == "/compact":
        loop = asyncio.get_event_loop()
        evt = loop.run_until_complete(session.agent.compact_now())
        if evt is None:
            session.echo("nothing to compact (need at least 2 turns)", "dim")
    else:
        session.echo(f"unknown command {cmd} — /help", "red")
    return True


if __name__ == "__main__":
    main()
