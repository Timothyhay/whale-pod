"""Rendering helpers for the CLI: status line, diffs, stats.

Uses ``rich`` when available and degrades to plain text for non-TTY output.

Why there is no full-screen layout here
    An earlier version kept the whole transcript inside a ``rich.Live`` with a
    pinned header. ``Live`` re-renders its entire renderable on every refresh,
    so once the conversation grew past one terminal height the transcript was
    clipped and scrollback was destroyed — and because streaming deltas arrived
    in a worker thread, the text appeared only after the turn finished.

    Instead: finished output is *printed* (so the terminal keeps scrollback and
    the user's own scrolling works), and the transient status lives on a single
    line that erases itself. That single line is the only thing we control.
"""
from __future__ import annotations

import shutil
import sys
import time

try:
    from rich import box
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    _HAS_RICH = True
except Exception:                                    # pragma: no cover
    _HAS_RICH = False
    Console = None


def is_tty() -> bool:
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False


def use_rich() -> bool:
    return _HAS_RICH and is_tty()


def make_console():
    return Console(soft_wrap=False, highlight=False) if _HAS_RICH else None


# --------------------------------------------------------- slash-help table
HELP_ITEMS = [
    ("/help",      "this message"),
    ("/config",    "configure endpoint + API key"),
    ("/mode",      "toggle thinking \u00b7 instant"),
    ("/stats",     "context + measured cache telemetry"),
    ("/context",   "what files are currently loaded"),
    ("/refresh",   "rescan the repo map"),
    ("/rollback",  "undo this session's writes"),
    ("/clear",     "start a fresh conversation"),
    ("/quit",      "exit"),
]


def render_help():
    """Return a ``rich.Table`` of slash commands, or plain text."""
    if not _HAS_RICH:
        return "\n".join(f"  {cmd:12s}{desc}"
                         for cmd, desc in HELP_ITEMS)
    table = Table(show_header=False, box=box.SIMPLE,
                  padding=(0, 2), border_style="dim blue")
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="dim")
    for cmd, desc in HELP_ITEMS:
        table.add_row(cmd, desc)
    return table


# ----------------------------------------------------------------- divider
def render_divider():
    """Return a subtle horizontal rule, or a plain line of dashes."""
    if not _HAS_RICH:
        return "\u2500" * 60
    return Rule(style="dim")


# ------------------------------------------------------------- banner
BANNER = r"""
 __      __ _           _      ___          _
 \ \    / /| |_  __ _  | | ___| _ \___  __| |
  \ \/\/ / | ' \/ _` | | |/ -_)  _/ _ \/ _` |
   \_/\_/  |_||_\__,_| |_|\___|_| \___/\__,_|
""".strip("\n")


def render_startup(version: str, model: str, root: str,
                   stable_tokens: int, symbol_count: int):
    """Return a styled welcome panel, or plain text for non-TTY."""
    if not _HAS_RICH:
        return (f"{BANNER}\n\n"
                f"WhalePod v{version} \u00b7 {model} \u00b7 {root}\n"
                f"stable prefix ~{stable_tokens:,} tok \u00b7 "
                f"{symbol_count} symbols mapped \u00b7 /help for commands")

    banner = Text()
    for i, line in enumerate(BANNER.split("\n")):
        if i > 0:
            banner.append("\n")
        g = 140 + i * 18
        b_val = 230 - i * 10
        banner.append(line, style=f"rgb(0,{min(g,255)},{max(b_val,20)})")

    info = Text()
    info.append(" WhalePod v", style="bold cyan")
    info.append(version, style="bold cyan")
    info.append("  \u00b7  ", style="dim")
    info.append(model, style="bold white")
    info.append("  \u00b7  ", style="dim")
    info.append(str(root), style="dim")
    info.append("\n")
    info.append(f" stable prefix ~{stable_tokens:,} tok  \u00b7  "
                f"{symbol_count} symbols mapped  \u00b7  /help for commands",
                style="dim")

    body = Text()
    body.append(banner)
    body.append("\n")
    body.append(info)

    return Panel(body, border_style="cyan", padding=(1, 2))


# ------------------------------------------------------------------ diff
def render_diff(diff_text: str, console=None) -> str:
    """Colourize a unified diff as rich markup (or return it unchanged)."""
    if not diff_text:
        return "(no diff)"
    if not _HAS_RICH:
        return diff_text
    out = []
    for ln in diff_text.splitlines():
        esc = ln.replace("[", "\\[")
        if ln.startswith(("+++", "---", "@@")):
            out.append(f"[bold blue]{esc}[/bold blue]")
        elif ln.startswith("+"):
            out.append(f"[green]{esc}[/green]")
        elif ln.startswith("-"):
            out.append(f"[red]{esc}[/red]")
        else:
            out.append(f"[dim]{esc}[/dim]")
    return "\n".join(out)


# ------------------------------------------------------------------ stats
def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def context_line(stats, model: str, mode: str, sandbox: str) -> str:
    """One-line context/cache summary from :class:`ContextStats`.

    The cache hit rate shown is the provider's *measured* number. When the
    provider reports nothing it prints "—" rather than a computed guess: the
    old status bar derived a hit rate from its own token estimate, so it
    displayed a healthy percentage even when nothing was cached at all.
    """
    hit = _pct(stats.cache_hit_rate)
    return (f"{model} · {mode.upper()} · ctx {stats.total:,}/{stats.window:,} "
            f"({stats.fill * 100:.1f}%) · cache {hit} · sandbox:{sandbox}")


def usage_line(usage) -> str:
    """Per-request token accounting, or a note that the provider sent none."""
    if usage is None:
        return "tokens: (provider reported no usage for this request)"
    parts = [f"in {usage.prompt_tokens:,}", f"out {usage.completion_tokens:,}"]
    if usage.reasoning_tokens:
        parts.append(f"reasoning {usage.reasoning_tokens:,}")
    if usage.cached_tokens or usage.cache_write_tokens:
        parts.append(f"cached {usage.cached_tokens:,} "
                     f"({_pct(usage.cache_hit_rate)})")
    if usage.cache_write_tokens:
        parts.append(f"cache-write {usage.cache_write_tokens:,}")
    return "tokens: " + " · ".join(parts)


def render_stats(context_line_text: str, detail_lines: list[str]):
    """Wrap stats in a panel when Rich is available, or return plain text."""
    if not _HAS_RICH:
        return "\n".join([context_line_text] + detail_lines)
    body = Text()
    body.append(context_line_text, style="bold")
    for line in detail_lines:
        body.append("\n")
        body.append(line, style="dim")
    return Panel(body, title="stats", border_style="blue",
                 title_align="left", padding=(0, 1))


# ------------------------------------------------------- transient status
_FRAMES = ("🐋", "🐳", "🌊", "🫧", "🐚", "🐟", "✨", "💫")


class StatusLine:
    """A single self-erasing terminal line for transient progress.

    Written to stderr so redirecting stdout to a file keeps the transcript
    clean. Silently does nothing when stderr is not a TTY.
    """

    def __init__(self, stream=None, enabled: bool = None):
        self.stream = stream or sys.stderr
        if enabled is None:
            try:
                enabled = self.stream.isatty()
            except Exception:
                enabled = False
        self.enabled = bool(enabled)
        self._len = 0
        self._start = time.monotonic()
        self._tick = 0

    def width(self) -> int:
        try:
            return max(20, shutil.get_terminal_size((80, 24)).columns - 1)
        except Exception:
            return 79

    def update(self, text: str) -> None:
        if not self.enabled:
            return
        self._tick += 1
        icon = _FRAMES[self._tick // 3 % len(_FRAMES)]
        line = f"{icon} {text}"[: self.width()]
        pad = " " * max(0, self._len - len(line))
        try:
            self.stream.write("\r" + line + pad)
            self.stream.flush()
        except Exception:
            self.enabled = False
        self._len = len(line)

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def clear(self) -> None:
        if not self.enabled or not self._len:
            return
        try:
            self.stream.write("\r" + " " * self._len + "\r")
            self.stream.flush()
        except Exception:
            pass
        self._len = 0
