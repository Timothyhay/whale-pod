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
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.syntax import Syntax
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


# ------------------------------------------------------- transient status
_FRAMES = ("🐋", "🐳", "🌊", "🫧")


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
