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
from datetime import datetime

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
    ("/trace",     "toggle the tool-call trace"),
    ("/stats",     "context + measured cache telemetry"),
    ("/context",   "what files are currently loaded"),
    ("/refresh",   "rescan the repo map"),
    ("/backups",   "backup points on a timeline (all · prune)"),
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


# ------------------------------------------------------- transcript marks
# The transcript is a flat stream of printed blocks, so every kind of block owns
# a mark in the left gutter:
#
#     ❯  the user's prompt          ▸  a tool call
#     ●  the assistant's answer     └  that call's result
#     ·  a notice (dim, stderr)
#
# Only the prompt had one before, so a tool trace, a notice and the model's prose
# arrived as unadorned lines and read as a single wall of text.
ANSWER_MARK = "●"
TOOL_MARK = "▸"
RESULT_MARK = "└"
GUTTER = "  "                  # continuation indent — the width of "● "

_ANSWER_ANSI = "\033[1;32m"    # green; the user's ❯ is cyan, so they can't be
_RESET = "\033[0m"             # confused at a glance


def answer_open(color: bool = True) -> str:
    """Opening gutter for a block of assistant text.

    Raw ANSI rather than Rich markup on purpose: the answer is streamed to
    stdout token by token, and handing each delta to a ``Console`` would re-wrap
    and re-highlight text that is already half-printed.
    """
    return f"{_ANSWER_ANSI}{ANSWER_MARK}{_RESET} " if color else f"{ANSWER_MARK} "


# ------------------------------------------------------------- tool trace
def _clip(text, limit: int) -> str:
    """Collapse whitespace and bound the length — trace lines are one line."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def tool_target(name: str, args: dict) -> str:
    """The one argument that says *which* call this is.

    Deliberately not a dump of the arguments: ``create_file`` carries a whole
    file and ``edit_file`` the whole replacement text, so printing them verbatim
    buries the trace in the content it exists to summarize.
    """
    args = args or {}
    if name == "read_file":
        start = int(args.get("start") or 0)
        end = int(args.get("end") or 0)
        rng = f":{start or 1}-{end or ''}" if (start or end) else ""
        return f"{args.get('path') or '?'}{rng}"
    if name in ("read_dir", "tree_view"):
        return args.get("path") or "."
    if name == "grep":
        where = args.get("path") or args.get("glob") or ""
        pattern = _clip(args.get("pattern", ""), 50)
        return f"{pattern} in {where}" if where else pattern
    if name in ("edit_file", "create_file", "delete_file"):
        return args.get("path") or "?"
    if name == "apply_patch":
        files = [ln[6:] for ln in (args.get("patch") or "").splitlines()
                 if ln.startswith("+++ b/")]
        shown = ", ".join(files[:3]) + ("…" if len(files) > 3 else "")
        return shown or "patch"
    if name == "run_command":
        return _clip(args.get("cmd", ""), 70)
    if not args:
        return ""
    return _clip(", ".join(f"{k}={v!r}" for k, v in args.items()), 70)


def tool_result_summary(name: str, result) -> str:
    """One line saying what a call produced — a size or a reason, never a dump."""
    if result is None:
        return "(no result)"
    meta = getattr(result, "meta", None) or {}
    if not result.ok:
        return _clip(result.error or result.output or "failed", 90)
    if meta.get("ledger_hit"):
        return "already in context"
    if name == "read_file":
        r = meta.get("read") or {}
        if r:
            if r.get("complete"):
                return f"{r.get('lines', 0)} lines"
            return (f"lines {r.get('start', 0)}-{r.get('end', 0)} of "
                    f"{r.get('lines', 0)}")
    if name == "run_command":
        code = meta.get("exit_code")
        # The output carries a two-line "$ cmd / (exit n)" header we already say.
        body = max(0, len((result.output or "").splitlines()) - 2)
        bits = [] if code is None else [f"exit {code}"]
        bits.append(f"{body} line{'' if body == 1 else 's'}")
        return " · ".join(bits)
    if name == "grep":
        out = (result.output or "").strip()
        if not out or out.startswith("(no matches)"):
            return "no matches"
        n = len([ln for ln in out.splitlines() if ln.strip()])
        return f"{n} match{'' if n == 1 else 'es'}"
    if meta.get("paths"):
        # A committed write puts "# edit_file: edit a.py (+3 -1)" on its first
        # line. The diffstat is the part worth reading; the path is already on
        # the call line directly above.
        lines = (result.output or "").splitlines()
        head = lines[0].lstrip("# ") if lines else ""
        if ": " in head:
            head = head.split(": ", 1)[1]
        return _clip(head or ", ".join(meta["paths"]), 90)
    lines = (result.output or "").splitlines()
    if not lines:
        return "ok"
    if len(lines) == 1:
        return _clip(lines[0].lstrip("# ").strip(), 90)
    return f"{len(lines)} lines"


def tool_call_text(name: str, args: dict) -> str:
    return f"{TOOL_MARK} {name}({tool_target(name, args)})"


def _timing(duration: float) -> str:
    # Below 100 ms the number is noise next to the line it sits on.
    return f"  ({duration:.1f}s)" if duration >= 0.1 else ""


def tool_result_text(name: str, result, duration: float = 0.0,
                     label: str = "") -> str:
    """``label`` names the call this result belongs to — see the note on
    :func:`render_tool_result`."""
    head = f"{label} " if label else ""
    return (f"{GUTTER}{RESULT_MARK} {head}{tool_result_summary(name, result)}"
            f"{_timing(duration)}")


def render_tool_call(name: str, args: dict):
    """Styled call line for a Rich console, or the plain string."""
    if not _HAS_RICH:
        return tool_call_text(name, args)
    t = Text()
    t.append(f"{TOOL_MARK} ", style="blue")
    t.append(name, style="bold blue")
    t.append(f"({tool_target(name, args)})", style="dim")
    return t


def render_tool_result(name: str, result, duration: float = 0.0,
                       label: str = ""):
    """Styled result line for a Rich console, or the plain string.

    ``label`` is set when the result does not sit directly under its own call
    line: a batch of reads runs concurrently and finishes in whatever order the
    filesystem hands them back, so an unlabelled result under two call lines
    would be guesswork.
    """
    if not _HAS_RICH:
        return tool_result_text(name, result, duration, label)
    ok = result is None or result.ok
    t = Text()
    t.append(f"{GUTTER}{RESULT_MARK} ", style="dim")
    if label:
        t.append(f"{label} ", style="blue")
    t.append(tool_result_summary(name, result), style="dim" if ok else "red")
    t.append(_timing(duration), style="dim")
    return t


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


# ---------------------------------------------------------------- backups
# The backup root is a flat pile of timestamped directories, which `ls` prints as
# 75 indistinguishable names. Drawn as a timeline, newest at the top, the four
# things a rollback decision needs sit on one line: which point to name, what it
# holds, which project it came from, and whether retention is about to reclaim it.
#
#     ╷ now
#     ● 20260807-034412  2 files ·  1.3 KB · 4m ago    WhalePod
#     │  ├ whalepod/cli.py
#     │  └ whalepod/config.py
#     ┊ 4 days earlier
#     × 20260802-191631  1 file  ·  0.2 KB · 5d ago    WhalePod   older than 14d
#     ╵ 65 older points hidden · /backups all
#
# Marks are all single-width on purpose: an emoji-presentation glyph like ⌛ is
# drawn double-width by most terminals, which shifts every column after it out of
# alignment on exactly the rows that matter most.
MARK_KEPT = "●"          # backups present, restorable
MARK_ROLLED_BACK = "○"   # already restored from
MARK_EXPIRING = "×"      # past retention; the next sweep reclaims it
MARK_ORPHAN = "◌"        # directory with no manifest (killed mid-write)
MAX_FILE_LINES = 3       # per point; the rest collapse to "+N more"


def _ago(when: datetime, now: datetime = None) -> str:
    now = now or datetime.now()
    secs = max(0.0, (now - when).total_seconds())
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86_400:
        return f"{int(secs // 3600)}h ago"
    if secs < 86_400 * 14:
        return f"{int(secs // 86_400)}d ago"
    return f"{int(secs // (86_400 * 7))}w ago"


def _size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024 ** 2:.1f} MB"


def retention_line(retention) -> str:
    """The policy in words. Says so explicitly when nothing is capped."""
    if retention is None:
        return "retention: not applied"
    bits = []
    if retention.max_sessions:
        bits.append(f"{retention.max_sessions} newest")
    if retention.max_age_days:
        bits.append(f"{retention.max_age_days:g} days")
    if retention.max_total_mb:
        bits.append(f"{retention.max_total_mb:g} MB")
    return "keep " + " · ".join(bits) if bits else "no limits set (grows forever)"


def _mark(info):
    if info.expires:
        return MARK_EXPIRING, "yellow"
    if not info.has_manifest:
        return MARK_ORPHAN, "red"
    if info.rolled_back:
        return MARK_ROLLED_BACK, "dim"
    return MARK_KEPT, "green"


def _files_text(info) -> str:
    if not info.has_manifest:
        return "no manifest"
    return f"{info.files} file{'' if info.files == 1 else 's'}"


def backup_rows(infos, base, retention=None, limit=None) -> list:
    """The timeline as rows of ``(text, style)`` segments.

    Split from the renderers so the plain-text and Rich versions cannot draw two
    different pictures — the art is decided once, here, and only the colour is
    applied twice.
    """
    rows = [
        [("🐋 backup points", "bold cyan"), (f" · {base}", "dim")],
    ]
    n = len(infos)
    rows.append([(f"   {n} point{'' if n == 1 else 's'} · "
                  f"{_size(sum(i.bytes for i in infos))} · "
                  f"{retention_line(retention)}", "dim")])
    expiring = [i for i in infos if i.expires]
    if expiring:
        rows.append([(f"   {len(expiring)} past retention", "yellow"),
                     (" — reclaimed by the next write, or now with "
                      "`/backups prune`", "dim")])
    if not infos:
        rows.append([("   nothing recorded yet — snapshots appear here the first "
                      "time a session writes a file", "dim")])
        return rows

    shown = infos if limit is None else infos[:max(0, limit)]
    w_files = max(len(_files_text(i)) for i in shown)
    w_size = max(len(_size(i.bytes)) for i in shown)
    w_age = max(len(_ago(i.when)) for i in shown)

    rows.append([("", "")])
    rows.append([("  ╷", "dim blue"), (" now", "dim")])
    prev = None
    for info in shown:
        if prev is not None:
            gap_days = int((prev.when - info.when).total_seconds() // 86_400)
            if gap_days >= 1:
                rows.append([("  ┊", "dim blue"),
                             (f" {gap_days} day{'' if gap_days == 1 else 's'} "
                              f"earlier", "dim")])
        mark, mark_style = _mark(info)
        stale = bool(info.expires or info.rolled_back)
        line = [
            (f"  {mark} ", mark_style),
            (info.session, "dim" if stale else "bold"),
            (f"  {_files_text(info):<{w_files}} · {_size(info.bytes):>{w_size}}"
             f" · {_ago(info.when):<{w_age}}", "dim"),
            (f"  {info.root.name if info.root else '?'}", "cyan"),
        ]
        if info.rolled_back:
            line.append(("  rolled back", "dim"))
        if info.expires:
            line.append((f"  {info.expires}", "yellow"))
        rows.append(line)

        head = info.paths[:MAX_FILE_LINES]
        extra = len(info.paths) - len(head)
        for j, p in enumerate(head):
            last = not extra and j == len(head) - 1
            rows.append([(f"  │  {'└' if last else '├'} ", "dim blue"),
                         (p, "dim")])
        if extra:
            rows.append([("  │  └ ", "dim blue"), (f"+{extra} more", "dim")])
        prev = info

    hidden = len(infos) - len(shown)
    if hidden:
        rows.append([("  ╵", "dim blue"),
                     (f" {hidden} older point{'' if hidden == 1 else 's'} hidden"
                      f" · `/backups all`", "dim")])
    else:
        rows.append([("  ╵", "dim blue"), (" oldest", "dim")])
    return rows


def backups_text(rows) -> str:
    return "\n".join("".join(seg for seg, _ in row) for row in rows)


def render_backups(infos, base, retention=None, limit=None):
    """Styled timeline for a Rich console, or the same art in plain text.

    Not wrapped in a ``Panel``: the list can run to hundreds of lines with
    ``/backups all``, and a border around that only steals width from the paths.
    """
    rows = backup_rows(infos, base, retention, limit)
    if not _HAS_RICH:
        return backups_text(rows)
    body = Text()
    for i, row in enumerate(rows):
        if i:
            body.append("\n")
        for seg, style in row:
            body.append(seg, style=style or None)
    return body


def render_prune(removed, base) -> str:
    """What a sweep reclaimed, or that there was nothing to reclaim."""
    if not removed:
        return "backups: nothing past retention"
    lines = [f"backups: reclaimed {len(removed)} point"
             f"{'' if len(removed) == 1 else 's'} from {base}"]
    for d, reason in reversed(removed):        # newest first, like the timeline
        lines.append(f"  × {d.name}  ({reason})")
    return "\n".join(lines)


# ------------------------------------------------------- transient status
_FRAMES = ("🐋", "🐳", "🌊", "🫧", "✨", "💫")


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
