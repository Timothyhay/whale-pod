"""Tool result and argument helpers shared across tools.

Truncation lives here because it is where tool output meets the context window,
and getting it wrong is expensive in two different ways. Too generous and one
`cat` of a build log eats the window; too clever and the model is told it has
seen content that was never sent.

There are three shapes, and which one a tool wants depends on where the
information is:

  * :func:`truncate_head` — keep the *beginning*. For file reads: line 1 is the
    top of the file and the model can ask for the rest by line number.
  * :func:`truncate_tail` — keep the *end*. For command output: the exit status,
    the traceback and the summary are all at the bottom, and a middle-out cut
    reliably threw away the one part that mattered.
  * :func:`truncate` — keep both ends. Only for text shown to the *user*
    (diffs, previews), where "…" in the middle is readable and nobody is going
    to act on the missing part programmatically.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    ok: bool = True
    output: str = ""
    error: str = ""
    # structured values tools may attach (e.g. parsed file-name for cache hints)
    meta: dict = field(default_factory=dict)

    def to_text(self) -> str:
        if self.ok:
            return self.output
        return f"ERROR: {self.error or self.output}"


def parse_args_checked(raw) -> tuple[dict, str]:
    """Parse a tool-call arguments JSON string. Returns ``(args, error)``.

    The error matters: swallowing a JSON failure into ``{}`` made a malformed
    tool call look like a call with no arguments, so the model was told
    "path not found: ''" and had no way to learn that its JSON was broken.
    """
    if raw is None or raw == "":
        return {}, ""
    if isinstance(raw, dict):
        return raw, ""
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        return {}, (f"arguments are not valid JSON ({e}); "
                    f"received: {str(raw)[:300]}")
    if not isinstance(d, dict):
        return {}, f"arguments must be a JSON object, got {type(d).__name__}"
    return d, ""


def parse_args(raw: str) -> dict:
    """Lenient variant kept for callers that cannot report an error."""
    return parse_args_checked(raw)[0]


def truncate(text: str, limit: int = 60_000) -> str:
    """Middle-out cut, for text a *human* reads (diffs, confirmation previews).

    Do not use this for tool output the model has to act on: the model cannot
    ask for "the middle", so a middle-out cut leaves a hole with no way to
    address it. Use :func:`truncate_head` or :func:`truncate_tail`, which cut at
    one end and can therefore say how to get the rest.
    """
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2):]
    return f"{head}\n… [truncated {len(text) - limit} chars] …\n{tail}"


DEFAULT_MAX_LINES = 2_000
DEFAULT_MAX_CHARS = 60_000


@dataclass
class Truncation:
    """What a line-granular cut actually delivered.

    Both limits are reported because "you got 2000 lines" and "you got 60 KB"
    lead the model to different next moves: the first says read on by line, the
    second says the lines are huge and it should narrow instead.
    """
    lines: list[str] = field(default_factory=list)
    truncated: bool = False
    by: str = ""                 # "lines" | "chars" | ""
    total_lines: int = 0
    kept_lines: int = 0
    # 1-based inclusive positions of the kept slice within the input lines.
    first: int = 0
    last: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _cut(lines: list[str], max_lines: int, max_chars: int,
         from_end: bool) -> Truncation:
    """Take whole lines from one end until either limit is reached."""
    total = len(lines)
    if total == 0:
        return Truncation()
    max_lines = max(1, max_lines)
    max_chars = max(1, max_chars)
    order = range(total - 1, -1, -1) if from_end else range(total)
    kept: list[int] = []
    used = 0
    by = ""
    for i in order:
        if len(kept) >= max_lines:
            by = "lines"
            break
        cost = len(lines[i]) + 1
        # The first line is taken unconditionally, so a budget smaller than one
        # line still returns something rather than an empty result.
        if kept and used + cost > max_chars:
            by = "chars"
            break
        kept.append(i)
        used += cost
    if from_end:
        kept.reverse()
    out = [lines[i] for i in kept]
    if used > max_chars:
        # ...but "unconditionally" cannot mean "unboundedly": one minified line
        # can be megabytes. Whole-line granularity is a preference, not a licence
        # to blow the budget, so the single kept line is hard-cut at the end we
        # are keeping.
        out = [out[0][-max_chars:] if from_end else out[0][:max_chars]]
        by = "chars"
    return Truncation(lines=out, truncated=bool(by), by=by, total_lines=total,
                      kept_lines=len(out), first=kept[0] + 1, last=kept[-1] + 1)


def truncate_head(lines: list[str], max_lines: int = DEFAULT_MAX_LINES,
                  max_chars: int = DEFAULT_MAX_CHARS) -> Truncation:
    """Keep whole lines from the start. For file reads."""
    return _cut(lines, max_lines, max_chars, from_end=False)


def truncate_tail(lines: list[str], max_lines: int = DEFAULT_MAX_LINES,
                  max_chars: int = DEFAULT_MAX_CHARS) -> Truncation:
    """Keep whole lines from the end. For command output."""
    return _cut(lines, max_lines, max_chars, from_end=True)


def truncate_line(text: str, limit: int = 500) -> str:
    """Bound one line (a grep hit). A 200 KB minified line is not a match
    anybody can read, and 200 of them are a window-filling accident."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… [+{len(text) - limit} chars]"
