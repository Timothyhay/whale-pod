"""Read-only tools (lazy loading). The agent pulls content on demand.

Read results carry ``meta`` describing exactly what was **delivered** — path,
line range, whether that was the complete file, file identity and an approximate
token cost. The agent feeds that to the context ledger so the same range is not
shipped into the window twice.

"Delivered" is load-bearing. This used to report the *requested* range, which is
0-0 ("whole file") for a plain read, while the output itself was cut in the
middle to fit a character budget. The ledger then believed the whole file was in
the window and answered a later read of the missing part with "it is already
above — scroll up", pointing the model at content that was never sent. Reads now
cut from the head at line granularity, report the range that actually went out,
and say how to ask for the rest.
"""
from __future__ import annotations

import fnmatch
import io
import os
import re
from pathlib import Path

from . import textfile
from .base import (
    DEFAULT_MAX_CHARS, DEFAULT_MAX_LINES, ToolResult, truncate, truncate_head,
    truncate_line,
)

IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
               ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
               "target", ".idea", ".vscode", ".tox", ".next"}


def _rel(guard, p: Path) -> str:
    try:
        return str(p.relative_to(guard.root)).replace("\\", "/")
    except ValueError:
        return str(p)


def tool_read_file(guard, repo_map, path: str, start: int = 0, end: int = 0,
                   max_lines: int = DEFAULT_MAX_LINES,
                   max_chars: int = DEFAULT_MAX_CHARS) -> ToolResult:
    """Read a file (optionally a line range), with line numbers.

    Output is bounded by whole lines from the *head* of the requested range. If
    the range does not fit, the result says which lines it contains and how to
    continue, and ``meta["read"]`` describes only what was delivered.
    """
    try:
        p = guard.resolve_within(path)
    except Exception as e:
        return ToolResult(ok=False, error=str(e))
    if not p.exists():
        return ToolResult(ok=False, error=f"file not found: {path}")
    if p.is_dir():
        return ToolResult(ok=False, error=f"{path} is a directory (use read_dir)")
    tf, err = textfile.read(p)
    if err or tf is None:
        return ToolResult(
            ok=False,
            error=str(err) if "utf-8" not in str(err) else f"{path} is binary (not text)")

    rel = _rel(guard, p)
    lines = tf.text.splitlines()
    total = len(lines)
    if total == 0:
        # An empty file used to come back as "has 0 lines; start=1 is past the
        # end", which reads like a bad argument rather than an empty file.
        return ToolResult(ok=True, output=f"# {rel}  (empty file)", meta={
            "read": {"path": rel, "start": 1, "end": 0, "lines": 0,
                     "complete": True, "truncated": False, "chars": 0}})
    req_start = max(1, start) if (start or end) else 1
    req_end = (min(total, end) if end else total) if (start or end) else total
    if req_start > total:
        return ToolResult(
            ok=False,
            error=f"{rel} has {total} lines; start={req_start} is past the end")

    cut = truncate_head(lines[req_start - 1:req_end], max_lines, max_chars)
    first = req_start + cut.first - 1
    last = req_start + cut.last - 1
    complete = not cut.truncated and req_start == 1 and last >= total

    body = "\n".join(f"{first + i:>6}  {ln}" for i, ln in enumerate(cut.lines))
    if complete:
        head = f"# {rel}  ({total} lines)"
    else:
        head = f"# {rel}  lines {first}-{last}/{total}"
    out = f"{head}\n{body}"
    if cut.truncated:
        # Actionable, not decorative: the model's next call is written out for
        # it, because "output truncated" on its own reliably produced either a
        # blind re-read of the same range or a guess at what was missing.
        why = ("the line budget" if cut.by == "lines"
               else f"the {max_chars:,}-character budget")
        nxt = last + 1
        more = (f"read_file(path='{rel}', start={nxt})" if nxt <= req_end
                else f"read_file(path='{rel}', start={nxt}, end=…)")
        out += (f"\n\n[truncated at {why}: lines {first}-{last} of {total} "
                f"shown. Continue with {more}]")
    return ToolResult(ok=True, output=out, meta={
        "read": {"path": rel, "start": first, "end": last, "lines": total,
                 "complete": complete, "truncated": cut.truncated,
                 "chars": len(out)},
    })


def tool_read_dir(guard, path: str = "") -> ToolResult:
    """List a directory's immediate contents."""
    try:
        p = guard.resolve_within(path or ".")
    except Exception as e:
        return ToolResult(ok=False, error=str(e))
    if not p.exists():
        return ToolResult(ok=False, error=f"dir not found: {path or '.'}")
    if not p.is_dir():
        return ToolResult(ok=False, error=f"{path or '.'} is not a directory")
    entries = []
    for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
        kind = "d" if child.is_dir() else "f"
        try:
            size = child.stat().st_size if child.is_file() else 0
        except OSError:
            size = 0
        name = child.name + ("/" if child.is_dir() else "")
        entries.append(f"{kind} {size:>10,}  {name}")
    return ToolResult(ok=True,
                      output=f"# {_rel(guard, p) or '.'}\n"
                             + ("\n".join(entries) or "(empty)"))


def tool_grep(guard, pattern: str, path: str = "", glob: str = "") -> ToolResult:
    """Search text files for a regex; return repo-relative file:line hits."""
    try:
        root = guard.resolve_within(path or ".")
    except Exception as e:
        return ToolResult(ok=False, error=str(e))
    if root.is_file():
        root = root.parent
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return ToolResult(ok=False, error=f"bad regex: {e}")
    matches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for fn in sorted(filenames):
            if glob and not (fnmatch.fnmatch(fn, glob)
                             or fnmatch.fnmatch(str(Path(dirpath) / fn), glob)):
                continue
            fpath = Path(dirpath) / fn
            try:
                with io.open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            # Paths are repo-relative, not relative to the
                            # search dir, so they can be passed straight back
                            # to read_file without guesswork.
                            matches.append(
                                f"{_rel(guard, fpath)}:{i}: "
                                f"{truncate_line(line.rstrip(), 300)}")
                            if len(matches) >= 200:
                                matches.append("… (match limit reached; narrow "
                                               "the pattern or pass a path)")
                                return ToolResult(ok=True,
                                                  output="\n".join(matches))
            except OSError:
                continue
    return ToolResult(ok=True,
                      output="\n".join(matches) if matches else "(no matches)")


def tool_tree_view(guard, path: str = "", max_depth: int = 4) -> ToolResult:
    """Friendly file tree, respecting common ignore dirs."""
    try:
        root = guard.resolve_within(path or ".")
    except Exception as e:
        return ToolResult(ok=False, error=str(e))
    if not root.is_dir():
        return ToolResult(ok=False, error=f"{path or '.'} is not a directory")
    lines: list[str] = []

    def rec(p: Path, depth: int, prefix: str):
        if depth > max_depth:
            lines.append(prefix + "…")
            return
        try:
            entries = [c for c in sorted(p.iterdir(), key=lambda c: c.name.lower())
                       if c.name not in IGNORE_DIRS and not c.name.startswith(".")]
        except OSError:
            return
        for c in [c for c in entries if c.is_dir()]:
            lines.append(f"{prefix}{c.name}/")
            rec(c, depth + 1, prefix + "  ")
        for c in [c for c in entries if c.is_file()]:
            lines.append(f"{prefix}{c.name}")

    rec(root, 0, "")
    return ToolResult(ok=True, output=truncate("\n".join(lines) or "(empty)"))
