"""Write tools.

The tools here are thin: every one of them computes a :class:`WritePlan` (see
:mod:`whalepod.tools.plan`) and then commits it. The agent loop inserts the
confirmation between those two steps, so what the user approves is the diff
that will actually be written.

This module keeps the unified-diff machinery, which is pure and testable:

  * ``_parse_patch``  — split patch text into per-file hunks
  * ``_apply_single`` — apply hunks to one file's text, verifying every context
    and removed line, with a bounded fuzzy search for the hunk anchor because
    models routinely emit line numbers that are off by a few
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from ..sandbox.guard import SandboxGuard
from ..sandbox.snapshot import SnapshotManager
from .base import ToolResult, truncate, truncate_tail
from .plan import (
    WritePlan, plan_apply_patch, plan_create_file, plan_delete_file,
    plan_edit_file, plan_run_command,
)

# How far from the stated @@ line number we will look for the hunk's context.
FUZZ_WINDOW = 200

# Command output budget. Lines as well as characters, because 40,000 short lines
# of progress dots and one 400 KB JSON blob are both window-filling and need
# different limits to catch.
COMMAND_MAX_LINES = 400
COMMAND_MAX_CHARS = 30_000


# --------------------------------------------------------------- tools ----
def tool_edit_file(guard: SandboxGuard, snapshots: SnapshotManager,
                   path: str, old: str, new: str, count: int = 1) -> ToolResult:
    return plan_edit_file(guard, path, old, new, count).commit(snapshots)


def tool_create_file(guard: SandboxGuard, snapshots: SnapshotManager,
                     path: str, content: str) -> ToolResult:
    return plan_create_file(guard, path, content).commit(snapshots)


def tool_delete_file(guard: SandboxGuard, snapshots: SnapshotManager,
                     path: str) -> ToolResult:
    return plan_delete_file(guard, path).commit(snapshots)


def tool_apply_patch(guard: SandboxGuard, snapshots: SnapshotManager,
                     patch: str) -> ToolResult:
    return plan_apply_patch(guard, patch).commit(snapshots)


def tool_run_command(guard: SandboxGuard, cmd: str, timeout: int = 60,
                     approved: bool = False) -> ToolResult:
    """Run a shell command in the project dir.

    ``approved`` is set by the agent once the user has confirmed the command;
    without it, a command the guard flags as sensitive is refused rather than
    run. (Previously the refusal was the *only* outcome — there was no path by
    which a flagged command could ever be approved and executed.)
    """
    plan = plan_run_command(guard, cmd, timeout)
    if not plan.ok:
        return ToolResult(ok=False, error=plan.error)
    return run_command_now(guard, cmd, timeout,
                           approved=approved,
                           reason=getattr(plan, "audit_reason", None))


def run_command_now(guard: SandboxGuard, cmd: str, timeout: int = 60,
                    approved: bool = False,
                    reason: Optional[str] = None) -> ToolResult:
    # A denied command is not approvable — the check is re-done here rather than
    # trusting the caller's `approved`, because auto-approve modes ('yes',
    # 'none') never route through a confirmation at all.
    denied = guard.deny_reason(cmd)
    if denied:
        return ToolResult(ok=False, error=f"{denied} — this is refused in every "
                                          f"sandbox mode. Do a narrower "
                                          f"operation instead.")
    if reason and not approved:
        return ToolResult(ok=False,
                          error=f"refused: {reason} (command={cmd!r})")
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=str(guard.root), encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error=f"command timed out after {timeout}s")
    except OSError as e:
        return ToolResult(ok=False, error=f"could not run command: {e}")
    out = (r.stdout or "")
    if r.stderr:
        out += ("\n" if out else "") + r.stderr
    out, spilled = _bound_command_output(out.strip(), guard)
    header = f"$ {cmd}\n(exit {r.returncode})"
    meta = {"exit_code": r.returncode}
    if spilled:
        meta["full_output"] = spilled
    return ToolResult(ok=r.returncode == 0,
                      output=f"{header}\n{out}" if out else header,
                      error="" if r.returncode == 0 else f"exit {r.returncode}",
                      meta=meta)


def _bound_command_output(text: str, guard: SandboxGuard,
                          max_lines: int = COMMAND_MAX_LINES,
                          max_chars: int = COMMAND_MAX_CHARS) -> tuple[str, str]:
    """Keep the *end* of command output; spill the whole thing to a file.

    Two changes from the middle-out cut this used to do, both from the same
    observation: for a command, the information is at the bottom. The traceback,
    the assertion, the "3 failed, 291 passed" line and the exit status all live
    in the last few lines, and a middle-out cut threw exactly those away while
    keeping the first half of a build log nobody needed.

    Nothing is lost either way: when the output is cut, the full text goes to a
    file under the sandbox root and the path is handed to the model, so a 200 MB
    test log stays greppable without any of it entering the context window.
    """
    if not text:
        return "", ""
    lines = text.split("\n")
    cut = truncate_tail(lines, max_lines, max_chars)
    if not cut.truncated:
        return cut.text, ""
    path = _spill(text, guard)
    hidden = cut.total_lines - cut.kept_lines
    why = "line" if cut.by == "lines" else "size"
    note = (f"[last {cut.kept_lines} of {cut.total_lines} lines shown "
            f"({hidden} earlier lines cut at the {why} budget)")
    note += f". Full output: {path}]" if path else "]"
    return f"{note}\n{cut.text}", path


def _spill(text: str, guard: SandboxGuard) -> str:
    """Write full output somewhere the model's own tools can reach it.

    Under the sandbox root rather than the system temp dir, because the tools
    the model would use to look at it (``grep``, ``read_file``) refuse paths
    outside the root — a temp path it cannot open is worse than no path at all.
    """
    import os
    import time
    try:
        d = guard.root / ".whalepod" / "output"
        d.mkdir(parents=True, exist_ok=True)
        # Timestamp for a human scanning the directory, random suffix because two
        # commands in one millisecond would otherwise overwrite each other and
        # the model would be handed a path to the wrong log.
        p = d / (f"cmd-{int(time.time() * 1000):x}-"
                 f"{os.urandom(3).hex()}.log")
        p.write_text(text, encoding="utf-8", errors="replace", newline="")
        return str(p.relative_to(guard.root)).replace("\\", "/")
    except OSError:
        return ""


# ----------------------------------------------------------------------
# Unified-diff parsing / applying (pure Python, no external `patch` binary).
# ----------------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _strip_prelude(patch: str) -> str:
    """Normalise line endings and drop any `diff --git`/commit-message prelude."""
    patch = patch.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    idx = patch.find("--- ")
    if idx > 0:
        patch = patch[idx:]
    return patch


def _find_anchor(old_lines: list[str], lines, seek: int, min_idx: int) -> int:
    """Locate where a hunk really starts.

    The stated ``@@ -N`` is treated as a hint. We prefer it, but if the
    context/removed lines don't match there we search outward up to
    ``FUZZ_WINDOW`` lines. Returns the index to apply at, or -1 if the hunk's
    content isn't present anywhere nearby.
    """
    expected = [c for (pfx, c) in lines if pfx in (" ", "-")]
    if not expected:
        return max(seek, min_idx)

    def matches(at: int) -> bool:
        if at < min_idx or at + len(expected) > len(old_lines):
            return False
        return old_lines[at:at + len(expected)] == expected

    if matches(seek):
        return seek
    for delta in range(1, FUZZ_WINDOW + 1):
        for cand in (seek - delta, seek + delta):
            if matches(cand):
                return cand
    return -1


def _apply_single(old_text: str, hunks) -> str:
    """Apply parsed hunks to ``old_text``.

    ``hunks`` is a list of ``(old_start, old_count, lines)`` where lines are
    ``(prefix, content)`` with prefix in ``' '``, ``'+'``, ``'-'``, ``'\\'``.
    Context and removed lines are verified against the real file content; a
    mismatch raises ValueError so a stale patch is rejected rather than
    silently corrupting the file.
    """
    old_lines = old_text.splitlines()
    had_trailing_nl = old_text.endswith(("\n", "\r"))
    out: list[str] = []
    idx = 0
    no_trailing_nl = False

    for old_start, _old_count, lines in hunks:
        seek = 0 if old_start == 0 else old_start - 1
        at = _find_anchor(old_lines, lines, seek, idx)
        if at < 0:
            head = next((c for (p, c) in lines if p in (" ", "-")), "")
            raise ValueError(
                f"hunk at line {old_start} does not match the file "
                f"(looked for {head!r} within {FUZZ_WINDOW} lines). "
                f"Re-read the file and regenerate the patch.")
        if at < idx:
            raise ValueError("hunks out of order or overlapping")
        out.extend(old_lines[idx:at])
        idx = at
        for prefix, content in lines:
            if prefix in (" ", "-"):
                if idx >= len(old_lines):
                    raise ValueError(
                        f"hunk wants to consume line {idx + 1} but file ended")
                actual = old_lines[idx]
                if actual != content:
                    raise ValueError(
                        f"context mismatch at line {idx + 1}: patch has "
                        f"{content!r}, file has {actual!r}")
                if prefix == " ":
                    out.append(actual)
                idx += 1
            elif prefix == "+":
                out.append(content)
            elif prefix == "\\":          # "\ No newline at end of file"
                no_trailing_nl = True
            else:
                raise ValueError(f"unknown hunk line prefix {prefix!r}")

    out.extend(old_lines[idx:])
    result = "\n".join(out)
    if had_trailing_nl and not no_trailing_nl and not result.endswith("\n"):
        result += "\n"
    return result


def _parse_patch(patch: str):
    """Split patch text into per-file sections.

    Returns a list of ``{old_file, new_file, hunks}``, where hunks is a list of
    ``(old_start, old_count, [(prefix, text), ...])``.
    """
    lines = patch.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    files: list[dict] = []
    cur = None
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("--- "):
            old_header = line[4:].strip()
            new_header = ""
            if i + 1 < n and lines[i + 1].startswith("+++ "):
                new_header = lines[i + 1][4:].strip()
                i += 1
            if cur is not None:
                files.append(cur)
            cur = {"old_file": _clean_header(old_header),
                   "new_file": _clean_header(new_header),
                   "hunks": []}
            i += 1
            continue
        if line.startswith("@@ ") and cur is not None:
            m = _HUNK_RE.match(line)
            if not m:
                raise ValueError(f"bad hunk header: {line!r}")
            old_start = int(m.group(1))
            old_count = int(m.group(2) or 1)
            body: list[tuple[str, str]] = []
            i += 1
            while i < n and not lines[i].startswith(("@@ ", "--- ", "diff ")):
                bl = lines[i]
                if not bl:
                    body.append((" ", ""))
                elif bl[0] in (" ", "+", "-", "\\"):
                    body.append((bl[0], bl[1:]))
                else:
                    # Tolerate a stripped leading space on a context line.
                    body.append((" ", bl))
                i += 1
            cur["hunks"].append((old_start, old_count, body))
            continue
        i += 1
    if cur is not None:
        files.append(cur)
    return files


def _clean_header(h: str) -> str:
    """``--- a/foo.py`` → ``foo.py``; ``/dev/null`` → ``""``."""
    if not h or h == "/dev/null":
        return ""
    h = h.split("\t")[0]
    for p in ("a/", "b/", "orig/", "new/"):
        if h.startswith(p):
            h = h[len(p):]
            break
    return h.strip()


__all__ = [
    "tool_edit_file", "tool_create_file", "tool_delete_file",
    "tool_apply_patch", "tool_run_command", "run_command_now",
    "WritePlan", "plan_edit_file", "plan_create_file", "plan_delete_file",
    "plan_apply_patch", "plan_run_command",
    "_parse_patch", "_apply_single", "_strip_prelude",
]
