"""Write planning — compute the change, show it, *then* commit it.

The original write tools wrote the file first and produced the diff afterwards,
which made confirmation theatre: by the time the user saw "apply this?", the
bytes were already on disk and answering "no" changed nothing. Multi-file
patches were worse — a patch that failed on its third file left the first two
applied.

So every write tool is split in two halves:

    plan_*(guard, ...) -> WritePlan     pure: reads, validates, computes the
                                        new content and a diff. Touches nothing.
    WritePlan.commit(snapshots)         snapshots and writes, all-or-nothing;
                                        if any file fails, the files already
                                        written in this plan are rolled back.

The confirmation the UI shows is ``plan.diff``, which is now literally the
change that will be made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import textfile
from ..sandbox.guard import SandboxGuard, diff_stat, make_unified_diff
from ..sandbox.snapshot import SnapshotManager
from .base import ToolResult, truncate

MAX_DIFF_PREVIEW = 20_000


@dataclass
class FileChange:
    """One file's before/after. ``after is None`` means delete.

    ``before``/``after`` are always LF-normalised and BOM-free so that diffs,
    matching and hunk application all work on one representation. ``bom`` and
    ``newline`` remember the file's own conventions so :func:`_write_text` can
    put them back — otherwise editing one line of a CRLF file rewrote every
    line ending in it.
    """
    path: Path
    rel: str
    before: Optional[str]        # None => file did not exist
    after: Optional[str]         # None => delete the file
    diff: str = ""
    bom: str = ""
    newline: str = "\n"

    @property
    def kind(self) -> str:
        if self.after is None:
            return "delete"
        if self.before is None:
            return "create"
        return "edit"


@dataclass
class WritePlan:
    """A validated, not-yet-applied change set."""
    tool: str
    changes: list[FileChange] = field(default_factory=list)
    error: str = ""
    command: str = ""            # run_command only
    note: str = ""               # extra context for the confirmation prompt

    # -- properties used by the UI / agent --------------------------------
    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def paths(self) -> list[str]:
        return [c.rel for c in self.changes]

    @property
    def diff(self) -> str:
        return "\n".join(c.diff for c in self.changes if c.diff)

    def summary(self) -> str:
        """One line for the confirmation header."""
        if self.error:
            return f"{self.tool}: {self.error}"
        if self.command:
            return f"{self.tool}: {self.command}"
        if not self.changes:
            return f"{self.tool}: (no change)"
        parts = []
        for c in self.changes:
            add, rem = diff_stat(c.diff)
            parts.append(f"{c.kind} {c.rel} (+{add} -{rem})")
        return f"{self.tool}: " + ", ".join(parts)

    def preview(self) -> str:
        body = self.diff
        if self.note:
            body = f"{self.note}\n{body}" if body else self.note
        return truncate(f"{self.summary()}\n{body}", MAX_DIFF_PREVIEW)

    def is_noop(self) -> bool:
        return self.ok and not self.command and all(
            c.before == c.after for c in self.changes)

    # -- commit ------------------------------------------------------------
    def commit(self, snapshots: SnapshotManager) -> ToolResult:
        """Apply the plan. All-or-nothing across files."""
        if self.error:
            return ToolResult(ok=False, error=self.error)
        if not self.changes:
            return ToolResult(ok=True, output=f"# {self.tool}: nothing to do")

        written: list[Path] = []
        for c in self.changes:
            snapshots.snapshot_file(c.path, tool=self.tool)
            try:
                if c.after is None:
                    if c.path.exists():
                        c.path.unlink()
                else:
                    c.path.parent.mkdir(parents=True, exist_ok=True)
                    _write_text(c.path, c.after, bom=c.bom, newline=c.newline)
                written.append(c.path)
            except OSError as e:
                # Undo this plan's writes so a multi-file patch can never land
                # halfway.
                undone = snapshots.rollback_paths(written)
                return ToolResult(
                    ok=False,
                    error=(f"{c.rel}: write failed: {e}; rolled back "
                           f"{len(undone)} file(s) from this change"),
                )
        lines = [f"# {self.tool}: {self.summary().split(': ', 1)[-1]}"]
        for c in self.changes:
            lines.append(c.diff or f"(no textual diff for {c.rel})")
        return ToolResult(ok=True, output=truncate("\n".join(lines)),
                          meta={"paths": self.paths, "tool": self.tool})


def _write_text(path: Path, text: str, bom: str = "",
                newline: str = "\n") -> None:
    """Write via a temp file + atomic replace, in the file's own conventions.

    ``newline=""`` on the handle stops Python from translating "\\n" to "\\r\\n"
    on Windows; the translation we *do* want is explicit, from the line ending
    the file already used. Without both halves, a one-line edit rewrote every
    line ending in the file and turned into a whole-file diff for the user's
    next `git diff`.
    """
    body = text.replace("\r\n", "\n").replace("\r", "\n")
    if newline != "\n":
        body = body.replace("\n", newline)
    tmp = path.with_name(path.name + ".whalepod.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(bom + body)
    import os
    os.replace(tmp, path)


def _fail(tool: str, msg: str) -> WritePlan:
    return WritePlan(tool=tool, error=msg)


def _read(path: Path) -> tuple[Optional[str], Optional[str]]:
    """(LF-normalised BOM-free text, error).

    Normalised because the model's ``old`` string comes from ``read_file``'s
    output, which is LF-joined and BOM-free; matching raw CRLF bytes against it
    could never succeed. The original conventions travel separately on the
    :class:`FileChange` so the write puts them back.
    """
    tf, err = textfile.read(path)
    if err or tf is None:
        return None, err or "file is not utf-8 text"
    return tf.text, None


def _conventions(path: Path) -> tuple[str, str]:
    """(bom, newline) of an existing file; ("", "\\n") if unreadable."""
    tf, err = textfile.read(path)
    if err or tf is None:
        return "", "\n"
    return tf.bom, tf.newline


def _prepare(guard: SandboxGuard, tool: str, path: str):
    """Shared sandbox checks. Returns (resolved_path, rel, error_plan)."""
    try:
        guard.assert_can_write(tool)
        p = guard.resolve_within(path)
    except Exception as e:
        return None, "", _fail(tool, str(e))
    try:
        rel = str(p.relative_to(guard.root)).replace("\\", "/")
    except ValueError:
        rel = str(p)
    return p, rel, None


# ------------------------------------------------------------------ planners
def plan_edit_file(guard: SandboxGuard, path: str, old: str, new: str,
                   count: int = 1) -> WritePlan:
    """Replace an exact substring.

    ``old`` must be unique unless ``count`` says otherwise: a snippet that
    appears twice is an ambiguous instruction, and silently editing the first
    occurrence is the kind of quiet wrong answer that is worst to debug.
    """
    p, rel, err = _prepare(guard, "edit_file", path)
    if err:
        return err
    if not p.exists():
        return _fail("edit_file", f"file not found: {rel}")
    if p.is_dir():
        return _fail("edit_file", f"{rel} is a directory")
    text, rerr = _read(p)
    if rerr:
        return _fail("edit_file", f"{rel}: {rerr}")
    if not old:
        return _fail("edit_file", "`old` is empty; use create_file to write a "
                                  "whole file")
    # The needle is normalised the same way the haystack is: a model that echoes
    # CRLF back (some do, when the text came from a Windows terminal) should not
    # get "not found" for text that is plainly there.
    old = old.replace("\r\n", "\n").replace("\r", "\n")
    new = new.replace("\r\n", "\n").replace("\r", "\n")
    hits = text.count(old)
    if hits == 0:
        return _fail("edit_file",
                     f"`old` text not found in {rel}. Read the file again — it "
                     f"must match exactly, including indentation.")
    if hits > 1 and count == 1:
        return _fail("edit_file",
                     f"`old` text appears {hits} times in {rel}; include more "
                     f"surrounding lines to make it unique (or pass "
                     f"count={hits} to replace them all).")
    new_text = text.replace(old, new) if count != 1 else text.replace(old, new, 1)
    diff = make_unified_diff(p, text, new_text, label=rel)
    bom, newline = _conventions(p)
    return WritePlan(tool="edit_file", changes=[
        FileChange(path=p, rel=rel, before=text, after=new_text, diff=diff,
                   bom=bom, newline=newline)])


def plan_create_file(guard: SandboxGuard, path: str, content: str) -> WritePlan:
    p, rel, err = _prepare(guard, "create_file", path)
    if err:
        return err
    if p.is_dir():
        return _fail("create_file", f"{rel} is a directory")
    before = None
    if p.exists():
        return _fail("create_file",
                     f"file already exists: {rel} (use edit_file, or "
                     f"delete_file first)")
    diff = make_unified_diff(p, "", content, label=rel)
    return WritePlan(tool="create_file", changes=[
        FileChange(path=p, rel=rel, before=before, after=content, diff=diff)])


def plan_delete_file(guard: SandboxGuard, path: str) -> WritePlan:
    p, rel, err = _prepare(guard, "delete_file", path)
    if err:
        return err
    if not p.exists():
        return _fail("delete_file", f"file not found: {rel}")
    if p.is_dir():
        return _fail("delete_file", f"{rel} is a directory")
    text, rerr = _read(p)
    lines = 0 if text is None else len(text.splitlines())
    return WritePlan(
        tool="delete_file",
        note=f"DELETE {rel} ({lines} lines){' [binary]' if rerr else ''}",
        changes=[FileChange(path=p, rel=rel, before=text, after=None,
                            diff=f"--- a/{rel}\n+++ /dev/null")])


def plan_apply_patch(guard: SandboxGuard, patch: str) -> WritePlan:
    """Parse and *simulate* a unified diff across all its files.

    Every hunk is verified against the current file content before anything is
    written, so a patch whose third file is stale fails as a whole instead of
    leaving the first two applied.
    """
    from .edit import _apply_single, _parse_patch, _strip_prelude

    try:
        guard.assert_can_write("apply_patch")
    except Exception as e:
        return _fail("apply_patch", str(e))
    if not patch.strip():
        return _fail("apply_patch", "empty patch")
    try:
        sections = _parse_patch(_strip_prelude(patch))
    except ValueError as e:
        return _fail("apply_patch", f"bad patch: {e}")
    if not sections:
        return _fail("apply_patch", "no file sections found in patch")

    changes: list[FileChange] = []
    for sec in sections:
        target_rel = sec["new_file"] or sec["old_file"]
        if not target_rel:
            continue
        p, rel, err = _prepare(guard, "apply_patch", target_rel)
        if err:
            return _fail("apply_patch", f"{target_rel}: {err.error}")
        existed = p.exists()
        is_deletion = bool(sec["old_file"]) and not sec["new_file"]
        before: Optional[str] = None
        if existed:
            before, rerr = _read(p)
            if rerr:
                return _fail("apply_patch", f"{rel}: {rerr}")
        if is_deletion:
            if not existed:
                return _fail("apply_patch",
                             f"{rel}: patch deletes a file that does not exist")
            changes.append(FileChange(path=p, rel=rel, before=before, after=None,
                                      diff=f"--- a/{rel}\n+++ /dev/null"))
            continue
        if not sec["hunks"]:
            continue
        try:
            after = _apply_single(before or "", sec["hunks"])
        except ValueError as e:
            return _fail("apply_patch", f"{rel}: {e}")
        bom, newline = _conventions(p) if existed else ("", "\n")
        changes.append(FileChange(
            path=p, rel=rel, before=before, after=after,
            diff=make_unified_diff(p, before or "", after, label=rel),
            bom=bom, newline=newline))

    if not changes:
        return _fail("apply_patch", "patch contained no applicable changes")
    return WritePlan(tool="apply_patch", changes=changes)


def plan_run_command(guard: SandboxGuard, cmd: str,
                     timeout: int = 60) -> WritePlan:
    """A command has no diff, but it still gets a plan so the agent path is
    uniform and the user always sees what is about to run."""
    if not cmd.strip():
        return _fail("run_command", "empty command")
    if guard.is_readonly():
        return _fail("run_command", "sandbox is readonly; commands are blocked")
    denied = guard.deny_reason(cmd)
    if denied:
        # Fail at plan time so the user is never asked to approve something that
        # would be refused after they said yes.
        return _fail("run_command", denied)
    reason = guard.audit_command(cmd)
    note = f"$ {cmd}" + (f"\n! {reason}" if reason else "")
    plan = WritePlan(tool="run_command", command=cmd, note=note)
    plan.timeout = timeout          # type: ignore[attr-defined]
    plan.audit_reason = reason      # type: ignore[attr-defined]
    return plan
