"""Filesystem sandbox + write-confirmation guard.

Modes:
  confirm   (default) writes require a diff preview + Yes/No/Edit decision
  readonly  all writes refused
  yes       auto-approve writes (CI/automation; must be explicit)
  none      no restrictions (not recommended)

Path access is restricted to a whitelist rooted at the project cwd (plus any
explicitly allowed roots).

Command classification
    A command line is split on shell operators (``;`` ``&&`` ``||`` ``|`` and
    newlines) and *every* segment is classified. Matching only the start of the
    whole string — as this used to — meant ``echo hi && rm -rf build`` and
    ``true; sudo pip install x`` were both classified as harmless ``echo``/
    ``true``, so the sensitive-command confirmation could be bypassed by
    prefixing anything benign.

    Two tiers:
      * *sensitive* — needs an explicit user OK in ``confirm`` mode.
      * *denied* — refused in every mode, including ``yes``/``none``. These are
        commands with no plausible place in an automated coding session
        (wiping a filesystem root, ``mkfs``, raw ``dd`` to a device). Without
        this tier, ``--yes`` in CI would happily approve them.
"""
from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Optional

# Programs whose side effects are too broad to run without extra confirmation.
_SENSITIVE_PROGRAMS = {
    "rm", "rmdir", "rd", "del", "erase", "shred", "truncate",
    "curl", "wget", "nc", "ncat", "ssh", "scp", "rsync", "ftp",
    "chmod", "chown", "chgrp", "icacls", "takeown", "attrib",
    "dd", "mkfs", "fdisk", "diskpart", "format", "mount", "umount",
    "sudo", "su", "runas", "doas",
    "kill", "killall", "taskkill", "pkill", "shutdown", "reboot",
    "systemctl", "service", "sc", "reg", "regedit", "crontab", "schtasks",
    "docker", "kubectl", "terraform", "helm",
    "eval", "exec", "source",
}

# (program, first-arg substrings) pairs that are sensitive only in some forms.
_SENSITIVE_SUBCOMMANDS = {
    "git": ("push", "reset", "clean", "checkout", "restore", "rebase",
            "filter-branch", "gc", "worktree", "remote"),
    "pip": ("install", "uninstall"),
    "pip3": ("install", "uninstall"),
    "npm": ("install", "i", "uninstall", "publish", "link", "run"),
    "pnpm": ("install", "add", "remove", "publish"),
    "yarn": ("add", "remove", "publish"),
    "cargo": ("install", "publish"),
    "go": ("install", "get"),
    "uv": ("pip", "add", "remove", "sync"),
    "poetry": ("add", "remove", "publish", "install"),
    "apt": ("install", "remove", "purge"),
    "apt-get": ("install", "remove", "purge"),
    "brew": ("install", "uninstall"),
    "winget": ("install", "uninstall"),
    "choco": ("install", "uninstall"),
}

# Never run these, in any mode — not even ``--yes``. Each is checked against a
# single normalized segment, so a benign prefix cannot hide them.
_DENY_PATTERNS = (
    (r"\bmkfs(\.\w+)?\b", "filesystem creation"),
    (r"\bdd\b.*\bof=\s*/dev/", "raw write to a device"),
    (r"\bdiskpart\b", "disk partitioning"),
    (r"\bformat\s+[a-zA-Z]:", "disk formatting"),
    # Any self-referential function that pipes itself into the background and
    # then calls itself — not just the classic `:(){ :|:& };:` spelling.
    (r"(?P<fn>[\w:.-]+)\s*\(\s*\)\s*\{[^}]*\|[^}]*&[^}]*\}\s*;?\s*(?P=fn)",
     "fork bomb"),
    (r">\s*/dev/(sd|nvme|hd|disk)", "raw write to a device"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "changes host power state"),
    (r"\bgit\s+push\b.*--force(-with-lease)?(?<!-with-lease)\s*$",
     "force push (rewrites published history)"),
)

# Locations nothing in a coding session should be pointed at. Deliberately does
# NOT include "." or "*" — plenty of harmless commands take those as arguments;
# recursive deletion of the cwd is caught by _recursive_delete_target instead.
_PROTECTED_TARGETS = ("/", "/*", "~", "~/", "~/*",
                      "c:", "c:\\", "c:/", "c:\\*", "/etc", "/usr", "/var",
                      "/home", "/root", "/boot", "/system32", "%windir%",
                      "%systemroot%", "$home", "$env:userprofile")

# Recursive deletion of one of these erases the working tree itself.
_SELF_DESTRUCT = ("", ".", "..", "*", "./", "./*", "../", "*/")

# "Write" intentions the guard classifies
WRITE_OPS = ("edit_file", "create_file", "apply_patch", "delete_file",
             "run_command")


class SandboxError(Exception):
    pass


class SandboxGuard:
    def __init__(self, root: Path, mode: str = "confirm",
                 allow: Optional[list[str]] = None):
        self.root = Path(root).resolve()
        self.mode = mode
        roots = [self.root]
        for a in allow or []:
            roots.append(Path(a).resolve())
        self.allowed = roots

    # -- path enforcement -------------------------------------------------
    def resolve_within(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        p = p.resolve()
        for root in self.allowed:
            try:
                p.relative_to(root)
                return p
            except ValueError:
                continue
        raise SandboxError(
            f"path {path!r} resolves outside allowed roots "
            f"({[str(r) for r in self.allowed]})"
        )

    # -- mode checks --------------------------------------------------------
    def assert_can_write(self, op: str) -> None:
        if self.mode == "readonly":
            raise SandboxError("sandbox is readonly; cannot perform write op " + op)
        if self.mode == "none":
            return
        # confirm / yes both allowed at this layer; 'yes' is handled by caller

    def is_readonly(self) -> bool:
        return self.mode == "readonly"

    def auto_approve(self) -> bool:
        return self.mode in ("yes", "none")

    # -- command classification ---------------------------------------------
    def is_sensitive_command(self, cmdline: str) -> bool:
        return self.classify_command(cmdline)[0] is not None

    def deny_reason(self, cmdline: str) -> Optional[str]:
        """Why this command is refused outright, or None.

        Checked in *every* mode. ``--yes`` means "don't ask me about edits",
        not "run anything at all".

        The whole line is matched **as well as** each segment. Segment matching
        alone is not enough for patterns that span shell operators: a fork bomb
        (``:(){ :|:& };:``) is built out of ``|``, ``&`` and ``;``, so splitting
        first shredded it into four harmless-looking segments and it ran. Found
        by ``bench/validate.py``'s subprocess tripwire.
        """
        for pattern, why in _DENY_PATTERNS:
            if re.search(pattern, cmdline, re.IGNORECASE):
                return f"refused: {why} (`{cmdline.strip()[:80]}`)"
        for segment in split_command(cmdline):
            for pattern, why in _DENY_PATTERNS:
                if re.search(pattern, segment, re.IGNORECASE):
                    return f"refused: {why} (`{segment.strip()[:80]}`)"
            target = _recursive_delete_target(segment)
            if target is not None:
                return (f"refused: recursive delete of a protected location "
                        f"({target!r})")
        return None

    def classify_command(self, cmdline: str) -> tuple[Optional[str], str]:
        """(reason, segment) for the first sensitive segment, else (None, "")."""
        for segment in split_command(cmdline):
            argv = _tokenize(segment)
            if not argv:
                continue
            prog = _program_name(argv[0])
            if prog in _SENSITIVE_PROGRAMS:
                return f"`{prog}` has broad side effects", segment
            subs = _SENSITIVE_SUBCOMMANDS.get(prog)
            if subs:
                sub = next((a for a in argv[1:] if not a.startswith("-")), "")
                if sub in subs:
                    return f"`{prog} {sub}` changes state outside this repo", segment
            if any(a in _PROTECTED_TARGETS for a in argv[1:]):
                return f"`{prog}` targets a protected location", segment
        return None, ""

    def audit_command(self, cmdline: str) -> Optional[str]:
        """None if the command may run unattended; else why it must be asked.

        A denial is returned here too, so the single call site cannot forget to
        check it — ``run_command_now`` refuses anything with a reason unless it
        was explicitly approved, and a denial is never approvable.
        """
        denied = self.deny_reason(cmdline)
        if denied:
            return denied
        if self.mode == "readonly":
            return "sandbox is readonly"
        if self.mode != "confirm":
            return None
        reason, segment = self.classify_command(cmdline)
        if reason:
            return f"{reason}: `{segment.strip()[:80]}`"
        return None

    def is_denied(self, cmdline: str) -> bool:
        return self.deny_reason(cmdline) is not None


# ------------------------------------------------------- command tokenizing
def split_command(cmdline: str) -> list[str]:
    """Split a command line into independently-executed segments.

    Quoted operators are left alone: ``echo "a && b"`` is one segment, because
    splitting inside the quotes would invent a command the shell never runs.
    """
    out: list[str] = []
    buf: list[str] = []
    quote: Optional[str] = None
    i = 0
    while i < len(cmdline):
        ch = cmdline[i]
        if quote:
            buf.append(ch)
            if ch == quote and cmdline[i - 1: i] != "\\":
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            i += 1
            continue
        two = cmdline[i:i + 2]
        if two in ("&&", "||"):
            out.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in ";|\n&":
            out.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    out.append("".join(buf))
    return [s for s in (seg.strip() for seg in out) if s]


def _tokenize(segment: str) -> list[str]:
    """Split into words, keeping Windows paths intact.

    ``posix=False`` matters: in posix mode shlex treats ``\\`` as an escape, so
    ``C:\\Windows\\System32\\del.exe`` collapses to ``C:WindowsSystem32del.exe``
    and the program name no longer matches anything. Quotes are stripped by hand
    instead.

    A malformed command must still be *classified* — returning nothing on a
    ValueError would let ``rm -rf / "`` through as "no tokens, nothing to see".
    """
    try:
        raw = shlex.split(segment, posix=False)
    except ValueError:
        raw = segment.replace('"', " ").replace("'", " ").split()
    out = []
    for tok in raw:
        if len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "\"'":
            tok = tok[1:-1]
        if tok:
            out.append(tok)
    return out


def _program_name(token: str) -> str:
    """Strip any path and extension: ``/usr/bin/sudo`` and ``sudo.exe`` -> sudo."""
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for ext in (".exe", ".cmd", ".bat", ".ps1", ".com"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    # `VAR=x rm ...` — the assignment is not the program.
    return name


def _recursive_delete_target(segment: str) -> Optional[str]:
    """The protected path a recursive delete in this segment would destroy."""
    argv = _tokenize(segment)
    if not argv:
        return None
    prog = _program_name(argv[0])
    if prog not in ("rm", "rmdir", "rd", "del", "erase", "remove-item"):
        return None
    flags = [a for a in argv[1:] if a.startswith("-")]
    recursive = any(c in "rR" for f in flags for c in f[1:]) or "-Recurse" in argv
    targets = [a for a in argv[1:] if not a.startswith("-")]
    for t in targets:
        low = t.lower()
        norm = low.rstrip("/\\") or low
        if norm in _PROTECTED_TARGETS or low in _PROTECTED_TARGETS:
            return t
        if recursive and (low in _SELF_DESTRUCT or norm in _SELF_DESTRUCT):
            return t
    return None


# ------------------------------------------------------------------ diff
def make_unified_diff(path: Optional[Path], old_text: str, new_text: str,
                      label: Optional[str] = None, context: int = 3) -> str:
    """Produce a compact unified diff for preview, without requiring git.

    Lines are split *without* keepends and joined with newlines: mixing
    ``keepends=True`` with ``lineterm=""`` (as this used to) leaves the ``---``
    and ``@@`` headers without a line ending, so they get glued onto the
    following line in the preview the user is asked to approve.
    """
    import difflib
    label = label or (str(path) if path else "file")
    diff = difflib.unified_diff(
        old_text.splitlines(), new_text.splitlines(),
        fromfile=f"a/{label}", tofile=f"b/{label}",
        lineterm="", n=context,
    )
    return "\n".join(diff)


def diff_stat(diff: str) -> tuple[int, int]:
    """(added, removed) line counts from a unified diff."""
    add = rem = 0
    for ln in diff.splitlines():
        if ln.startswith("+") and not ln.startswith("+++"):
            add += 1
        elif ln.startswith("-") and not ln.startswith("---"):
            rem += 1
    return add, rem
