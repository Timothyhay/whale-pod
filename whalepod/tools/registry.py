"""Tool registry: JSON-schema definitions (for the API) + dispatcher.

Tool definitions are byte-stable across a session so they stay inside the
prefix-cache stable zone. The dispatcher routes a parsed tool call to its
handler with the shared sandbox guard, snapshot manager and context ledger.

Write tools are dispatched in two steps — ``plan(name, args)`` then
``commit(plan)`` — so the agent can put a confirmation between computing a
change and performing it. ``dispatch`` still works end-to-end for read tools
and for auto-approved runs.

**Usage guidance lives here, next to the tool it describes** (``guidelines=``),
and :meth:`ToolRegistry.guidelines` feeds it into the system prompt. This was
learned from the pi reference agent and fixes two things a monolithic prompt got
wrong:

  * a new tool could ship with a schema but no usage rules, because the rules
    lived in a different file;
  * a readonly session still advertised the write tools *and* still carried
    three paragraphs telling the model how to edit files. Now the tool set is
    filtered by sandbox mode and the guidance follows it automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..sandbox.guard import SandboxGuard
from ..sandbox.snapshot import SnapshotManager
from . import read as R
from . import edit as E
from .base import ToolResult
from .plan import WritePlan

WRITE_TOOL_NAMES = ("edit_file", "create_file", "delete_file", "apply_patch",
                    "run_command")


def _schema(name, description, props, required=(), guidelines=()):
    """One tool definition.

    ``guidelines`` are prompt lines, not part of the wire schema: they are
    stripped out by :meth:`ToolRegistry.schemas` and assembled into the system
    prompt instead. Kept on the definition so the two cannot drift.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props,
                           "required": list(required),
                           "additionalProperties": False},
        },
        "guidelines": list(guidelines),
    }


READ_TOOLS = [
    _schema("read_file",
            "Read a file's contents (optionally a line range). Content already "
            "shown earlier in this conversation is not re-sent — you will get a "
            "pointer instead, so re-reading an unchanged file is cheap but "
            "pointless.",
            {"path": {"type": "string", "description": "repo-relative path"},
             "start": {"type": "integer", "minimum": 1},
             "end": {"type": "integer", "minimum": 1}},
            required=["path"],
            guidelines=[
                "Look before you leap: read the relevant files before editing "
                "them. Never guess at a file's contents.",
                "A long file is truncated from the top down and tells you the "
                "range you got; continue with the `start` it suggests rather "
                "than re-reading from line 1.",
                "Content you have already been shown stays in this "
                "conversation. Re-reading an unchanged file returns only a "
                "pointer, so rely on what is above instead.",
            ]),
    _schema("read_dir",
            "List a directory's immediate contents (files + subdirs + sizes).",
            {"path": {"type": "string", "description": "dir path, default '.'"}}),
    _schema("grep",
            "Search text files with a regex; returns repo-relative "
            "file:line:match hits.",
            {"pattern": {"type": "string"}, "path": {"type": "string"},
             "glob": {"type": "string"}},
            required=["pattern"],
            guidelines=[
                "Use `grep` and `tree_view` to locate things; use `read_file` "
                "with a line range when you only need part of a large file.",
            ]),
    _schema("tree_view",
            "Show the repo file tree (respects common ignore dirs).",
            {"path": {"type": "string"}, "max_depth": {"type": "integer"}}),
    _schema("repo_map_refresh",
            "Re-scan the repo to refresh the symbol map. Costs a prefix-cache "
            "miss, so only after large external changes.",
            {}),
]

WRITE_TOOLS = [
    _schema("edit_file",
            "Replace an exact substring in a file. `old` must appear exactly "
            "once — include surrounding lines to make it unique. The diff is "
            "shown and confirmed before anything is written.",
            {"path": {"type": "string"}, "old": {"type": "string"},
             "new": {"type": "string"},
             "count": {"type": "integer",
                       "description": "replace this many occurrences "
                                      "(default 1; requires uniqueness)"}},
            required=["path", "old", "new"],
            guidelines=[
                "`edit_file` needs an `old` string that occurs exactly once, "
                "copied verbatim from the file including indentation. If it is "
                "ambiguous, include more surrounding lines.",
                "Line endings and a byte-order mark are handled for you: match "
                "the text as it was shown to you and the file's own "
                "conventions are preserved on write.",
                "Prefer several small, verifiable edits over one large "
                "speculative change.",
                "Match the style, naming and structure of the surrounding "
                "code. Do not add comments that restate what the code says.",
                "Every write is shown to the user as a diff and requires their "
                "approval. If they decline, do not retry the same edit — ask "
                "what they would prefer.",
            ]),
    _schema("create_file",
            "Create a new file with content. Fails if the file exists. "
            "Confirmed before writing.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            required=["path", "content"]),
    _schema("delete_file",
            "Delete a file. Requires confirmation.",
            {"path": {"type": "string"}}, required=["path"]),
    _schema("apply_patch",
            "Apply a unified diff to one or more files. The whole patch is "
            "verified against the current files first: if any hunk does not "
            "match, nothing is written.",
            {"patch": {"type": "string", "description": "unified diff text"}},
            required=["patch"],
            guidelines=[
                "`apply_patch` is for multi-file changes; the whole patch is "
                "verified against the files before anything is written, so it "
                "either lands completely or not at all.",
            ]),
    _schema("run_command",
            "Run a shell command in the project dir. Sensitive commands need "
            "explicit confirmation.",
            {"cmd": {"type": "string"}, "timeout": {"type": "integer"}},
            required=["cmd"],
            guidelines=[
                "`run_command` runs in the project directory. Use it for "
                "tests, builds and version control inspection. Commands with "
                "broad side effects require the user's explicit approval.",
                "Long output is cut from the front, keeping the tail where the "
                "failure usually is; the full text is spilled to a file whose "
                "path is in the result, so `grep` it rather than re-running "
                "the command.",
            ]),
]

ALL_TOOLS = READ_TOOLS + WRITE_TOOLS


class ToolRegistry:
    def __init__(self, root: Path, sandbox_mode: str = "confirm",
                 ledger=None, snapshots: Optional[SnapshotManager] = None,
                 retention=None):
        self.root = Path(root).resolve()
        self.guard = SandboxGuard(self.root, mode=sandbox_mode)
        self.snapshots = snapshots or SnapshotManager(root=self.root,
                                                     retention=retention)
        self.repo_index = None      # set by the CLI after the initial scan
        self.ledger = ledger        # ContextLedger, set by the agent
        # A readonly sandbox refuses every write, so advertising the write tools
        # only buys a round trip that ends in "sandbox is readonly". The enabled
        # set is fixed for the session's lifetime, which is what the prefix cache
        # needs; it is not re-derived per request.
        self._enabled = [t for t in ALL_TOOLS
                         if not (self.guard.is_readonly()
                                 and t["function"]["name"] in WRITE_TOOL_NAMES)]
        self._schema_cache = [
            {"type": t["type"], "function": t["function"]}
            for t in self._enabled
        ]

    # -- schemas -----------------------------------------------------------
    def schemas(self) -> list[dict]:
        """The same objects every turn — a re-worded schema is a cache miss.

        ``guidelines`` are dropped here: they belong in the system prompt, and an
        unknown key in a tool definition is a 400 on some providers.
        """
        return self._schema_cache

    def guidelines(self) -> list[str]:
        """Prompt lines contributed by the *enabled* tools, in tool order.

        Deduplicated, because two tools may legitimately want to say the same
        thing and the prompt should say it once.
        """
        out: list[str] = []
        seen: set[str] = set()
        for t in self._enabled:
            for line in t.get("guidelines") or []:
                if line not in seen:
                    seen.add(line)
                    out.append(line)
        return out

    def is_write(self, name: str) -> bool:
        return name in WRITE_TOOL_NAMES

    def known(self, name: str) -> bool:
        return name in self.tool_names()

    def tool_names(self) -> list[str]:
        """Names the model may call — the enabled set, not everything defined."""
        return [t["function"]["name"] for t in self._enabled]

    # -- planning (write tools) --------------------------------------------
    def plan(self, name: str, args: dict) -> Optional[WritePlan]:
        """Compute — but do not perform — a write. None for read tools."""
        if not self.is_write(name):
            return None
        try:
            if name == "edit_file":
                return E.plan_edit_file(self.guard, args.get("path", ""),
                                        args.get("old", ""), args.get("new", ""),
                                        int(args.get("count", 1) or 1))
            if name == "create_file":
                return E.plan_create_file(self.guard, args.get("path", ""),
                                          args.get("content", ""))
            if name == "delete_file":
                return E.plan_delete_file(self.guard, args.get("path", ""))
            if name == "apply_patch":
                return E.plan_apply_patch(self.guard, args.get("patch", ""))
            if name == "run_command":
                return E.plan_run_command(self.guard, args.get("cmd", ""),
                                          int(args.get("timeout", 60) or 60))
        except Exception as e:      # defensive: a planner must never crash the loop
            return WritePlan(tool=name,
                             error=f"{name} planning failed: "
                                   f"{type(e).__name__}: {e}")
        return None

    def commit(self, plan: WritePlan, approved: bool = True) -> ToolResult:
        """Perform a previously computed plan."""
        if plan is None:
            return ToolResult(ok=False, error="no plan to commit")
        if not plan.ok:
            return ToolResult(ok=False, error=plan.error)
        if plan.command:
            return E.run_command_now(self.guard, plan.command,
                                     getattr(plan, "timeout", 60),
                                     approved=approved,
                                     reason=getattr(plan, "audit_reason", None))
        result = plan.commit(self.snapshots)
        # Anything we wrote invalidates what the model was told the file said.
        if self.ledger is not None:
            for rel in plan.paths:
                self.ledger.invalidate(rel)
        return result

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> ToolResult:
        """Run a tool end to end (read tools, or writes already approved)."""
        if not self.known(name):
            # Covers both a hallucinated name and one that is defined but not
            # offered this session (writes in a readonly sandbox). Without this,
            # a disabled write tool still reached its planner and came back with
            # "sandbox is readonly", which reads like a fixable permission
            # problem rather than "that tool does not exist for you".
            return ToolResult(
                ok=False,
                error=f"unknown tool: {name!r}. Available: "
                      f"{', '.join(self.tool_names())}")
        g = self.guard
        try:
            if name == "read_file":
                return R.tool_read_file(g, self.repo_index, args.get("path", ""),
                                        int(args.get("start", 0) or 0),
                                        int(args.get("end", 0) or 0))
            if name == "read_dir":
                return R.tool_read_dir(g, args.get("path", ""))
            if name == "grep":
                return R.tool_grep(g, args.get("pattern", ""),
                                   args.get("path", ""), args.get("glob", ""))
            if name == "tree_view":
                return R.tool_tree_view(g, args.get("path", ""),
                                        int(args.get("max_depth", 4) or 4))
            if name == "repo_map_refresh":
                return self.refresh_repo_map()
            if self.is_write(name):
                plan = self.plan(name, args)
                return self.commit(plan)
        except Exception as e:      # defensive
            return ToolResult(ok=False,
                              error=f"{name} raised: {type(e).__name__}: {e}")
        return ToolResult(
            ok=False,
            error=f"unknown tool: {name!r}. Available: "
                  f"{', '.join(self.tool_names())}")

    # -- repo map ----------------------------------------------------------
    def refresh_repo_map(self) -> ToolResult:
        """Rescan the repo map, reusing the existing index's mtime cache."""
        from ..context.repo_map import build_repo_map
        if self.repo_index is not None:
            self.repo_index.scan()
        else:
            self.repo_index = build_repo_map(self.root)
        idx = self.repo_index
        return ToolResult(
            ok=True,
            output=(f"# refreshed repo map: {len(idx.by_file)} files, "
                    f"{idx.symbol_count} symbols"),
            # The agent re-installs the rendered map into the stable prefix.
            meta={"repo_map": True})

    def rollback(self):
        return self.snapshots.rollback()
