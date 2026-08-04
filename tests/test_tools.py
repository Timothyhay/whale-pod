"""Sandbox, planning, write tools, snapshots and rollback.

Everything here runs against a scratch directory, and ``WHALEPOD_BACKUP_DIR``
is redirected so the suite never touches the user's real ``~/.whalepod``.
"""
import os
import tempfile
import unittest
from pathlib import Path

from whalepod.sandbox.guard import (
    SandboxGuard, diff_stat, make_unified_diff, split_command,
)
from whalepod.sandbox.snapshot import SnapshotManager
from whalepod.tools.plan import (
    plan_apply_patch, plan_create_file, plan_delete_file, plan_edit_file,
)
from whalepod.tools.registry import ToolRegistry


class ScratchCase(unittest.TestCase):
    """A temp project root, with snapshots kept inside it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.backups = self.root / ".backups"
        self._prev = os.environ.get("WHALEPOD_BACKUP_DIR")
        os.environ["WHALEPOD_BACKUP_DIR"] = str(self.backups)

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("WHALEPOD_BACKUP_DIR", None)
        else:
            os.environ["WHALEPOD_BACKUP_DIR"] = self._prev
        self._tmp.cleanup()

    def write(self, rel: str, text: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="")
        return p

    def read(self, rel: str) -> str:
        return (self.root / rel).read_text(encoding="utf-8")

    def registry(self, mode="yes") -> ToolRegistry:
        return ToolRegistry(self.root, sandbox_mode=mode)


# --------------------------------------------------------------- sandbox
class TestSandboxPaths(ScratchCase):
    def test_path_escape_blocked(self):
        guard = SandboxGuard(self.root, mode="confirm")
        with self.assertRaises(Exception):
            guard.resolve_within("../escape")

    def test_absolute_path_outside_root_blocked(self):
        guard = SandboxGuard(self.root, mode="confirm")
        with self.assertRaises(Exception):
            guard.resolve_within(str(Path(tempfile.gettempdir()) / "elsewhere"))

    def test_readonly_refuses_writes(self):
        guard = SandboxGuard(self.root, mode="readonly")
        with self.assertRaises(Exception):
            guard.assert_can_write("edit_file")


class TestCommandClassification(unittest.TestCase):
    """Regression: only the start of the whole command line was inspected, so a
    benign prefix hid anything after ``&&`` or ``;``."""

    def setUp(self):
        self.guard = SandboxGuard(Path("."), mode="confirm")

    def test_splits_on_shell_operators(self):
        self.assertEqual(split_command("ls && rm -r build; echo done | wc -l"),
                         ["ls", "rm -r build", "echo done", "wc -l"])

    def test_quoted_operators_are_not_split(self):
        self.assertEqual(split_command('echo "a && b"'), ['echo "a && b"'])

    def test_sensitive_command_hidden_behind_a_benign_prefix(self):
        for cmd in ("echo hi && rm -r build",
                    "true; sudo pip install requests",
                    "ls | sudo tee /etc/hosts",
                    "cat x && git push origin main"):
            with self.subTest(cmd=cmd):
                self.assertTrue(self.guard.is_sensitive_command(cmd), cmd)
                self.assertIsNotNone(self.guard.audit_command(cmd))

    def test_ordinary_commands_are_not_flagged(self):
        for cmd in ("ls -la", "python -m pytest -q", "git status",
                    "git log --oneline -5", "grep -r TODO .", "echo done"):
            with self.subTest(cmd=cmd):
                self.assertIsNone(self.guard.audit_command(cmd), cmd)

    def test_program_path_and_extension_do_not_hide_it(self):
        self.assertTrue(self.guard.is_sensitive_command("/usr/bin/sudo apt install x"))
        self.assertTrue(self.guard.is_sensitive_command("C:\\Windows\\System32\\del.exe f"))

    def test_broken_quoting_is_still_classified(self):
        self.assertTrue(self.guard.is_sensitive_command('rm -rf build "'))

    def test_deny_list_applies_in_every_mode(self):
        for mode in ("confirm", "yes", "none"):
            guard = SandboxGuard(Path("."), mode=mode)
            for cmd in ("mkfs.ext4 /dev/sda1", "rm -rf /", "echo x && rm -rf ~",
                        "dd if=/dev/zero of=/dev/sda", "shutdown -h now"):
                with self.subTest(mode=mode, cmd=cmd):
                    self.assertTrue(guard.is_denied(cmd), cmd)

    def test_deny_patterns_that_span_shell_operators(self):
        """A fork bomb is *made of* `|`, `&` and `;`.

        Matching only per-segment shredded `:(){ :|:& };:` into four harmless
        fragments and let it through to subprocess — caught by the bench's
        tripwire, so the whole line is now matched as well as each segment.
        """
        for mode in ("confirm", "yes", "none"):
            guard = SandboxGuard(Path("."), mode=mode)
            for cmd in (":(){ :|:& };:",
                        ":() { :|:& }; :",
                        "bomb(){ bomb|bomb& };bomb",
                        "echo hi; :(){ :|:& };:"):
                with self.subTest(mode=mode, cmd=cmd):
                    self.assertTrue(guard.is_denied(cmd), cmd)

    def test_function_definitions_are_not_all_fork_bombs(self):
        for cmd in ("build(){ make -j4; }; build",
                    "t(){ pytest -q; }; t",
                    "f(){ ls | wc -l; }; f"):
            with self.subTest(cmd=cmd):
                self.assertFalse(self.guard.is_denied(cmd), cmd)

    def test_deny_list_does_not_swallow_normal_deletes(self):
        for cmd in ("rm -rf build", "rm -rf ./dist", "rm f.txt"):
            with self.subTest(cmd=cmd):
                self.assertFalse(self.guard.is_denied(cmd), cmd)
                # still sensitive: it needs a confirmation, just not a refusal
                self.assertIsNotNone(self.guard.audit_command(cmd))

    def test_recursive_delete_of_the_worktree_is_denied(self):
        for cmd in ("rm -rf .", "rm -rf *", "rm -fr ./"):
            with self.subTest(cmd=cmd):
                self.assertTrue(self.guard.is_denied(cmd), cmd)

    def test_denied_command_is_refused_even_when_approved(self):
        from whalepod.tools.edit import run_command_now
        r = run_command_now(SandboxGuard(Path("."), mode="none"),
                            "mkfs.ext4 /dev/sda1", approved=True)
        self.assertFalse(r.ok)
        self.assertIn("refused", r.error)

    def test_flagged_command_can_be_approved(self):
        """It used to be impossible to ever run a flagged command."""
        from whalepod.tools.edit import run_command_now
        with tempfile.TemporaryDirectory() as d:
            guard = SandboxGuard(Path(d), mode="confirm")
            reason = guard.audit_command("git status")
            self.assertIsNone(reason)
            r = run_command_now(guard, "git status", approved=True,
                                reason="pretend it was flagged")
            self.assertIsNotNone(r)     # ran (or failed on git), but not refused
            self.assertNotIn("refused", r.error or "")


# ------------------------------------------------------------------ diff
class TestDiff(unittest.TestCase):
    def test_unified_diff_contains_marks(self):
        d = make_unified_diff(Path("a.py"), "hello\nworld\n", "hello\nmoon\n")
        self.assertIn("+moon", d)
        self.assertIn("-world", d)

    def test_headers_are_on_their_own_lines(self):
        """Regression: keepends=True with lineterm="" glued ``---``/``@@`` onto
        the following line, in the very preview the user approves."""
        d = make_unified_diff(Path("a.py"), "a\nb\n", "a\nc\n")
        lines = d.splitlines()
        self.assertTrue(lines[0].startswith("--- a/"))
        self.assertTrue(lines[1].startswith("+++ b/"))
        self.assertTrue(lines[2].startswith("@@"))
        for ln in lines:
            self.assertNotIn("@@ ---", ln)
            self.assertFalse(ln.startswith("@@ -1,2 +1,2 @@ "), ln)

    def test_diff_stat(self):
        d = make_unified_diff(Path("a.py"), "a\nb\nc\n", "a\nB\nc\nd\n")
        self.assertEqual(diff_stat(d), (2, 1))


# -------------------------------------------------------------- planning
class TestPlanning(ScratchCase):
    """Plans must compute the change without touching the filesystem."""

    def test_plan_edit_does_not_write(self):
        self.write("f.py", "x = 1\n")
        guard = SandboxGuard(self.root, mode="confirm")
        plan = plan_edit_file(guard, "f.py", "x = 1", "x = 2")
        self.assertTrue(plan.ok, plan.error)
        self.assertIn("+x = 2", plan.diff)
        self.assertEqual(self.read("f.py"), "x = 1\n")     # untouched

    def test_declining_a_plan_leaves_the_file_alone(self):
        """The whole point of the split: 'no' now actually means no."""
        self.write("f.py", "x = 1\n")
        reg = self.registry("confirm")
        plan = reg.plan("edit_file", {"path": "f.py", "old": "x = 1",
                                      "new": "x = 2"})
        self.assertTrue(plan.ok)
        # ...user says no => commit is never called
        self.assertEqual(self.read("f.py"), "x = 1\n")

    def test_plan_diff_matches_what_commit_writes(self):
        self.write("f.py", "a\nb\nc\n")
        reg = self.registry()
        plan = reg.plan("edit_file", {"path": "f.py", "old": "b", "new": "B"})
        promised = plan.changes[0].after
        reg.commit(plan)
        self.assertEqual(self.read("f.py"), promised)

    def test_ambiguous_edit_is_refused(self):
        self.write("f.py", "dup\ndup\n")
        guard = SandboxGuard(self.root, mode="confirm")
        plan = plan_edit_file(guard, "f.py", "dup", "x")
        self.assertFalse(plan.ok)
        self.assertIn("appears 2 times", plan.error)

    def test_count_allows_replacing_all(self):
        self.write("f.py", "dup\ndup\n")
        guard = SandboxGuard(self.root, mode="confirm")
        plan = plan_edit_file(guard, "f.py", "dup", "x", count=2)
        self.assertTrue(plan.ok, plan.error)
        self.assertEqual(plan.changes[0].after, "x\nx\n")

    def test_missing_old_text_explains_itself(self):
        self.write("f.py", "x = 1\n")
        guard = SandboxGuard(self.root, mode="confirm")
        plan = plan_edit_file(guard, "f.py", "nope", "x")
        self.assertFalse(plan.ok)
        self.assertIn("not found", plan.error)

    def test_create_existing_file_is_refused(self):
        self.write("f.py", "x\n")
        guard = SandboxGuard(self.root, mode="confirm")
        self.assertFalse(plan_create_file(guard, "f.py", "y").ok)

    def test_delete_missing_file_is_refused(self):
        guard = SandboxGuard(self.root, mode="confirm")
        self.assertFalse(plan_delete_file(guard, "nope.py").ok)

    def test_noop_edit_is_detected(self):
        self.write("f.py", "same\n")
        guard = SandboxGuard(self.root, mode="confirm")
        plan = plan_edit_file(guard, "f.py", "same", "same")
        self.assertTrue(plan.is_noop())

    def test_readonly_sandbox_fails_at_plan_time(self):
        self.write("f.py", "x = 1\n")
        guard = SandboxGuard(self.root, mode="readonly")
        self.assertFalse(plan_edit_file(guard, "f.py", "x = 1", "x = 2").ok)

    def test_summary_mentions_files_and_line_counts(self):
        self.write("f.py", "a\nb\n")
        guard = SandboxGuard(self.root, mode="confirm")
        s = plan_edit_file(guard, "f.py", "b", "B").summary()
        self.assertIn("f.py", s)
        self.assertIn("+1", s)


class TestAtomicMultiFilePatches(ScratchCase):
    def test_stale_third_file_leaves_nothing_applied(self):
        """Regression: a patch failing on file 3 left files 1-2 written."""
        self.write("a.py", "a1\n")
        self.write("b.py", "b1\n")
        self.write("c.py", "TOTALLY DIFFERENT\n")
        patch = (
            "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a1\n+a2\n"
            "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-b1\n+b2\n"
            "--- a/c.py\n+++ b/c.py\n@@ -1 +1 @@\n-c1\n+c2\n"
        )
        reg = self.registry()
        r = reg.dispatch("apply_patch", {"patch": patch})
        self.assertFalse(r.ok)
        self.assertEqual(self.read("a.py"), "a1\n")
        self.assertEqual(self.read("b.py"), "b1\n")

    def test_whole_patch_applies_when_every_hunk_matches(self):
        self.write("a.py", "line1\nline2\nline3\n")
        self.write("b.py", "x=1\n")
        patch = (
            "--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n line1\n-line2\n+two\n line3\n"
            "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x=1\n+x=99\n"
        )
        r = self.registry().dispatch("apply_patch", {"patch": patch})
        self.assertTrue(r.ok, r.error)
        self.assertEqual(self.read("a.py"), "line1\ntwo\nline3\n")
        self.assertEqual(self.read("b.py"), "x=99\n")

    def test_context_mismatch_rejected(self):
        self.write("f.py", "AAA\nBBB\n")
        r = self.registry().dispatch("apply_patch", {
            "patch": "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n AAA\n-UNMATCHED\n+ZZZ\n"})
        self.assertFalse(r.ok)
        self.assertEqual(self.read("f.py"), "AAA\nBBB\n")

    def test_patch_creates_and_deletes_files(self):
        self.write("gone.py", "a\nb\nc\n")
        reg = self.registry()
        self.assertTrue(reg.dispatch("apply_patch", {
            "patch": "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,2 @@\n+z = 0\n+# new\n"
        }).ok)
        self.assertIn("z = 0", self.read("n.py"))
        self.assertTrue(reg.dispatch("apply_patch", {
            "patch": "--- a/gone.py\n+++ /dev/null\n@@ -1,3 +0,0 @@\n-a\n-b\n-c\n"
        }).ok)
        self.assertFalse((self.root / "gone.py").exists())

    def test_path_escape_rejected(self):
        r = self.registry().dispatch("apply_patch", {
            "patch": "--- a/../evil.py\n+++ b/../evil.py\n@@ -0,0 +1,1 @@\n+x\n"})
        self.assertFalse(r.ok)

    def test_wrong_line_numbers_still_anchor(self):
        """Models routinely emit @@ hints that are off by a few lines."""
        self.write("f.py", "\n".join(f"line{i}" for i in range(1, 21)) + "\n")
        r = self.registry().dispatch("apply_patch", {
            "patch": "--- a/f.py\n+++ b/f.py\n"
                     "@@ -3,3 +3,3 @@\n line9\n-line10\n+TEN\n line11\n"})
        self.assertTrue(r.ok, r.error)
        self.assertIn("TEN", self.read("f.py"))


class TestLineEndings(ScratchCase):
    def test_lf_file_is_not_rewritten_to_crlf(self):
        """Regression: text-mode writes translated every "\\n" on Windows, so a
        one-line edit produced a whole-file diff in the user's next git diff."""
        self.write("f.py", "a\nb\nc\n")
        self.registry().dispatch("edit_file", {"path": "f.py", "old": "b",
                                              "new": "B"})
        raw = (self.root / "f.py").read_bytes()
        self.assertNotIn(b"\r\n", raw)
        self.assertEqual(raw, b"a\nB\nc\n")

    def test_crlf_file_keeps_crlf(self):
        self.write("f.py", "a\r\nb\r\n")
        self.registry().dispatch("edit_file", {"path": "f.py", "old": "b",
                                              "new": "B"})
        self.assertEqual((self.root / "f.py").read_bytes(), b"a\r\nB\r\n")


# ------------------------------------------------------------- snapshots
class TestSnapshotsAndRollback(ScratchCase):
    def test_edit_and_rollback(self):
        self.write("f.py", "x = 1\n")
        reg = self.registry()
        self.assertTrue(reg.dispatch("edit_file", {
            "path": "f.py", "old": "x = 1", "new": "x = 2"}).ok)
        self.assertIn("x = 2", self.read("f.py"))
        reg.rollback()
        self.assertIn("x = 1", self.read("f.py"))

    def test_rollback_from_a_fresh_process(self):
        """Regression: `whalepod rollback` built an empty in-memory manager, so
        it was a permanent no-op no matter how many files had been rewritten."""
        self.write("f.py", "original\n")
        reg = self.registry()
        reg.dispatch("edit_file", {"path": "f.py", "old": "original",
                                   "new": "changed"})
        self.assertIn("changed", self.read("f.py"))

        # A different process: nothing but the manifest on disk.
        mgr = SnapshotManager.load_latest(backup_dir=self.backups, root=self.root)
        self.assertIsNotNone(mgr)
        report = mgr.rollback()
        self.assertTrue(any("restored" in line for line in report))
        self.assertEqual(self.read("f.py"), "original\n")

    def test_created_files_are_removed_on_rollback(self):
        reg = self.registry()
        reg.dispatch("create_file", {"path": "new.py", "content": "z=0"})
        self.assertTrue((self.root / "new.py").exists())
        SnapshotManager.load_latest(backup_dir=self.backups,
                                    root=self.root).rollback()
        self.assertFalse((self.root / "new.py").exists())

    def test_deleted_files_come_back_on_rollback(self):
        self.write("bye.py", "content\n")
        reg = self.registry()
        self.assertTrue(reg.dispatch("delete_file", {"path": "bye.py"}).ok)
        self.assertFalse((self.root / "bye.py").exists())
        reg.rollback()
        self.assertEqual(self.read("bye.py"), "content\n")

    def test_earliest_state_wins_across_repeated_edits(self):
        self.write("f.py", "v1\n")
        reg = self.registry()
        for old, new in (("v1", "v2"), ("v2", "v3")):
            reg.dispatch("edit_file", {"path": "f.py", "old": old, "new": new})
        self.assertIn("v3", self.read("f.py"))
        reg.rollback()
        self.assertEqual(self.read("f.py"), "v1\n")

    def test_same_named_files_in_different_dirs_do_not_collide(self):
        """Regression: backup names were flattened from the filename only."""
        self.write("one/cfg.py", "ONE\n")
        self.write("two/cfg.py", "TWO\n")
        reg = self.registry()
        reg.dispatch("edit_file", {"path": "one/cfg.py", "old": "ONE",
                                   "new": "X"})
        reg.dispatch("edit_file", {"path": "two/cfg.py", "old": "TWO",
                                   "new": "Y"})
        reg.rollback()
        self.assertEqual(self.read("one/cfg.py"), "ONE\n")
        self.assertEqual(self.read("two/cfg.py"), "TWO\n")

    def test_no_backup_dir_when_nothing_is_written(self):
        reg = self.registry("readonly")
        reg.dispatch("read_dir", {"path": "."})
        self.assertFalse(self.backups.exists())

    def test_sessions_lists_only_recorded_sessions(self):
        self.write("f.py", "a\n")
        self.registry().dispatch("edit_file", {"path": "f.py", "old": "a",
                                              "new": "b"})
        found = SnapshotManager.sessions(self.backups)
        self.assertEqual(len(found), 1)
        self.assertTrue((found[0] / "manifest.json").is_file())


# ----------------------------------------------------------- read tools
class TestReadTools(ScratchCase):
    def test_read_dir_and_grep(self):
        self.write("a.txt", "hello world\n")
        reg = self.registry("confirm")
        rd = reg.dispatch("read_dir", {"path": "."})
        self.assertTrue(rd.ok)
        self.assertIn("a.txt", rd.output)
        gr = reg.dispatch("grep", {"pattern": "world"})
        self.assertTrue(gr.ok)
        self.assertIn("world", gr.output)

    def test_grep_returns_repo_relative_paths(self):
        """Absolute paths made every hit unusable as a tool argument."""
        self.write("pkg/mod.py", "TARGET = 1\n")
        gr = self.registry("confirm").dispatch("grep", {"pattern": "TARGET"})
        self.assertIn("pkg/mod.py", gr.output.replace("\\", "/"))
        self.assertNotIn(str(self.root), gr.output)

    def test_read_file_reports_metadata_for_the_ledger(self):
        self.write("f.py", "a\nb\nc\n")
        r = self.registry("confirm").dispatch("read_file", {"path": "f.py"})
        self.assertTrue(r.ok)
        self.assertEqual(r.meta["read"]["path"], "f.py")
        self.assertEqual(r.meta["read"]["lines"], 3)

    def test_create_then_delete_roundtrip(self):
        reg = self.registry()
        self.assertTrue(reg.dispatch("create_file", {"path": "n.py",
                                                     "content": "z=0"}).ok)
        self.assertTrue((self.root / "n.py").exists())
        self.assertTrue(reg.dispatch("delete_file", {"path": "n.py"}).ok)
        self.assertFalse((self.root / "n.py").exists())


class TestArgumentParsing(unittest.TestCase):
    def test_malformed_json_is_reported_not_swallowed(self):
        """Regression: bad JSON became {}, so the model was told "path not
        found: ''" and had no idea its arguments were the problem."""
        from whalepod.tools.base import parse_args_checked
        args, err = parse_args_checked('{"path": "a.py"')
        self.assertEqual(args, {})
        self.assertTrue(err)
        self.assertIn("JSON", err)

    def test_valid_json_parses_cleanly(self):
        from whalepod.tools.base import parse_args_checked
        self.assertEqual(parse_args_checked('{"path": "a.py"}'),
                         ({"path": "a.py"}, ""))

    def test_empty_arguments_are_an_empty_dict(self):
        from whalepod.tools.base import parse_args_checked
        self.assertEqual(parse_args_checked(""), ({}, ""))


class TestSchemaStability(unittest.TestCase):
    def test_schemas_are_the_same_objects_every_call(self):
        """A rebuilt (or re-worded) schema list is a prefix-cache miss."""
        reg = ToolRegistry(Path.cwd(), sandbox_mode="readonly")
        a, b = reg.schemas(), reg.schemas()
        self.assertIs(a, b)
        self.assertEqual(repr(a), repr(b))

    def test_unknown_tool_lists_the_real_ones(self):
        reg = ToolRegistry(Path.cwd(), sandbox_mode="readonly")
        r = reg.dispatch("nonexistent_tool", {})
        self.assertFalse(r.ok)
        self.assertIn("read_file", r.error)


if __name__ == "__main__":
    unittest.main()
