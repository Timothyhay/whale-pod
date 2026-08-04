"""Where tool output meets the context window.

Every case here comes from a defect found by reading the pi reference agent.
The common thread is that a middle-out cut is wrong
for anything the *model* has to act on: the model cannot ask for "the middle", so
a hole in the centre is unaddressable. Which end to keep depends on where the
information is — the top for a file, the bottom for a command.

The second half covers line endings and the BOM, which are not cosmetic: they
decided whether a multi-line ``edit_file`` could match at all on a Windows
checkout.
"""
import os
import tempfile
import unittest
from pathlib import Path

from whalepod.sandbox.guard import SandboxGuard
from whalepod.tools import textfile
from whalepod.tools.base import (
    truncate, truncate_head, truncate_line, truncate_tail,
)
from whalepod.tools.edit import _bound_command_output, run_command_now
from whalepod.tools.plan import plan_edit_file
from whalepod.tools.read import tool_read_file


class ScratchCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self._prev = os.environ.get("WHALEPOD_BACKUP_DIR")
        os.environ["WHALEPOD_BACKUP_DIR"] = str(self.root / ".backups")
        self.guard = SandboxGuard(self.root, mode="yes")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("WHALEPOD_BACKUP_DIR", None)
        else:
            os.environ["WHALEPOD_BACKUP_DIR"] = self._prev
        self._tmp.cleanup()

    def write_bytes(self, rel: str, data: bytes) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p


# ------------------------------------------------------------- primitives
class TestCutPrimitives(unittest.TestCase):
    def test_head_keeps_the_beginning(self):
        cut = truncate_head([f"L{i}" for i in range(100)], max_lines=10)
        self.assertEqual(cut.lines[0], "L0")
        self.assertEqual((cut.first, cut.last, cut.kept_lines), (1, 10, 10))
        self.assertEqual((cut.truncated, cut.by), (True, "lines"))

    def test_tail_keeps_the_end(self):
        cut = truncate_tail([f"L{i}" for i in range(100)], max_lines=10)
        self.assertEqual(cut.lines[-1], "L99")
        self.assertEqual((cut.first, cut.last, cut.kept_lines), (91, 100, 10))

    def test_nothing_is_reported_as_truncated_when_it_all_fits(self):
        cut = truncate_head(["a", "b"], max_lines=10, max_chars=100)
        self.assertFalse(cut.truncated)
        self.assertEqual(cut.by, "")
        self.assertEqual(cut.text, "a\nb")

    def test_the_character_budget_is_reported_separately(self):
        """'You got 10 lines' and 'you got 4 KB' call for different next moves:
        read on by line, versus these lines are huge, narrow the range."""
        cut = truncate_head(["x" * 100] * 50, max_lines=1_000, max_chars=400)
        self.assertEqual(cut.by, "chars")
        self.assertLess(cut.kept_lines, 50)

    def test_cuts_land_on_whole_lines(self):
        cut = truncate_head([f"L{i}" for i in range(50)], max_chars=41)
        self.assertTrue(all(ln.startswith("L") for ln in cut.lines))

    def test_one_line_larger_than_the_whole_budget_still_returns_content(self):
        cut = truncate_head(["y" * 10_000], max_lines=10, max_chars=100)
        self.assertEqual(len(cut.lines), 1)
        self.assertEqual(len(cut.lines[0]), 100)
        self.assertTrue(cut.truncated)

    def test_empty_input(self):
        cut = truncate_head([])
        self.assertEqual(cut.lines, [])
        self.assertFalse(cut.truncated)

    def test_a_long_grep_hit_is_bounded_and_says_by_how_much(self):
        out = truncate_line("z" * 1_000, 100)
        self.assertTrue(out.startswith("z" * 100))
        self.assertIn("+900 chars", out)

    def test_middle_out_survives_for_human_facing_text(self):
        out = truncate("a" * 1_000, limit=100)
        self.assertIn("truncated", out)
        self.assertTrue(out.startswith("a"))
        self.assertTrue(out.endswith("a"))


# ------------------------------------------------------------- file reads
class TestReadTruncation(ScratchCase):
    def read(self, rel, **kw):
        return tool_read_file(self.guard, None, rel, **kw)

    def test_a_short_file_is_complete(self):
        self.write_bytes("a.py", b"x = 1\ny = 2\n")
        r = self.read("a.py")
        info = r.meta["read"]
        self.assertTrue(info["complete"])
        self.assertFalse(info["truncated"])
        self.assertEqual((info["start"], info["end"], info["lines"]), (1, 2, 2))

    def test_a_long_file_is_cut_from_the_head_not_the_middle(self):
        self.write_bytes("big.py", b"".join(
            f"line {i}\n".encode() for i in range(5_000)))
        r = self.read("big.py", max_lines=100)
        self.assertIn("line 0", r.output)
        self.assertNotIn("line 4999", r.output)
        self.assertIn("truncated at the line budget", r.output)

    def test_the_ledger_is_told_only_what_was_delivered(self):
        """The bug this fixes: meta reported the *requested* 0-0 (whole file)
        while the output had been cut, so the ledger answered a later read of the
        missing part with 'it is already above'."""
        self.write_bytes("big.py", b"".join(
            f"line {i}\n".encode() for i in range(5_000)))
        info = self.read("big.py", max_lines=100).meta["read"]
        self.assertFalse(info["complete"])
        self.assertTrue(info["truncated"])
        self.assertEqual((info["start"], info["end"]), (1, 100))
        self.assertEqual(info["lines"], 5_000)

    def test_it_writes_out_the_next_call_for_the_model(self):
        """'output truncated' alone produced either a blind re-read of the same
        range or a guess at what was missing."""
        self.write_bytes("big.py", b"".join(
            f"line {i}\n".encode() for i in range(5_000)))
        out = self.read("big.py", max_lines=100).output
        self.assertIn("read_file(path='big.py', start=101", out)

    def test_continuing_from_the_suggested_line_covers_the_gap(self):
        self.write_bytes("big.py", b"".join(
            f"line {i}\n".encode() for i in range(300)))
        first = self.read("big.py", max_lines=100).meta["read"]
        second = self.read("big.py", start=first["end"] + 1, max_lines=100)
        self.assertIn("line 100", second.output)
        self.assertEqual(second.meta["read"]["start"], 101)

    def test_a_requested_range_is_reported_in_absolute_line_numbers(self):
        self.write_bytes("big.py", b"".join(
            f"line {i}\n".encode() for i in range(300)))
        info = self.read("big.py", start=200, end=250).meta["read"]
        self.assertEqual((info["start"], info["end"]), (200, 250))
        self.assertFalse(info["complete"])   # a range is never the whole file

    def test_a_start_past_the_end_says_how_long_the_file_is(self):
        self.write_bytes("a.py", b"x\ny\n")
        r = self.read("a.py", start=99)
        self.assertFalse(r.ok)
        self.assertIn("has 2 lines", r.error)

    def test_an_empty_file_reads_as_empty_not_as_a_bad_argument(self):
        self.write_bytes("empty.py", b"")
        r = self.read("empty.py")
        self.assertTrue(r.ok, r.error)
        self.assertIn("empty file", r.output)
        self.assertTrue(r.meta["read"]["complete"])

    def test_a_single_enormous_line_cannot_blow_the_char_budget(self):
        self.write_bytes("min.js", b"var x=1;" * 100_000)
        r = self.read("min.js", max_chars=2_000)
        self.assertTrue(r.ok, r.error)
        self.assertLess(len(r.output), 3_000)
        self.assertTrue(r.meta["read"]["truncated"])

    def test_a_binary_file_is_refused_not_mangled(self):
        self.write_bytes("blob.bin", b"\x00\x01\x02\xff\xfe")
        r = self.read("blob.bin")
        self.assertFalse(r.ok)
        self.assertIn("binary", r.error)


# --------------------------------------------------------- command output
class TestCommandOutput(ScratchCase):
    def test_short_output_is_untouched(self):
        text, spilled = _bound_command_output("all good", self.guard)
        self.assertEqual(text, "all good")
        self.assertEqual(spilled, "")

    def test_the_tail_is_kept_because_that_is_where_the_failure_is(self):
        body = "\n".join([f"progress {i}" for i in range(5_000)]
                         + ["Traceback (most recent call last):",
                            "AssertionError: boom"])
        text, spilled = _bound_command_output(body, self.guard, max_lines=50)
        self.assertIn("AssertionError: boom", text)
        self.assertNotIn("progress 0", text)
        self.assertIn("earlier lines cut", text)

    def test_the_full_output_is_spilled_where_the_model_can_grep_it(self):
        body = "\n".join(f"line {i}" for i in range(5_000))
        text, spilled = _bound_command_output(body, self.guard, max_lines=50)
        self.assertTrue(spilled)
        self.assertIn(spilled, text)
        # A path outside the sandbox root would be unopenable by the very tools
        # the model would use to look at it.
        full = (self.root / spilled).read_text(encoding="utf-8")
        self.assertIn("line 0", full)
        self.assertIn("line 4999", full)
        self.assertEqual(self.guard.resolve_within(spilled), self.root / spilled)

    def test_two_spills_do_not_overwrite_each_other(self):
        body = "\n".join(f"line {i}" for i in range(5_000))
        a = _bound_command_output(body, self.guard, max_lines=50)[1]
        b = _bound_command_output(body + "\nsecond", self.guard, max_lines=50)[1]
        self.assertNotEqual(a, b)

    def test_a_failing_command_keeps_its_error_text(self):
        r = run_command_now(self.guard, "python -c \"import sys;"
                                        "[print(i) for i in range(3000)];"
                                        "sys.exit(3)\"",
                            timeout=60, approved=True)
        self.assertFalse(r.ok)
        self.assertEqual(r.meta["exit_code"], 3)
        self.assertIn("2999", r.output)
        self.assertIn("full_output", r.meta)


# -------------------------------------------------- line endings and BOM
class TestFileConventions(ScratchCase):
    def test_crlf_is_normalised_for_matching(self):
        tf = textfile.decode(b"a\r\nb\r\nc\r\n")
        self.assertEqual(tf.text, "a\nb\nc\n")
        self.assertEqual(tf.newline, "\r\n")

    def test_the_bom_is_stripped_and_remembered(self):
        tf = textfile.decode("﻿a\nb\n".encode("utf-8"))
        self.assertEqual(tf.text, "a\nb\n")
        self.assertEqual(tf.bom, "﻿")

    def test_restore_puts_the_conventions_back(self):
        tf = textfile.decode("﻿a\r\nb\r\n".encode("utf-8"))
        self.assertEqual(tf.restore("a\nB\n"), "﻿a\r\nB\r\n")

    def test_mixed_endings_go_to_the_majority(self):
        self.assertEqual(textfile.decode(b"a\r\nb\r\nc\n").newline, "\r\n")
        self.assertEqual(textfile.decode(b"a\r\nb\nc\n").newline, "\n")

    def test_a_multi_line_edit_matches_a_crlf_file(self):
        """The defect: read_file renders LF, so the model's `old` had "\\n"
        where a CRLF file had "\\r\\n" — every multi-line edit failed with
        "`old` text not found", and re-reading did not help."""
        self.write_bytes("m.py", b"def f():\r\n    return 1\r\n")
        plan = plan_edit_file(self.guard, "m.py",
                              "def f():\n    return 1\n",
                              "def f():\n    return 2\n")
        self.assertTrue(plan.ok, plan.error)
        plan.commit(_Snapshots(self.root))
        self.assertEqual((self.root / "m.py").read_bytes(),
                         b"def f():\r\n    return 2\r\n")

    def test_an_edit_on_the_first_line_of_a_bom_file_matches(self):
        """The BOM is invisible in read output, so the model never includes it in
        `old`, and an edit anchored at line 1 could not match."""
        self.write_bytes("b.py", "﻿import os\nx = 1\n".encode("utf-8"))
        plan = plan_edit_file(self.guard, "b.py", "import os", "import sys")
        self.assertTrue(plan.ok, plan.error)
        plan.commit(_Snapshots(self.root))
        self.assertEqual((self.root / "b.py").read_bytes(),
                         "﻿import sys\nx = 1\n".encode("utf-8"))

    def test_reading_a_crlf_file_shows_no_carriage_returns(self):
        self.write_bytes("c.py", b"a\r\nb\r\n")
        out = tool_read_file(self.guard, None, "c.py").output
        self.assertNotIn("\r", out)


def _Snapshots(root):
    from whalepod.sandbox.snapshot import SnapshotManager
    return SnapshotManager(root=root)


if __name__ == "__main__":
    unittest.main()
