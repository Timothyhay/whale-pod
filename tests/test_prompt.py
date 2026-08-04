"""The system prompt, and its coupling to the tool set that is actually offered.

Usage rules used to be prose in ``prompt.py`` while the tools they describe lived
in ``registry.py``. Two things went wrong with that, both borrowed from the pi
reference agent's design:

  * a tool could ship with a schema and no rules, because the rules were in
    another file nobody edited;
  * a readonly session still advertised the write tools *and* carried three
    paragraphs about how to edit files.

Guidance now hangs off each tool definition and the enabled tool set drives both.
The prompt is still the head of the prefix-cached zone, so the byte-stability
tests below matter as much as the content ones.
"""
import tempfile
import unittest
from pathlib import Path

from whalepod.core.prompt import build_system_prompt, repo_map_section
from whalepod.tools.registry import ALL_TOOLS, WRITE_TOOL_NAMES, ToolRegistry


def _registry(mode="confirm"):
    return ToolRegistry(Path(tempfile.gettempdir()).resolve(),
                        sandbox_mode=mode)


class TestGuidelinesLiveWithTheirTool(unittest.TestCase):
    def test_every_guideline_reaches_the_prompt(self):
        reg = _registry()
        prompt = build_system_prompt(guidelines=reg.guidelines())
        for line in reg.guidelines():
            self.assertIn(line, prompt)

    def test_guidelines_are_not_sent_as_part_of_the_wire_schema(self):
        """An unknown key in a tool definition is a 400 on some providers."""
        for schema in _registry().schemas():
            self.assertEqual(set(schema), {"type", "function"})
            self.assertEqual(set(schema["function"]),
                             {"name", "description", "parameters"})

    def test_duplicates_are_said_once(self):
        reg = _registry()
        gl = reg.guidelines()
        self.assertEqual(len(gl), len(set(gl)))

    def test_the_edit_rules_are_present_when_editing_is_possible(self):
        prompt = build_system_prompt(guidelines=_registry().guidelines())
        self.assertIn("occurs exactly once", prompt)
        self.assertIn("run_command", prompt)


class TestReadonlyDropsWriteTooling(unittest.TestCase):
    def test_write_tools_are_not_offered(self):
        """A readonly sandbox refuses every write, so advertising them buys only
        a round trip that ends in "sandbox is readonly"."""
        names = _registry("readonly").tool_names()
        for w in WRITE_TOOL_NAMES:
            self.assertNotIn(w, names)
        self.assertIn("read_file", names)

    def test_their_guidance_goes_with_them(self):
        prompt = build_system_prompt(guidelines=_registry("readonly").guidelines())
        self.assertNotIn("occurs exactly once", prompt)
        self.assertNotIn("run_command", prompt)
        self.assertIn("Look before you leap", prompt)

    def test_confirm_mode_offers_everything(self):
        self.assertEqual(len(_registry("confirm").schemas()), len(ALL_TOOLS))

    def test_the_unknown_tool_message_lists_only_offered_tools(self):
        reg = _registry("readonly")
        err = reg.dispatch("edit_file", {"path": "a", "old": "x",
                                         "new": "y"}).error
        self.assertIn("unknown tool", err)
        self.assertNotIn("edit_file,", err)


class TestPrefixStability(unittest.TestCase):
    def test_the_prompt_is_byte_identical_between_builds(self):
        reg = _registry()
        a = build_system_prompt("PROJ", guidelines=reg.guidelines())
        b = build_system_prompt("PROJ", guidelines=reg.guidelines())
        self.assertEqual(a, b)

    def test_schemas_are_the_same_objects_every_call(self):
        reg = _registry()
        self.assertIs(reg.schemas(), reg.schemas())

    def test_nothing_session_specific_leaks_in(self):
        """cwd, mode and the clock belong in the user turn, not the cached head."""
        prompt = build_system_prompt(guidelines=_registry().guidelines())
        self.assertNotIn(str(Path.cwd()), prompt)
        self.assertNotIn(tempfile.gettempdir(), prompt)

    def test_project_instructions_come_last(self):
        prompt = build_system_prompt("USE TABS",
                                     guidelines=_registry().guidelines())
        self.assertIn("USE TABS", prompt)
        self.assertGreater(prompt.index("USE TABS"), prompt.index("Answering"))

    def test_it_is_usable_with_no_registry_at_all(self):
        prompt = build_system_prompt()
        self.assertIn("WhalePod", prompt)
        self.assertIn("Using the tools", prompt)
        self.assertIn("Answering", prompt)

    def test_no_empty_heading_when_there_is_no_guidance(self):
        prompt = build_system_prompt(guidelines=[])
        self.assertNotIn("Using the tools", prompt)
        self.assertIn("Answering", prompt)


class TestRepoMapSection(unittest.TestCase):
    def test_it_says_what_it_is(self):
        out = repo_map_section("a.py:1 def f()")
        self.assertIn("Repository map", out)
        self.assertIn("a.py:1 def f()", out)

    def test_an_empty_map_adds_nothing(self):
        self.assertEqual(repo_map_section(""), "")


class TestProjectInstructionFiles(unittest.TestCase):
    """WHALEPOD.md first, then the conventions other agents established.

    AGENTS.md / CLAUDE.md is where a repo already records its build command,
    test invocation and house style; a session that ignores them opens by
    rediscovering all of it.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def load(self):
        from whalepod.cli import Session
        session = Session.__new__(Session)     # no endpoint, no network
        session.root = self.root
        session._instructions_from = ""
        return session._project_instructions(), session._instructions_from

    def test_nothing_is_not_an_error(self):
        self.assertEqual(self.load(), ("", ""))

    def test_agents_md_is_read(self):
        (self.root / "AGENTS.md").write_text("run: pytest -q", encoding="utf-8")
        text, src = self.load()
        self.assertIn("pytest -q", text)
        self.assertEqual(src, "AGENTS.md")

    def test_claude_md_is_read(self):
        (self.root / "CLAUDE.md").write_text("use tabs", encoding="utf-8")
        self.assertEqual(self.load()[1], "CLAUDE.md")

    def test_our_own_file_wins(self):
        (self.root / "AGENTS.md").write_text("generic", encoding="utf-8")
        (self.root / "WHALEPOD.md").write_text("specific", encoding="utf-8")
        text, src = self.load()
        self.assertEqual(src, "WHALEPOD.md")
        self.assertNotIn("generic", text)

    def test_an_empty_file_does_not_shadow_a_later_one(self):
        (self.root / "WHALEPOD.md").write_text("   \n", encoding="utf-8")
        (self.root / "AGENTS.md").write_text("real content", encoding="utf-8")
        self.assertEqual(self.load()[1], "AGENTS.md")


if __name__ == "__main__":
    unittest.main()
