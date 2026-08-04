"""Repo map tests against the fixture repo."""
import tempfile
import unittest
from pathlib import Path

from whalepod.core.tokenizer import estimate_tokens
from whalepod.context.repo_map import (
    _fair_share, build_repo_map, regex_extract, render_compact,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "repo"


class TestRepoMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx = build_repo_map(FIXTURE)

    def test_python_symbols_found(self):
        rels = {s.name for s in self.idx.by_file.get("src/auth.py", [])}
        self.assertIn("TokenManager", rels)
        self.assertIn("authenticate_token", rels)
        self.assertIn("issue", rels)

    def test_js_symbols_found(self):
        rels = {s.name for s in self.idx.by_file.get("src/util.js", [])}
        self.assertIn("formatUser", rels)
        self.assertIn("Greeter", rels)

    def test_go_symbols_found(self):
        rels = {s.name for s in self.idx.by_file.get("src/worker.go", [])}
        self.assertIn("Server", rels)
        self.assertIn("NewServer", rels)

    def test_render_compact(self):
        out = render_compact(self.idx, 100)
        self.assertIsInstance(out, str)
        self.assertGreater(len(out), 0)

    def test_regex_fallback(self):
        src = b"def foo(a):\n    pass\nclass Bar:\n    pass\n"
        syms = regex_extract(Path("x.py"), "python", src)
        names = {s.name for s in syms}
        self.assertIn("foo", names)
        self.assertIn("Bar", names)


class TestMapBudget(unittest.TestCase):
    """The map sits in the stable prefix, so its *token* cost is the budget.

    What used to happen: the only cap was a symbol count. On this very repo a
    2,000-symbol map rendered to ~70k tokens, because 2,000 TypeScript
    signatures are not the size of 2,000 Python ones. And the cut was plain
    alphabetical, so a vendored ``reference/**`` tree sorted ahead of
    ``whalepod/**`` and took the whole map with it.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def _long_module(self, n, prefix):
        """n functions with signatures long enough to cost real tokens."""
        args = ", ".join(f"{prefix}_argument_{i}=None" for i in range(8))
        return "".join(f"def {prefix}_function_{i}({args}):\n    pass\n\n"
                       for i in range(n))

    def test_token_budget_is_respected_where_a_symbol_cap_would_not_be(self):
        self.write("vendored/big.py", self._long_module(120, "vendored"))
        idx = build_repo_map(self.root)
        # Symbol cap far above the symbol count: only tokens can bite here.
        out = idx.render(max_symbols=5000, max_tokens=600)
        self.assertLessEqual(estimate_tokens(out), 700)      # + the notes
        self.assertLess(len([l for l in out.splitlines()
                             if not l.startswith("#")]), 120)

    def test_one_subtree_cannot_starve_another(self):
        # 'aaa' sorts first and is 20x bigger: under a plain alphabetical cut it
        # consumed the entire budget and 'zzz' never appeared at all.
        self.write("aaa/huge.py", self._long_module(200, "aaa"))
        self.write("zzz/small.py", self._long_module(3, "zzz"))
        out = build_repo_map(self.root).render(max_symbols=5000, max_tokens=900)
        self.assertIn("aaa/huge.py", out)
        self.assertIn("zzz/small.py", out)

    def test_auxiliary_trees_yield_to_source(self):
        """Equal shares gave tests/ a full render while source was truncated."""
        self.write("tests/test_all.py", self._long_module(100, "test"))
        self.write("src/core.py", self._long_module(100, "src"))
        out = build_repo_map(self.root).render(max_symbols=5000, max_tokens=1200)
        src = sum(1 for l in out.splitlines() if l.startswith("ƒ src/"))
        tst = sum(1 for l in out.splitlines() if l.startswith("ƒ tests/"))
        self.assertGreater(src, tst)
        self.assertGreater(tst, 0)          # still mentioned, not erased

    def test_truncation_is_reported_not_silent(self):
        """A silently cut map reads as 'this is the whole repo' to the model."""
        self.write("src/big.py", self._long_module(200, "src"))
        out = build_repo_map(self.root).render(max_symbols=5000, max_tokens=500)
        self.assertIn("not shown", out)
        self.assertIn("grep", out)
        self.assertIn("map truncated", out)

    def test_exclude_globs_drop_a_vendored_tree(self):
        self.write("reference/dep/lib.py", self._long_module(20, "dep"))
        self.write("app.py", self._long_module(2, "app"))
        idx = build_repo_map(self.root, exclude=["reference"])
        self.assertEqual(list(idx.by_file), ["app.py"])

    def test_scan_order_is_deterministic_and_shallow_first(self):
        """Under the scan cap, os.walk order decided *which* files were mapped,
        so the prefix differed between machines — a cache miss every turn."""
        self.write("top.py", self._long_module(4, "top"))
        self.write("a/b/c/d/deep.py", self._long_module(4, "deep"))
        idx = build_repo_map(self.root, max_symbols=4)
        self.assertIn("top.py", idx.by_file)
        self.assertNotIn("a/b/c/d/deep.py", idx.by_file)

    def test_rescan_keeps_the_configured_caps(self):
        self.write("src/big.py", self._long_module(50, "src"))
        idx = build_repo_map(self.root, max_symbols=10, max_tokens=400)
        idx.scan()                                   # repo_map_refresh path
        self.assertEqual(idx.max_symbols, 10)
        self.assertLessEqual(estimate_tokens(idx.render(max_symbols=5000)), 500)

    def test_fair_share_hands_back_what_is_not_needed(self):
        got = _fair_share({"src": 1000, "other": 10}, 500)
        self.assertEqual(got["other"], 10)           # takes only what it needs
        self.assertEqual(sum(got.values()), 500)     # ...the rest goes to src
        self.assertEqual(_fair_share({"only": 40}, 500), {"only": 40})
        self.assertEqual(_fair_share({"a": 5}, 0), {"a": 0})


if __name__ == "__main__":
    unittest.main()
