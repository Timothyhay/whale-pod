"""Repo Map — compact codebase symbol table for the stable context prefix.

Two backends:
  - Tree-sitter (preferred): real AST, accurate declarations.
  - Regex fallback: line-based extraction when a grammar is unavailable.

The renderer emits one line per symbol. That block is installed in the *stable
prefix* (see :mod:`whalepod.core.messages`), so it must be small, sorted and
deterministic — an unstable map would cost a cache miss every turn.

What a "signature" is
    The declaration only: ``def f(a, b) -> int``, ``class A(B)``. This used to
    be the collapsed text of the *entire node* truncated to 90 characters,
    i.e. the first 90 characters of the function body — which read like a
    signature but was mostly noise, and inflated the prefix for nothing.

Budgets are in tokens, and fairly shared
    The cap used to be a symbol *count*. But 2,000 symbols of TypeScript is not
    the same size as 2,000 symbols of Python: on this very repo, a 2,000-symbol
    map rendered to **70k tokens**. A budget has to be denominated in the
    currency the window is denominated in, so ``max_tokens`` is the real limit
    and ``max_symbols`` is only a backstop.

    The cut also used to be plain alphabetical, which let one subtree starve
    every other: a vendored ``reference/**`` tree sorted before ``whalepod/**``
    and consumed the whole map, so the agent's map of *its own project* was
    empty. Each top-level component now gets a fair share of the budget
    (:func:`_fair_share`), and truncation is reported per component instead of
    being silent.

Caching
    Scanning re-parses only files whose (mtime, size) changed since the last
    scan, so ``repo_map_refresh`` on a large repo is cheap. Candidates are
    visited in a deterministic relevance order rather than in ``os.walk``
    order, so hitting the cap drops the least relevant files — and drops the
    *same* ones on every machine, which matters because this block sits in the
    cached prefix.
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..core.tokenizer import estimate_tokens
from .grammar import get_language, has_treesitter, language_for_file

_DEFAULT_IGNORE = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", "target",
    ".idea", ".vscode", ".next", ".output", "coverage", ".hg", ".svn",
}
_IGNORE_EXTS = {".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe", ".dylib",
                ".lock", ".min.js", ".map"}
MAX_FILE_BYTES = 2_000_000
# Token budget for the rendered map. It lives in the stable prefix, so it is
# paid once per session (cached afterwards) — but it is also subtracted from the
# window for the whole session, which is why it is bounded at all.
DEFAULT_MAP_TOKENS = 8_000

_NODE_KIND = {
    "python": {
        "function_definition": "def",
        "class_definition": "class",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
    },
    "go": {
        "function_declaration": "func",
        "method_declaration": "method",
        "type_spec": "type",
    },
}

_ICONS = {"def": "ƒ", "class": "◈", "func": "ƒ", "function": "ƒ",
          "method": "m", "interface": "◇", "type": "▣"}


@dataclass
class Symbol:
    name: str
    kind: str            # def / class / func / interface / type / method
    file: str            # repo-relative, forward slashes
    line: int            # 1-based start line
    signature: str = ""  # declaration only (no body)
    doc: str = ""        # one-line docstring hint


def _node_text(node, src: bytes) -> str:
    try:
        return src[node.start_byte:node.end_byte].decode("utf-8", "replace")
    except Exception:
        return ""


def _snip(s: str, limit: int = 100) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


class TreeSitterExtractor:
    def __init__(self):
        self.parsers: dict[str, object] = {}
        self._missing: set[str] = set()

    def parser_for(self, lang: str):
        if lang in self._missing:
            return None
        if lang not in self.parsers:
            parser, ok = get_language(lang)
            if not ok:
                self._missing.add(lang)
                return None
            self.parsers[lang] = parser
        return self.parsers.get(lang)

    def extract(self, rel: str, lang: str, src: bytes) -> list[Symbol]:
        parser = self.parser_for(lang)
        if parser is None:
            return []
        kinds = _NODE_KIND.get(lang, {})
        try:
            tree = parser.parse(src)
        except Exception:
            return []
        out: list[Symbol] = []
        self._walk(tree.root_node, rel, kinds, src, out)
        return out

    def _walk(self, node, rel: str, kinds: dict, src: bytes, out: list[Symbol]):
        if node.type in kinds:
            sym = self._make_symbol(node, rel, kinds[node.type], src)
            if sym and sym.name:
                out.append(sym)
        for child in node.children:
            self._walk(child, rel, kinds, src, out)

    def _make_symbol(self, node, rel: str, kind: str, src: bytes):
        name_node = node.child_by_field_name("name")
        if name_node is None or not name_node.type[0].isalpha():
            return None
        name = _node_text(name_node, src).strip()
        if not name:
            return None
        body = node.child_by_field_name("body")
        # Declaration = everything before the body. That is the useful part.
        end = body.start_byte if body is not None else node.end_byte
        sig = _snip(src[node.start_byte:end].decode("utf-8", "replace")
                    .rstrip().rstrip(":{").rstrip())
        doc = ""
        if body is not None and len(body.children) > 0:
            first = body.children[0]
            if first.type in ("expression_statement", "string", "comment"):
                raw = _node_text(first, src)
                if raw.lstrip()[:1] in ('"', "'", "/", "#"):
                    doc = _snip(raw.strip().strip('"\'/#* \n'), 70)
        return Symbol(name=name, kind=kind, file=rel,
                      line=node.start_point[0] + 1, signature=sig, doc=doc)


# ------------------------------------------------------------------ regex
_PY_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)\s*(\([^)]*\))", re.M)
_PY_CLASS_RE = re.compile(r"^\s*class\s+(\w+)\s*(\([^)]*\))?", re.M)
_JS_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(\([^)]*\))|"
    r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)|"
    r"^\s*(?:export\s+)?(?:interface|type)\s+(\w+)",
    re.M,
)
_GO_RE = re.compile(
    r"^\s*func\s+(?:\((\w+\s+\*?\w+)\)\s+)?(\w+)\s*(\([^)]*\))|"
    r"^\s*type\s+(\w+)\s+(struct|interface)",
    re.M,
)


def regex_extract(rel: str, lang: str, src: bytes) -> list[Symbol]:
    text = src.decode("utf-8", "replace")
    out: list[Symbol] = []

    def line_of(m) -> int:
        return text[: m.start()].count("\n") + 1

    if lang == "python":
        for m in _PY_DEF_RE.finditer(text):
            out.append(Symbol(m.group(1), "def", rel, line_of(m),
                              signature=_snip(f"def {m.group(1)}{m.group(2)}")))
        for m in _PY_CLASS_RE.finditer(text):
            bases = m.group(2) or ""
            out.append(Symbol(m.group(1), "class", rel, line_of(m),
                              signature=_snip(f"class {m.group(1)}{bases}")))
    elif lang in ("javascript", "typescript"):
        for m in _JS_RE.finditer(text):
            if m.group(1):
                out.append(Symbol(m.group(1), "function", rel, line_of(m),
                                  signature=_snip(f"function {m.group(1)}"
                                                  f"{m.group(2)}")))
            elif m.group(3):
                out.append(Symbol(m.group(3), "class", rel, line_of(m),
                                  signature=f"class {m.group(3)}"))
            elif m.group(4):
                out.append(Symbol(m.group(4), "type", rel, line_of(m),
                                  signature=f"type {m.group(4)}"))
    elif lang == "go":
        for m in _GO_RE.finditer(text):
            if m.group(2):
                recv = f"({m.group(1)}) " if m.group(1) else ""
                out.append(Symbol(m.group(2), "method" if m.group(1) else "func",
                                  rel, line_of(m),
                                  signature=_snip(f"func {recv}{m.group(2)}"
                                                  f"{m.group(3)}")))
            elif m.group(4):
                out.append(Symbol(m.group(4), "type", rel, line_of(m),
                                  signature=f"type {m.group(4)} {m.group(5)}"))
    return out


# ------------------------------------------------------------- gitignore
def load_ignore_globs(root: Path) -> list[str]:
    """Very small .gitignore reader: plain globs and directory names only.

    Not a full gitignore implementation — negations and anchored patterns are
    skipped rather than half-honoured, so the map errs toward including a file.
    """
    globs: list[str] = []
    gi = root / ".gitignore"
    if not gi.is_file():
        return globs
    try:
        for raw in gi.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            globs.append(line.rstrip("/"))
    except OSError:
        pass
    return globs


def _relevance_key(rel: str) -> tuple:
    """Ordering used both for what gets scanned and for what gets rendered.

    Shallow first, then alphabetical. It is a crude proxy for relevance, but a
    load-bearing one: a project's own entry points and packages sit near the
    root, while vendored, generated and example trees are buried
    (``reference/pi-main/packages/coding-agent/src/core/...``). Under a cap,
    "shallow first" keeps the code the user is actually working on.
    """
    return (rel.count("/"), rel)


def _ignored(rel: str, globs: Iterable[str]) -> bool:
    parts = rel.split("/")
    for g in globs:
        if "/" in g:
            if fnmatch.fnmatch(rel, g.lstrip("/")):
                return True
        elif any(fnmatch.fnmatch(part, g) for part in parts):
            return True
    return False


# --------------------------------------------------------------- scanning
class RepoIndex:
    """path -> list[Symbol], with an mtime cache so rescans are cheap."""

    def __init__(self, root: Path, languages: Optional[list[str]] = None,
                 use_treesitter: bool = True,
                 exclude: Optional[list[str]] = None,
                 max_tokens: int = DEFAULT_MAP_TOKENS):
        self.root = Path(root).resolve()
        self.by_file: dict[str, list[Symbol]] = {}
        self.errors: list[str] = []
        self.languages = set(languages) if languages else None
        self.use_treesitter = use_treesitter and has_treesitter()
        self.ts_used = False
        self.skipped = 0
        self.max_tokens = max_tokens
        self.max_symbols = 5000        # set by scan(); reused by rescans
        self.exclude = list(exclude or [])
        self._stat_cache: dict[str, tuple[int, int]] = {}
        self._ignore_globs: list[str] = []
        self._extractor = TreeSitterExtractor()

    # -- scan ----------------------------------------------------------
    def scan(self, max_symbols: Optional[int] = None) -> "RepoIndex":
        # A rescan (``repo_map_refresh``) must not silently fall back to the
        # library default and widen a cap the user narrowed.
        max_symbols = self.max_symbols if max_symbols is None else max_symbols
        self.max_symbols = max_symbols
        self._ignore_globs = load_ignore_globs(self.root) + self.exclude
        # Collect first, then sort: ``os.walk`` order is filesystem-dependent,
        # so under the cap the *contents* of the map differed between machines
        # and between runs — an unstable prefix, i.e. a cache miss every turn.
        candidates: list[tuple[str, Path, str]] = []
        for path in self._walk_files():
            rel = str(path.relative_to(self.root)).replace("\\", "/")
            if _ignored(rel, self._ignore_globs):
                continue
            lang = language_for_file(str(path))
            if lang is None or (self.languages and lang not in self.languages):
                continue
            candidates.append((rel, path, lang))
        candidates.sort(key=lambda c: _relevance_key(c[0]))
        seen = {rel for rel, _, _ in candidates}
        count = 0
        for rel, path, lang in candidates:
            try:
                st = path.stat()
                ident = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
            if st.st_size > MAX_FILE_BYTES:
                self.skipped += 1
                continue
            if self._stat_cache.get(rel) == ident and rel in self.by_file:
                count += len(self.by_file[rel])
                continue          # unchanged since the last scan
            if count >= max_symbols:
                self.skipped += 1
                continue
            try:
                src = path.read_bytes()
            except OSError as e:
                self.errors.append(f"{rel}: {e}")
                continue
            syms = []
            if self.use_treesitter:
                syms = self._extractor.extract(rel, lang, src)
                if syms:
                    self.ts_used = True
            if not syms:
                syms = regex_extract(rel, lang, src)
            self._stat_cache[rel] = ident
            if syms:
                self.by_file[rel] = syms
                count += len(syms)
            else:
                self.by_file.pop(rel, None)
        # Files deleted since the last scan must leave the map.
        for gone in [r for r in self.by_file if r not in seen]:
            self.by_file.pop(gone, None)
            self._stat_cache.pop(gone, None)
        return self

    def _walk_files(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames
                           if d not in _DEFAULT_IGNORE and not d.startswith(".")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                if any(fn.lower().endswith(e) for e in _IGNORE_EXTS):
                    continue
                yield Path(dirpath) / fn

    # -- summary -------------------------------------------------------
    @property
    def symbol_count(self) -> int:
        return sum(len(v) for v in self.by_file.values())

    def render(self, max_symbols: int = 2000,
               max_tokens: Optional[int] = None) -> str:
        return render_compact(
            self, max_symbols,
            self.max_tokens if max_tokens is None else max_tokens)


# ---------------------------------------------------------------- render
# Directories that are worth *mentioning* in the map but are not the code
# under work. They are also the most repetitive (a test file is fifty
# ``def test_x(self)`` lines), so they buy the least understanding per token.
_AUX_DIRS = {
    "test", "tests", "spec", "specs", "doc", "docs", "example", "examples",
    "sample", "samples", "benchmark", "benchmarks", "fixtures", "testdata",
    "scripts", "tools", "vendor", "third_party", "thirdparty", "reference",
}
_AUX_WEIGHT = 0.35


def _weight(group: str) -> float:
    return _AUX_WEIGHT if group.lower() in _AUX_DIRS else 1.0


def _fair_share(need: dict[str, int], budget: int) -> dict[str, int]:
    """Weighted max-min fair split of ``budget`` over top-level components.

    Everyone gets a share proportional to its weight; whatever a component does
    not need is handed back to the others. Processing in ascending order of
    need-per-weight is what makes that work in a single pass.

    Two failures this replaces:

    * Without any sharing the map was a plain alphabetical cut, so one big
      subtree consumed the whole budget and the rest of the repo was invisible.
    * With *equal* sharing, ``tests/`` was rendered in full while 171 symbols of
      the actual implementation were dropped — fair, and useless.
    """
    out: dict[str, int] = {}
    left = max(0, budget)
    order = sorted(need, key=lambda g: (need[g] / _weight(g), g))
    remaining_weight = sum(_weight(g) for g in order)
    for group in order:
        w = _weight(group)
        share = int(left * w / remaining_weight) if remaining_weight > 0 else 0
        take = min(need[group], share)
        out[group] = take
        left -= take
        remaining_weight -= w
    return out


def _symbol_line(rel: str, sym: Symbol) -> str:
    icon = _ICONS.get(sym.kind, "•")
    line = f"{icon} {rel}:{sym.line}  {sym.signature or f'{sym.kind} {sym.name}'}"
    return f"{line}  # {sym.doc}" if sym.doc else line


def render_compact(index: RepoIndex, max_symbols: int = 2000,
                   max_tokens: int = DEFAULT_MAP_TOKENS) -> str:
    """One line per symbol — the compact stable-prefix block.

    Bounded by ``max_tokens`` (the real constraint) with ``max_symbols`` as a
    backstop, and fairly shared across top-level components so no single
    subtree can crowd out the rest.
    """
    groups: dict[str, list[str]] = {}
    costs: dict[str, list[int]] = {}
    for rel in sorted(index.by_file, key=_relevance_key):
        top = rel.split("/")[0] if "/" in rel else "."
        for sym in index.by_file[rel]:
            line = _symbol_line(rel, sym)
            groups.setdefault(top, []).append(line)
            # +1 for the newline that joins it to the block.
            costs.setdefault(top, []).append(estimate_tokens(line) + 1)
    if not groups:
        return "# (no symbols indexed)"

    allowance = _fair_share({g: sum(c) for g, c in costs.items()}, max_tokens)
    out: list[str] = []
    dropped_total = 0
    for group in sorted(groups):
        spent, shown = 0, 0
        for line, cost in zip(groups[group], costs[group]):
            if spent + cost > allowance[group] or len(out) >= max_symbols:
                break
            out.append(line)
            spent += cost
            shown += 1
        missing = len(groups[group]) - shown
        if missing:
            dropped_total += missing
            # Say what is missing. A silently truncated map reads to the model
            # like "this is the whole repo", and it stops looking.
            out.append(f"# …{group}: {missing} more symbol(s) not shown (map "
                       f"budget); use grep / tree_view / read_file there")
    if dropped_total:
        shown_total = index.symbol_count - dropped_total
        bound = (f"{max_symbols:,} symbols" if shown_total >= max_symbols
                 else f"~{max_tokens:,} tokens")
        out.append(f"# map truncated to {bound} ({shown_total} of "
                   f"{index.symbol_count} symbols shown)")
    return "\n".join(out)


def build_repo_map(root: Path, max_symbols: int = 5000,
                   languages: Optional[list[str]] = None,
                   use_treesitter: bool = True,
                   exclude: Optional[list[str]] = None,
                   max_tokens: int = DEFAULT_MAP_TOKENS) -> RepoIndex:
    return RepoIndex(root, languages=languages, use_treesitter=use_treesitter,
                     exclude=exclude, max_tokens=max_tokens).scan(max_symbols)
