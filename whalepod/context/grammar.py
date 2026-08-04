"""Tree-sitter grammar loading with graceful regex fallback.

Loads compiled language grammars lazily. If tree-sitter or a specific grammar
is unavailable, that language degrades to a regex-based extractor so Repo Map
still works everywhere.
"""
from __future__ import annotations

import re

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    _TS = True
except Exception:
    _TS = False
    Language = None
    Parser = None


_LOADERS = {}
_tried = set()
_parsers = {}


def _load_python():
    import tree_sitter_python
    return Language(tree_sitter_python.language())


def _load_javascript():
    import tree_sitter_javascript
    return Language(tree_sitter_javascript.language())


def _load_go():
    import tree_sitter_go
    return Language(tree_sitter_go.language())


def get_language(lang: str):
    """Return (tree_sitter Language | None, parser | None). None => use regex."""
    if not _TS:
        return None, None
    if lang in _tried:
        return _parsers.get(lang), (_parsers.get(lang) is not None)
    _tried.add(lang)
    loader = {
        "python": _load_python,
        "javascript": _load_javascript,
        "typescript": _load_javascript,   # TS-JS grammar covers JS; TS uses same
        "go": _load_go,
    }.get(lang)
    if loader is None:
        return None, None
    try:
        lang_obj = loader()
        parser = Parser(lang_obj)
        _parsers[lang] = parser
        return parser, True
    except Exception:
        return None, False


def has_treesitter() -> bool:
    return _TS


# ----------------------------------------------------------- regex fallback
_EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
}


def language_for_file(path: str) -> str | None:
    import os
    ext = os.path.splitext(path)[1].lower()
    return _EXT_LANG.get(ext)


def all_supported_langs() -> list[str]:
    return ["python", "javascript", "typescript", "go"]
