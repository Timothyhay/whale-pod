"""Local token estimation for context accounting.

Prefers a byte-accurate encoder (tiktoken for OpenAI/DeepSeek-style BPE) when
available; otherwise falls back to a fast heuristic that accounts for CJK
characters (roughly 1 token each) vs. general text (English-ish rules).
Used only for context_stats / warnings — never for billing accuracy.
"""
from __future__ import annotations

import re

_HAS_TIKTOKEN = False
_TRY_TIKTOKEN = True
_enc_cache = None

_CHAR_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # Extension A
    (0x3040, 0x30FF),   # Hiragana + Katakana
    (0xAC00, 0xD7AF),   # Hangul
    (0x3130, 0x318F),   # Hangul Jamo
)


def _has_cjk(char: str) -> bool:
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _CHAR_RANGES)


def _get_tiktoken():
    global _HAS_TIKTOKEN, _enc_cache, _TRY_TIKTOKEN
    if not _TRY_TIKTOKEN:
        return None
    try:
        import tiktoken  # noqa: F401
        # DeepSeek uses cl100k_base-compatible tokenizer for chat
        if _enc_cache is None:
            _enc_cache = tiktoken.get_encoding("cl100k_base")
        _HAS_TIKTOKEN = True
        return _enc_cache
    except Exception:
        _TRY_TIKTOKEN = False
        return None


def estimate_tokens(text: str) -> int:
    """Estimate token count of a string. Returns >= 0."""
    if not text:
        return 0
    enc = _get_tiktoken()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass  # fall through to heuristic
    # Heuristic: CJK ~1 token/char, general text ~1 token per 4 chars,
    # split words/whitespace conservative.
    cjk = sum(1 for ch in text if _has_cjk(ch))
    rest = len(text) - cjk
    return cjk + max(1, int(rest / 3.2))


def estimate_messages(messages) -> int:
    """Rough tokens for a list of Message-like objects (with .content, .role)."""
    n = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        role = getattr(m, "role", "")
        n += estimate_tokens(content) + 3  # per-message overhead
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            n += estimate_tokens(str(tcs))
    return n
