"""Local token estimation for context accounting.

Tokenizer resolution order:
  1. **DeepSeek V4 official tokenizer** — a real BPE encoder whose vocabulary
     differs from tiktoken's ``cl100k_base`` (GPT-4's). Used for the benchmark so
     token counts and 64-token cache-block alignment match what the server
     reports. Loaded lazily from a local ``tokenizer.json`` (no network at
     runtime) — see ``bench/fetch_tokenizer.py``.
  2. **tiktoken** — byte-accurate for OpenAI-style models.
  3. **heuristic** — fast fallback that accounts for CJK vs. general text.

Used only for context_stats / warnings and for benchmark accounting — never for
billing accuracy (the provider's ``usage`` block is the only billing source).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_HAS_TIKTOKEN = False
_TRY_TIKTOKEN = True
_enc_cache = None

# The canonical DeepSeek V4 tokenizer lives on the HF repo below; Flash and Pro
# share one vocabulary. Only ``tokenizer.json`` is needed, not the weights.
DEEPSEEK_V4_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_DSV4_SUBDIR = "deepseek-v4-flash"

_dsv4_counter = False       # False = not loaded/absent; callable once loaded
DSV4_PATH_ENV = "WHALEPOD_TOKENIZER_JSON"

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


def dsv4_tokenizer_path() -> Path:
    """Where the DeepSeek V4 tokenizer is looked up: env > ~/.whalepod."""
    val = os.environ.get(DSV4_PATH_ENV)
    if val:
        return Path(val)
    return (Path.home() / ".whalepod" / "tokenizers"
            / DEFAULT_DSV4_SUBDIR / "tokenizer.json")


def _try_load_dsv4() -> bool:
    """Load the DeepSeek V4 encoder once, from a local file only. Returns
    True iff a real encoder is now available. Never touches the network."""
    global _dsv4_counter
    if _dsv4_counter is not False:
        return bool(_dsv4_counter)
    path = dsv4_tokenizer_path()
    try:
        if not path.is_file():
            _dsv4_counter = False
            return False
        # Prefer the lightweight `tokenizers` package; fall back to `transformers`
        # for full feature coverage. Both are local reads.
        try:
            from tokenizers import Tokenizer  # type: ignore
            _dsv4_counter = Tokenizer.from_file(str(path)).encode
        except ImportError:
            from transformers import AutoTokenizer  # type: ignore
            _dsv4_counter = AutoTokenizer.from_pretrained(
                str(path), local_files_only=True).encode
        return True
    except Exception:
        _dsv4_counter = False
        return False


def _get_tiktoken():
    global _HAS_TIKTOKEN, _enc_cache, _TRY_TIKTOKEN
    if not _TRY_TIKTOKEN:
        return None
    try:
        import tiktoken  # noqa: F401
        if _enc_cache is None:
            _enc_cache = tiktoken.get_encoding("cl100k_base")
        _HAS_TIKTOKEN = True
        return _enc_cache
    except Exception:
        _TRY_TIKTOKEN = False
        return None


def active_tokenizer_name() -> str:
    """Describe the estimator in effect, for result metadata."""
    if _try_load_dsv4():
        return f"deepseek-v4 ({Path(dsv4_tokenizer_path()).name})"
    if _get_tiktoken() is not None:
        return "tiktoken cl100k_base"
    return "heuristic"


def estimate_tokens_dsv4(text: str) -> Optional[int]:
    """Count tokens with the real DeepSeek V4 encoder, or None if unavailable."""
    if text is None:
        text = ""
    if not text:
        return 0
    if _try_load_dsv4():
        try:
            return len(_dsv4_counter(text))
        except Exception:
            return None
    return None


def estimate_tokens(text: str) -> int:
    """Estimate token count of a string. Returns >= 0.

    Prefers the DeepSeek V4 BPE when its tokenizer is available locally (the
    benchmark's measurements and the runtime's prune decisions should both agree
    with the server). Falls back to tiktoken, then a fast heuristic.
    """
    if text is None:
        text = ""
    if not text:
        return 0
    if _try_load_dsv4():
        try:
            return len(_dsv4_counter(text))
        except Exception:
            pass  # fall through
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
