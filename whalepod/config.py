"""Configuration, API-key management, and endpoint resolution for WhalePod.

Priority (highest first):
  1. command-line flags (--endpoint, --api-key, --model)
  2. environment variables (WHALEPOD_*, or provider-specific like HF_TOKEN)
  3. project config  (.whalepod.json  in cwd)
  4. global config    (~/.whalepod/config.json)
  5. defaults
"""
from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

CONFIG_DIR_ENV = "WHALEPOD_CONFIG_DIR"
DEFAULT_CONFIG_DIR = Path.home() / ".whalepod"
DEFAULT_CONFIG_FILE = "config.json"
PROJECT_CONFIG_FILE = ".whalepod.json"

# provider -> list of env vars to check, in order
_ENV_FOR_TYPE = {
    "deepseek": ["WHALEPOD_API_KEY", "DEEPSEEK_API_KEY"],
    "custom":   ["WHALEPOD_API_KEY"],
    "openai":   ["WHALEPOD_API_KEY", "OPENAI_API_KEY"],
    "anthropic": ["WHALEPOD_API_KEY", "ANTHROPIC_API_KEY"],
}

_DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "custom": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "openai": "gpt-5",
    "anthropic": "claude-sonnet-5",
}

# Public defaults only. A self-hosted / custom endpoint has no sensible default
# and must be configured.
_DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "custom": "",
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
}


@dataclass
class RepoMapConfig:
    languages: list[str] = field(default_factory=lambda: ["python", "javascript", "typescript", "go"])
    # max_tokens is the real budget: the map lives in the stable prefix, and a
    # symbol count says nothing about what it costs there (2,000 TypeScript
    # signatures rendered to ~70k tokens). max_symbols is only a backstop.
    # 0 = auto, scaled to the context window (see Config.resolved_map_tokens):
    # what a map costs only means anything relative to the window it sits in,
    # and a fixed default that is 1.5% of a 1M window is 25% of a 32k one.
    max_tokens: int = 0
    max_symbols: int = 5000
    use_treesitter: bool = True  # falls back to regex if unavailable
    # Extra ignore globs, on top of .gitignore — for vendored/reference trees
    # that are checked in but are not the code being worked on.
    exclude: list[str] = field(default_factory=list)


@dataclass
class EndpointConfig:
    type: str = "custom"                  # deepseek | custom | openai | anthropic
    # No default URL: one developer's private HuggingFace endpoint used to be
    # baked in here, so a fresh checkout silently pointed every request at it.
    # An unset URL is resolved per provider (see resolved_base_url) or reported.
    base_url: str = ""
    api_key: Optional[str] = None
    api_key_env: str = ""                 # if set, env var name to read key from
    model: Optional[str] = None
    extra_headers: dict = field(default_factory=dict)
    # Extra JSON fields merged into every request body. Needed to hold provider
    # affinity on a router: a prefix cache is one server's KV cache, so requests
    # spread across providers are all cold misses (measured: ~0% vs 98%). On
    # OpenRouter that means
    #   "extra_body": {"provider": {"order": ["DeepInfra"],
    #                               "allow_fallbacks": false}}
    extra_body: dict = field(default_factory=dict)
    timeout: float = 120.0


@dataclass
class Config:
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    default_mode: str = "thinking"        # thinking | instant
    sandbox: str = "confirm"              # confirm | readonly | none | yes
    repo_map: RepoMapConfig = field(default_factory=RepoMapConfig)
    theme: str = "whale"
    # Print each tool call and its result as the turn runs. On by default: a turn
    # that reads six files and runs the tests is mostly work the model never
    # mentions in its prose, and without the trace the user cannot see it happen.
    show_tool_calls: bool = True
    history_file: str = ""                # if empty, default to config dir / history
    context_window: int = 1_000_000       # DeepSeek V4
    # Pruning thresholds, as fractions of the window. These replace the old
    # `warn_threshold`, which nothing ever read: a threshold that only prints a
    # warning does not keep the context inside the window.
    prune_at: float = 0.90                # start dropping turns above this fill
    prune_to: float = 0.50                # ...and go this low in one pass
    # Absolute headroom floor, on top of prune_at. A fraction does not know how
    # big a reply is: 0.90 of 1M leaves 100k, but 0.90 of a 32k window leaves
    # 3.2k, which one reasoning reply plus one tool result overruns — and the
    # request then fails with a context-length error instead of reducing.
    reserve_tokens: int = 16_384
    # Summarize the turns being cut instead of just deleting them. One extra
    # small call per reduction; falls back to plain dropping if it fails.
    compaction: bool = True
    compaction_max_tokens: int = 2_000
    _loaded_from: list[str] = field(default_factory=list, repr=False)

    # -- helpers ---------------------------------------------------------
    def resolved_api_key(self) -> Optional[str]:
        """API key resolution: flag(api_key) > env(var named api_key_env) >
        provider env vars > config-file value."""
        if self.endpoint.api_key:
            return self.endpoint.api_key
        if self.endpoint.api_key_env:
            v = os.environ.get(self.endpoint.api_key_env)
            if v:
                return v
        for var in _ENV_FOR_TYPE.get(self.endpoint.type, []):
            v = os.environ.get(var)
            if v:
                return v
        return None

    def resolved_model(self) -> str:
        return self.endpoint.model or _DEFAULT_MODELS.get(self.endpoint.type, "deepseek-chat")

    def resolved_map_tokens(self) -> int:
        """Token budget for the repo map: explicit value, else ~1.5% of window.

        Clamped to [4k, 16k]. Below 4k the map stops being a usable index of
        anything; above 16k it stops being a *map* and becomes the thing lazy
        loading was supposed to avoid.
        """
        if self.repo_map.max_tokens > 0:
            return self.repo_map.max_tokens
        return max(4_000, min(16_000, int(self.context_window * 0.015)))

    def resolved_base_url(self) -> str:
        """Base URL from config/env, or the provider's public default."""
        url = (self.endpoint.base_url
               or os.environ.get("WHALEPOD_BASE_URL", "")).strip()
        return url or _DEFAULT_BASE_URLS.get(self.endpoint.type, "")

    def check(self) -> None:
        """Raise a message the user can act on, instead of failing at request time."""
        if not self.resolved_base_url():
            raise ValueError(
                f"no endpoint URL configured for type {self.endpoint.type!r}.\n"
                f"Run `whalepod auth`, or set WHALEPOD_BASE_URL.")
        if not self.resolved_api_key():
            raise ValueError(
                f"no API key found for {self.endpoint.type!r}. Run "
                f"`whalepod auth`, or set "
                f"{' / '.join(_ENV_FOR_TYPE.get(self.endpoint.type, []))}.")


# ---------------------------------------------------------------- io ---
def config_dir() -> Path:
    return Path(os.environ.get(CONFIG_DIR_ENV, str(DEFAULT_CONFIG_DIR)))


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(cli_overrides: Optional[dict] = None) -> Config:
    """Load config merging global -> project -> cli overrides."""
    gdir = config_dir()
    paths = [gdir / DEFAULT_CONFIG_FILE, Path.cwd() / PROJECT_CONFIG_FILE]
    merged: dict = {}
    loaded_from: list[str] = []
    for p in paths:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            merged = _deep_merge(merged, data)
            loaded_from.append(str(p))
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    cfg = _from_dict(Config(), merged)
    cfg._loaded_from = loaded_from or ["(defaults)"]
    return cfg


def _from_dict(cfg: Config, data: dict) -> Config:
    for k, v in data.items():
        if k == "endpoint" and isinstance(v, dict):
            for ek, ev in v.items():
                if hasattr(cfg.endpoint, ek):
                    setattr(cfg.endpoint, ek, ev)
        elif k == "repo_map" and isinstance(v, dict):
            for rk, rv in v.items():
                if hasattr(cfg.repo_map, rk):
                    setattr(cfg.repo_map, rk, rv)
        elif hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def save_global_config(cfg: Config) -> Path:
    data = {
        "endpoint": asdict(cfg.endpoint),
        "default_mode": cfg.default_mode,
        "sandbox": cfg.sandbox,
        "repo_map": asdict(cfg.repo_map),
        "theme": cfg.theme,
        "show_tool_calls": cfg.show_tool_calls,
        "context_window": cfg.context_window,
        "prune_at": cfg.prune_at,
        "prune_to": cfg.prune_to,
        "reserve_tokens": cfg.reserve_tokens,
        "compaction": cfg.compaction,
        "compaction_max_tokens": cfg.compaction_max_tokens,
    }
    gdir = config_dir()
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / DEFAULT_CONFIG_FILE
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 600
    except OSError:
        pass
    return path


def ensure_secret_permissions(path: Path) -> None:
    """Best-effort 0600 on files that may contain API keys."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
