"""The system prompt — one copy, used by every entry point.

There were three near-identical prompts in the CLI (one for `ask`, one for the
REPL, one for the status header), which meant `whalepod ask` and the REPL were
talking to differently-briefed agents, and any two of them sharing a session
would each invalidate the other's cached prefix.

The prompt is deliberately static: it is the head of the prefix-cached zone, so
anything session-specific (cwd, mode, time) must NOT go in here. The repo map
follows it as a separate section, and volatile facts belong in the user turn.

Per-tool usage rules are *not* written here. They are declared next to each tool
definition in :mod:`whalepod.tools.registry` and assembled into the "Using the
tools" section by :func:`build_system_prompt`. That way a tool that is not
offered this session (writes in a readonly sandbox) does not leave three
paragraphs of instructions about it in the prompt, and a newly added tool cannot
ship without its rules. The idea comes from the pi reference agent.
"""
from __future__ import annotations

from typing import Iterable, Optional

PREAMBLE = """\
You are WhalePod, a command-line coding agent operating directly on the user's \
repository.\
"""

CLOSING = """\
Answering
- Be concise and concrete. Reference code as `path:line`.
- Report what you actually did and what you verified. If a test failed or you \
skipped part of the task, say so plainly.\
"""

# Used when a caller supplies no tool guidance (tests, and any embedding that has
# not wired a registry). Keeping the prompt usable without a registry matters: the
# alternative is a prompt with a heading and nothing under it.
DEFAULT_GUIDELINES = [
    "Look before you leap: read the relevant files with the tools before "
    "editing them. Never guess at a file's contents.",
    "Prefer several small, verifiable steps over one large speculative change.",
]


def _section(guidelines: Iterable[str]) -> str:
    lines = [f"- {g}" for g in guidelines if g]
    if not lines:
        return ""
    return "Using the tools\n" + "\n".join(lines)


def build_system_prompt(extra: str = "",
                        guidelines: Optional[Iterable[str]] = None) -> str:
    """The canonical prompt for the tools actually on offer.

    ``guidelines`` comes from :meth:`ToolRegistry.guidelines`. ``extra`` is for
    durable project context (``WHALEPOD.md`` / ``AGENTS.md`` / ``CLAUDE.md``),
    not for per-turn state — both are part of the cached prefix, so both must be
    stable for the whole session.
    """
    gl = DEFAULT_GUIDELINES if guidelines is None else list(guidelines)
    parts = [PREAMBLE, _section(gl), CLOSING]
    if extra:
        parts.append(f"Project-specific instructions\n{extra.strip()}")
    return "\n\n".join(p for p in parts if p) + "\n"


# Backwards compatibility for callers that imported the constant directly.
SYSTEM_PROMPT = build_system_prompt()


def repo_map_section(repo_map_text: str) -> str:
    """Wrap the repo map so the model knows what it is looking at."""
    if not repo_map_text:
        return ""
    return ("Repository map (symbol signatures only — read files for bodies)\n"
            f"{repo_map_text.strip()}\n")
