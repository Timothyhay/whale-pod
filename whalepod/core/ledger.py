"""Context ledger — what the model has already been shown.

The thing that actually wastes a large window is not one big file, it is the
*same* file arriving three times because the model forgot it already asked.
Every copy is permanent (history is append-only), so a duplicate read costs
tokens for the rest of the session.

The ledger records each file range that has been delivered into the context,
along with the file's identity at the time (mtime + size). When a read repeats
a range that is still current, the agent answers with a short pointer instead
of the full text — cheap, and it keeps the prefix stable. When the file has
changed on disk, or WhalePod itself edited it, the entry is invalidated and a
genuine re-read is allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def file_identity(path: Path) -> tuple[int, int]:
    """(mtime_ns, size) — cheap staleness key. (0, -1) if unreadable."""
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, -1)


@dataclass
class LedgerEntry:
    path: str                 # repo-relative, forward slashes
    start: int                # 1-based inclusive; 0 => whole file (legacy)
    end: int                  # inclusive; 0 => whole file (legacy)
    identity: tuple           # (mtime_ns, size) when delivered
    turn: int                 # which agent turn delivered it
    tokens: int = 0
    # id of the tool result that carried the content. The ledger's claim is
    # "this is still in the window", so when pruning drops that message the
    # claim stops being true and the entry has to go with it.
    message_id: str = ""
    # True only when the *entire* file went into the window. A truncated read
    # delivers a concrete range with complete=False, so a later request for the
    # whole file is a genuine miss instead of being answered with a pointer to
    # content that was cut off.
    complete: bool = False

    @property
    def whole_file(self) -> bool:
        return self.complete or (self.start == 0 and self.end == 0)

    def covers(self, start: int, end: int) -> bool:
        if start == 0 and end == 0:
            return self.whole_file          # asking for the whole file
        if self.start == 0 and self.end == 0:
            return True                     # legacy whole-file entry
        return self.start <= start and end <= self.end

    def label(self) -> str:
        if self.start == 0 and self.end == 0:
            return self.path
        if self.complete:
            return f"{self.path} (all {self.end} lines)"
        return f"{self.path}:{self.start}-{self.end}"


@dataclass
class ContextLedger:
    entries: list[LedgerEntry] = field(default_factory=list)
    turn: int = 0
    hits: int = 0
    saved_tokens: int = 0

    # -- recording -------------------------------------------------------
    def note_read(self, path: str, start: int, end: int, identity: tuple,
                  tokens: int = 0, message_id: str = "",
                  complete: bool = False) -> LedgerEntry:
        # Normalized on the way in as well as on lookup: storing "./f.py" and
        # querying "f.py" would never match, silently disabling the ledger.
        e = LedgerEntry(path=_norm(path), start=start, end=end,
                        identity=identity, turn=self.turn, tokens=tokens,
                        message_id=message_id, complete=complete)
        self.entries.append(e)
        return e

    def invalidate(self, path: str) -> None:
        """Forget everything known about a path (it was written or deleted)."""
        norm = _norm(path)
        self.entries = [e for e in self.entries if e.path != norm]

    def forget_messages(self, message_ids) -> int:
        """Drop entries delivered by messages that are no longer in the window.

        Pruning removes whole turns, tool results included. Without this the
        ledger goes on answering a re-read with "it is already above — scroll
        up" for content that was elided, and the model is stuck arguing with a
        window that no longer contains the file.
        """
        ids = {i for i in message_ids if i}
        if not ids:
            return 0
        before = len(self.entries)
        self.entries = [e for e in self.entries
                        if not (e.message_id and e.message_id in ids)]
        return before - len(self.entries)

    # -- querying --------------------------------------------------------
    def find_current(self, path: str, start: int, end: int,
                     identity: tuple) -> Optional[LedgerEntry]:
        """An entry covering this range whose file is unchanged, else None."""
        norm = _norm(path)
        for e in reversed(self.entries):
            if e.path != norm:
                continue
            if e.identity != identity:
                continue                    # file changed on disk -> stale
            if e.covers(start, end):
                return e
        return None

    def record_hit(self, entry: LedgerEntry) -> None:
        self.hits += 1
        self.saved_tokens += entry.tokens

    def summary(self) -> str:
        if not self.entries:
            return "(nothing loaded yet)"
        return ", ".join(e.label() for e in self.entries[-24:])

    def stats_line(self) -> str:
        return (f"{len(self.entries)} range(s) loaded · {self.hits} duplicate "
                f"read(s) avoided (~{self.saved_tokens:,} tok saved)")


def _norm(path: str) -> str:
    """Repo-relative, forward slashes, no "./" prefix.

    ``lstrip("./")`` would be wrong here — it strips a *set of characters*, so
    ".github/workflows/ci.yml" would become "github/workflows/ci.yml".
    """
    p = str(path).replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p
