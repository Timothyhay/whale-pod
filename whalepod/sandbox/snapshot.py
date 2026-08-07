"""Pre-edit snapshots for rollback.

Before any write tool touches a file, its current bytes are copied aside. The
copy is git-agnostic on purpose: we don't use `git stash`, which would collide
with whatever the user has staged.

Every snapshot is also recorded in a ``manifest.json`` next to the backups.
That file is the whole point: ``whalepod rollback`` runs in a *different
process* from the edit session, so an in-memory list would have made the
command a permanent no-op — it would report "nothing to roll back" no matter
how many files had been rewritten. With the manifest, rollback re-opens the
most recent session and undoes it.

Layout::

    <backup dir>/<session>/manifest.json
    <backup dir>/<session>/0001-config.py.bak      # pre-edit bytes

Backup filenames are index-prefixed rather than derived only from the path, so
two different files that flatten to the same name cannot overwrite each other's
backup. Absence of a pre-existing file is recorded in the manifest
(``existed: false``) instead of via a sibling ``.absent`` file, which used to
collide with a real backup whose name differed only by suffix.

Nothing used to remove any of this. One session dir per invocation that wrote a
file accumulates forever, so the retention sweep below runs once per session, at
the moment the session first writes something — see :class:`Retention`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_BACKUP_DIR = Path.home() / ".whalepod" / "backups"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

# Session dirs are named by local timestamp, so the name sorts chronologically
# and is the age of the session. Retention only ever considers directories that
# match it: a backup root is under the user's home, and anything they parked
# there by hand is not ours to delete.
SESSION_STAMP = "%Y%m%d-%H%M%S"
_STAMP_RE = re.compile(r"\d{8}-\d{6}$")


def backup_root() -> Path:
    env = os.environ.get("WHALEPOD_BACKUP_DIR")
    return Path(env) if env else DEFAULT_BACKUP_DIR


@dataclass
class Retention:
    """How much snapshot history to keep. Any limit set to 0 is disabled.

    A snapshot is insurance, not history: it gets used minutes after the edit
    that created it, or never. So the defaults are deliberately short — the
    directory grew without bound because *nothing* expired, not because two
    weeks was too aggressive.

    ``max_sessions`` is counted over the *finished* sessions only. The session
    currently being written is never counted and never deleted, so a full
    directory transiently holds ``max_sessions + 1``.
    """
    max_sessions: int = 20       # keep this many newest sessions
    max_age_days: float = 14.0   # delete sessions older than this
    max_total_mb: float = 0.0    # delete oldest until the tree fits (0 = no cap)


def session_time(session_dir: Path) -> datetime:
    """When a session ran, from its directory name, falling back to mtime.

    The name is preferred because it is what the session recorded for itself:
    copying or restoring a backup tree rewrites mtimes but not names.
    """
    session_dir = Path(session_dir)
    try:
        return datetime.strptime(session_dir.name[-15:], SESSION_STAMP)
    except ValueError:
        pass
    try:
        return datetime.fromtimestamp(session_dir.stat().st_mtime)
    except OSError:
        return datetime.now()


def dir_size(session_dir: Path) -> int:
    total = 0
    try:
        for p in Path(session_dir).iterdir():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def prune_plan(backup_dir: Optional[Path] = None,
               retention: Optional[Retention] = None,
               keep: Iterable[Path] = ()) -> list[tuple[Path, str]]:
    """Which session dirs the policy no longer covers, oldest first.

    Returned as ``(dir, reason)`` so both the sweep and ``/backups`` can say
    *why* something is going away instead of just making it vanish. This is the
    dry run; :func:`prune` is the same selection plus the ``rmtree``.

    Dirs with no manifest are included: a session killed between ``mkdir`` and
    the first manifest write leaves one behind, and excluding them would make
    that debris the one thing retention could never reclaim.
    """
    ret = retention or Retention()
    base = Path(backup_dir) if backup_dir else backup_root()
    if not base.is_dir():
        return []
    kept = set()
    for k in keep:
        try:
            kept.add(Path(k).resolve())
        except OSError:
            pass
    try:
        cand = [d for d in base.iterdir()
                if d.is_dir() and _STAMP_RE.search(d.name)
                and d.resolve() not in kept]
    except OSError:
        return []
    cand.sort(key=lambda d: d.name)                  # oldest first
    doomed: dict[Path, str] = {}

    if ret.max_age_days and ret.max_age_days > 0:
        cutoff = datetime.now() - timedelta(days=ret.max_age_days)
        for d in cand:
            if session_time(d) < cutoff:
                doomed[d] = f"older than {ret.max_age_days:g}d"

    live = [d for d in cand if d not in doomed]
    if ret.max_sessions and ret.max_sessions > 0:
        over = max(0, len(live) - ret.max_sessions)
        for d in live[:over]:
            doomed[d] = f"over {ret.max_sessions} sessions"
        live = live[over:]

    if ret.max_total_mb and ret.max_total_mb > 0:
        budget = ret.max_total_mb * 1024 * 1024
        sizes = {d: dir_size(d) for d in live}
        total = sum(sizes.values())
        # ``live[:-1]``: the newest point is never traded for disk. One session
        # bigger than the whole budget would otherwise take every other point
        # with it and then itself, leaving nothing to roll back to.
        for d in live[:-1]:                          # oldest first
            if total <= budget:
                break
            total -= sizes[d]
            doomed[d] = f"over {ret.max_total_mb:g} MB"

    return [(d, doomed[d]) for d in cand if d in doomed]


def prune(backup_dir: Optional[Path] = None,
          retention: Optional[Retention] = None,
          keep: Iterable[Path] = ()) -> list[tuple[Path, str]]:
    """Apply :func:`prune_plan`. Returns what was actually deleted.

    A dir that cannot be removed (locked file, permissions) is left out of the
    report rather than counted as reclaimed — the next sweep will try again.
    """
    removed: list[tuple[Path, str]] = []
    for d, reason in prune_plan(backup_dir, retention, keep):
        try:
            shutil.rmtree(d)
        except OSError:
            continue
        removed.append((d, reason))
    return removed


@dataclass
class SessionInfo:
    """One backup point, as ``/backups`` needs to show it."""
    dir: Path
    when: datetime
    files: int = 0
    paths: list[str] = field(default_factory=list)
    root: Optional[Path] = None
    rolled_back: bool = False
    bytes: int = 0
    has_manifest: bool = True
    expires: str = ""            # non-empty: the next sweep reclaims this one

    @property
    def session(self) -> str:
        return self.dir.name


def describe_sessions(backup_dir: Optional[Path] = None,
                      retention: Optional[Retention] = None,
                      keep: Iterable[Path] = ()) -> list[SessionInfo]:
    """Every backup point, **newest first** — the order the timeline prints.

    ``retention`` is only read to mark which points are already past it, so the
    listing can warn before the next session silently reclaims them.
    """
    base = Path(backup_dir) if backup_dir else backup_root()
    if not base.is_dir():
        return []
    doomed = dict(prune_plan(base, retention, keep)) if retention else {}
    out: list[SessionInfo] = []
    try:
        dirs = [d for d in base.iterdir()
                if d.is_dir() and _STAMP_RE.search(d.name)]
    except OSError:
        return []
    for d in sorted(dirs, key=lambda p: p.name, reverse=True):
        info = SessionInfo(dir=d, when=session_time(d), bytes=dir_size(d),
                           expires=doomed.get(d, ""))
        mf = d / MANIFEST_NAME
        info.has_manifest = mf.is_file()
        if info.has_manifest:
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            entries = data.get("entries") or []
            info.files = len(entries)
            info.rolled_back = bool(data.get("rolled_back"))
            info.root = Path(data["root"]) if data.get("root") else None
            info.paths = [_relative(e.get("path", ""), info.root)
                          for e in entries]
        out.append(info)
    return out


def _relative(path: str, root: Optional[Path]) -> str:
    """Path as it reads in the session's own repo, absolute if it is elsewhere."""
    if not path:
        return "?"
    p = Path(path)
    if root:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            pass
    return p.as_posix()


@dataclass
class Snapshot:
    path: Path            # absolute path of the snapshotted file
    backup: Optional[Path]  # pre-edit bytes (None when the file did not exist)
    existed: bool         # True if the file existed before the edit
    tool: str = ""        # which tool caused it (for the rollback report)
    ts: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["path"] = str(self.path)
        d["backup"] = str(self.backup) if self.backup else None
        return d

    @staticmethod
    def from_json(d: dict) -> "Snapshot":
        return Snapshot(
            path=Path(d["path"]),
            backup=Path(d["backup"]) if d.get("backup") else None,
            existed=bool(d.get("existed", True)),
            tool=d.get("tool", ""),
            ts=d.get("ts", ""),
        )


class SnapshotManager:
    def __init__(self, backup_dir: Optional[Path] = None,
                 session_dir: Optional[Path] = None,
                 root: Optional[Path] = None,
                 retention: Optional[Retention] = None):
        base = Path(backup_dir) if backup_dir else backup_root()
        if session_dir is not None:
            self.session_dir = Path(session_dir)
        else:
            stamp = datetime.now().strftime(SESSION_STAMP)
            self.session_dir = base / stamp
        # Created on first snapshot, not here: every WhalePod invocation builds a
        # manager, so eager mkdir left an empty directory behind for read-only
        # commands that never wrote anything.
        self._dir_ready = False
        self.root = Path(root).resolve() if root else None
        self.snapshots: list[Snapshot] = []
        self.rolled_back = False
        # ``None`` means "do not sweep" — used by ``load()``, which re-opens an
        # old session to restore it. Reclaiming disk is not that command's job,
        # and an unlucky policy could otherwise delete its neighbours mid-restore.
        self.retention = retention
        self.pruned: list[tuple[Path, str]] = []

    # -- session discovery -------------------------------------------------
    @property
    def session_id(self) -> str:
        return self.session_dir.name

    @classmethod
    def sessions(cls, backup_dir: Optional[Path] = None) -> list[Path]:
        """All session dirs holding a manifest, newest last."""
        base = Path(backup_dir) if backup_dir else backup_root()
        if not base.is_dir():
            return []
        found = [d for d in base.iterdir()
                 if d.is_dir() and (d / MANIFEST_NAME).is_file()]
        return sorted(found, key=lambda d: d.name)

    @classmethod
    def load(cls, session_dir: Path) -> "SnapshotManager":
        """Re-open a recorded session (used by `whalepod rollback`)."""
        session_dir = Path(session_dir)
        mgr = cls(session_dir=session_dir)
        data = {}
        mf = session_dir / MANIFEST_NAME
        if mf.is_file():
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
        mgr.root = Path(data["root"]).resolve() if data.get("root") else None
        mgr.rolled_back = bool(data.get("rolled_back"))
        mgr.snapshots = [Snapshot.from_json(e) for e in data.get("entries", [])]
        return mgr

    @classmethod
    def load_latest(cls, backup_dir: Optional[Path] = None,
                    root: Optional[Path] = None) -> Optional["SnapshotManager"]:
        """Most recent session, optionally restricted to one project root."""
        for d in reversed(cls.sessions(backup_dir)):
            mgr = cls.load(d)
            if not mgr.snapshots:
                continue
            if root is not None and mgr.root is not None:
                if mgr.root != Path(root).resolve():
                    continue
            return mgr
        return None

    # -- recording ---------------------------------------------------------
    def snapshot_file(self, path: Path, tool: str = "") -> Snapshot:
        """Record the pre-edit state of ``path`` (once per path per session).

        The *earliest* observed state wins, so rolling back after several edits
        to one file returns it to how the session found it.
        """
        path = Path(path).resolve()
        for s in self.snapshots:
            if s.path == path:
                return s
        self._ensure_dir()
        existed = path.exists() and path.is_file()
        backup: Optional[Path] = None
        if existed:
            backup = self.session_dir / f"{len(self.snapshots) + 1:04d}-{_safe(path.name)}"
            try:
                shutil.copy2(path, backup)
            except OSError:
                # Unreadable source: record the attempt but admit we cannot
                # restore it, rather than pretending a backup exists.
                backup = None
        snap = Snapshot(path=path, backup=backup, existed=existed, tool=tool,
                        ts=datetime.now().isoformat(timespec="seconds"))
        self.snapshots.append(snap)
        self._write_manifest()
        return snap

    def _ensure_dir(self) -> None:
        if not self._dir_ready:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True
            # Once, here, rather than per snapshot or per startup: this is the
            # one moment we know the tree just grew, and a session that only
            # read files should not pay for a directory scan.
            if self.retention is not None:
                self.pruned = prune(self.session_dir.parent, self.retention,
                                    keep=[self.session_dir])

    def _write_manifest(self) -> None:
        if not self.snapshots and not self._dir_ready:
            return          # nothing was ever snapshotted; don't create a dir
        self._ensure_dir()
        data = {
            "version": MANIFEST_VERSION,
            "session": self.session_id,
            "root": str(self.root) if self.root else None,
            "rolled_back": self.rolled_back,
            "entries": [s.to_json() for s in self.snapshots],
        }
        mf = self.session_dir / MANIFEST_NAME
        tmp = mf.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
            os.replace(tmp, mf)   # atomic: never leave a half-written manifest
        except OSError:
            pass

    # -- rollback ----------------------------------------------------------
    def rollback(self) -> list[str]:
        """Restore every snapshotted file to its pre-session state."""
        report: list[str] = []
        for snap in reversed(self.snapshots):
            if snap.existed:
                if snap.backup and Path(snap.backup).exists():
                    try:
                        snap.path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(snap.backup, snap.path)
                        report.append(f"restored {snap.path}")
                    except OSError as e:
                        report.append(f"FAILED {snap.path}: {e}")
                else:
                    report.append(f"FAILED {snap.path}: backup missing")
            else:
                if snap.path.exists():
                    try:
                        snap.path.unlink()
                        report.append(f"removed new file {snap.path}")
                    except OSError as e:
                        report.append(f"FAILED {snap.path}: {e}")
        if report:
            self.rolled_back = True
            self._write_manifest()
        else:
            report.append("(nothing to roll back)")
        return report

    def rollback_paths(self, paths) -> list[str]:
        """Roll back only the given paths (used to undo one failed plan)."""
        wanted = {Path(p).resolve() for p in paths}
        report: list[str] = []
        for snap in reversed(self.snapshots):
            if snap.path not in wanted:
                continue
            if snap.existed and snap.backup and Path(snap.backup).exists():
                try:
                    shutil.copy2(snap.backup, snap.path)
                    report.append(f"restored {snap.path}")
                except OSError as e:
                    report.append(f"FAILED {snap.path}: {e}")
            elif not snap.existed and snap.path.exists():
                try:
                    snap.path.unlink()
                    report.append(f"removed {snap.path}")
                except OSError as e:
                    report.append(f"FAILED {snap.path}: {e}")
        return report

    def touched_paths(self) -> list[Path]:
        return [s.path for s in self.snapshots]

    def clear(self) -> None:
        self.snapshots.clear()
        self._write_manifest()


def _safe(name: str) -> str:
    keep = "-_.() "
    out = "".join(c if (c.isalnum() or c in keep) else "_" for c in name)
    return (out or "file")[:80] + ".bak"
