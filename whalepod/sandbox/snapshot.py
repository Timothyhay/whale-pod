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
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_BACKUP_DIR = Path.home() / ".whalepod" / "backups"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


def backup_root() -> Path:
    env = os.environ.get("WHALEPOD_BACKUP_DIR")
    return Path(env) if env else DEFAULT_BACKUP_DIR


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
                 root: Optional[Path] = None):
        base = Path(backup_dir) if backup_dir else backup_root()
        if session_dir is not None:
            self.session_dir = Path(session_dir)
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.session_dir = base / stamp
        # Created on first snapshot, not here: every WhalePod invocation builds a
        # manager, so eager mkdir left an empty directory behind for read-only
        # commands that never wrote anything.
        self._dir_ready = False
        self.root = Path(root).resolve() if root else None
        self.snapshots: list[Snapshot] = []
        self.rolled_back = False

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
