"""Text-file decoding that survives Windows checkouts.

Every tool that matches model-supplied text against file content has to agree on
what "the file's text" is, and on Windows the naive answer is wrong twice over:

  * **Line endings.** ``read_file`` renders content line by line, so what the
    model sees — and copies back into ``edit_file``'s ``old`` — is LF-joined. A
    file checked out with CRLF then fails ``text.count(old)`` for *any*
    multi-line ``old``: the needle has "\\n" where the haystack has "\\r\\n".
    Single-line edits worked, multi-line edits never did, and the error message
    ("`old` text not found — read the file again") sent the model round the same
    loop. So matching happens on an LF-normalised copy and the file's original
    ending is restored on write, which also keeps a one-line edit from showing up
    as a whole-file rewrite in the user's `git diff`.

  * **BOM.** A UTF-8 BOM decodes to a U+FEFF at the very start of the text. It
    is invisible in the read output, so the model never includes it in ``old``,
    and an edit anchored at the first line of the file could not match. It is
    stripped for matching and put back on write.

``read_file`` and the write planners both go through here so they cannot
disagree about what is in the file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BOM = "﻿"


@dataclass
class TextFile:
    """A file's content, normalised for matching, plus how to write it back."""
    text: str                    # BOM-stripped, LF line endings
    bom: str = ""                # "" or the BOM that was present
    newline: str = "\n"          # dominant original ending: "\n" or "\r\n"

    def restore(self, new_text: str) -> str:
        """Put ``new_text`` back into the file's own byte conventions."""
        out = new_text.replace("\r\n", "\n").replace("\r", "\n")
        if self.newline != "\n":
            out = out.replace("\n", self.newline)
        return self.bom + out

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")


def decode(data: bytes) -> Optional[TextFile]:
    """Decode file bytes, or None if they are not UTF-8 text."""
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    bom = ""
    if raw.startswith(BOM):
        bom, raw = BOM, raw[len(BOM):]
    crlf = raw.count("\r\n")
    lf = raw.count("\n") - crlf
    # Majority wins for mixed files: rewriting the minority to match is a
    # smaller, more honest diff than leaving the file in two conventions.
    newline = "\r\n" if crlf > lf else "\n"
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return TextFile(text=text, bom=bom, newline=newline)


def read(path: Path) -> tuple[Optional[TextFile], str]:
    """(TextFile, error). Binary or unreadable files are refused, not mangled."""
    try:
        data = path.read_bytes()
    except OSError as e:
        return None, str(e)
    tf = decode(data)
    if tf is None:
        return None, "file is not utf-8 text"
    return tf, ""


__all__ = ["TextFile", "decode", "read", "BOM"]
