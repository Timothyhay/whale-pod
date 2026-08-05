"""Download the DeepSeek V4 tokenizer (just ``tokenizer.json``) for accurate
benchmark / runtime token accounting.

The offline bench and the runtime estimator prefer the *real* DeepSeek V4 BPE
over tiktoken's ``cl100k_base`` or the character heuristic, because its
vocabulary differs and that changes both token counts and 64-token cache-block
alignment. Only the tokenizer file is fetched (no model weights, ~tens of MB),
so this is a one-time setup:

    python bench/fetch_tokenizer.py
    # or point the dump elsewhere / reuse an existing file
    python bench/fetch_tokenizer.py --repo deepseek-ai/DeepSeek-V4-Flash-0731

It needs network, but the *bench itself does not*: once JSON is on disk,
``python bench/validate.py`` runs fully offline with exact token counts.

The path resolved by the runtime estimator (default `~/.whalepod/tokenizers/
deepseek-v4-flash/tokenizer.json`) can be overridden with `WHALEPOD_TOKENIZER_JSON`.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from whalepod.core.tokenizer import DEEPSEEK_V4_REPO, dsv4_tokenizer_path


def download(repo: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        raise SystemExit(
            f"{dest} already exists (use --force to overwrite). "
            f"Set WHALEPOD_TOKENIZER_JSON to use a different file.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/tokenizer.json"
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - https
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=DEEPSEEK_V4_REPO,
                    help="HF repo owning the tokenizer.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path (default: the estimator's lookup path)")
    ap.add_argument("--force", action="store_true", help="overwrite existing file")
    args = ap.parse_args(argv)
    dest = args.out or dsv4_tokenizer_path()
    download(args.repo, dest, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())