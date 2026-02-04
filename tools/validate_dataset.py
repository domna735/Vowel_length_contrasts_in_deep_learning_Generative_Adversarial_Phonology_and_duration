"""Validate dataset folder structure and class balance.

This script is intentionally lightweight: by default it does NOT decode audio.
It scans a dataset root (default: language_mp3/) and reports counts by:
- language (top-level folder under root)
- duration class (inferred from path: 'long' or 'short')

Example:
  python tools\validate_dataset.py --root language_mp3 --out runs\dataset_summary.csv

Notes:
- Class inference matches training conventions used elsewhere in this project:
  any parent directory containing 'long' => class=long
  any parent directory containing 'short' => class=short
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@dataclass(frozen=True)
class Row:
    language: str
    duration_class: str  # long|short|unknown
    rel_path: str


def iter_audio_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def infer_class(path: Path) -> str:
    lowered_parts = [p.lower() for p in path.parts]
    if any("long" in p for p in lowered_parts):
        return "long"
    if any("short" in p for p in lowered_parts):
        return "short"
    return "unknown"


def infer_language(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    # language is first path segment under root
    return rel.parts[0] if rel.parts else "(root)"


def write_csv(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "duration_class", "rel_path"])
        for r in rows:
            w.writerow([r.language, r.duration_class, r.rel_path])


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Validate dataset: counts by language and duration class.")
    p.add_argument("--root", default="language_mp3", help="Dataset root folder (default: language_mp3)")
    p.add_argument("--out", default="", help="Optional CSV output path for per-file rows")
    p.add_argument(
        "--exts",
        default=",".join(sorted(AUDIO_EXTS)),
        help="Comma-separated audio extensions to include",
    )
    args = p.parse_args(argv)

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root folder not found: {root}")

    exts = {e.strip().lower() for e in str(args.exts).split(",") if e.strip()}

    rows: list[Row] = []
    counts_by_lang: Counter[str] = Counter()
    counts_by_lang_class: dict[str, Counter[str]] = defaultdict(Counter)
    counts_by_class: Counter[str] = Counter()

    for fp in iter_audio_files(root, exts):
        language = infer_language(root, fp)
        duration_class = infer_class(fp)
        rel_path = fp.relative_to(root).as_posix()

        rows.append(Row(language=language, duration_class=duration_class, rel_path=rel_path))
        counts_by_lang[language] += 1
        counts_by_lang_class[language][duration_class] += 1
        counts_by_class[duration_class] += 1

    total = len(rows)
    print(f"Scanned root: {root}")
    print(f"Total audio files: {total}")
    print("\nOverall by class:")
    for k in ("long", "short", "unknown"):
        if counts_by_class[k]:
            print(f"  {k:7s}: {counts_by_class[k]}")

    print("\nBy language:")
    for language in sorted(counts_by_lang.keys()):
        lc = counts_by_lang_class[language]
        long_n = lc.get("long", 0)
        short_n = lc.get("short", 0)
        unknown_n = lc.get("unknown", 0)
        ratio = (long_n / short_n) if short_n else float("inf") if long_n else 0.0
        ratio_str = f"{ratio:.2f}" if ratio != float("inf") else "inf"
        print(
            f"  {language}: total={counts_by_lang[language]} long={long_n} short={short_n} unknown={unknown_n} long:short={ratio_str}"
        )

    if args.out:
        out_path = Path(args.out)
        write_csv(rows, out_path)
        print(f"\nWrote per-file CSV: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
