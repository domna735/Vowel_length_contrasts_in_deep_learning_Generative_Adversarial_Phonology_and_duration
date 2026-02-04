"""Select representative generated samples for a deliverables package.

Selection rule (simple, reproducible): pick N samples whose VOT is closest to the
class median.

Typical usage (100-epoch eval example):
  python tools\select_representative_samples.py \
    --long-vot-csv runs\vot_100ep_long_250.csv  --long-audio-dir runs\gen\ciwgan_eval_long \
    --short-vot-csv runs\vot_100ep_short_250.csv --short-audio-dir runs\gen\ciwgan_eval_short \
    --out runs\deliverables\vietnamese_100ep\samples --n 5

The output folder will contain copied WAVs and a manifest CSV.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional


@dataclass(frozen=True)
class VotRow:
    rel_path: str
    vot_ms: float


def read_vot_csv(path: Path) -> list[VotRow]:
    rows: list[VotRow] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            return rows
        for row in r:
            rel_path = (row.get("rel_path") or "").strip()
            vot_str = (row.get("vot_ms") or "").strip()
            if not rel_path or not vot_str:
                continue
            try:
                vot_ms = float(vot_str)
            except ValueError:
                continue
            rows.append(VotRow(rel_path=rel_path, vot_ms=vot_ms))
    return rows


def resolve_audio_path(audio_dir: Path, rel_path: str) -> Optional[Path]:
    # 1) direct join
    candidate = audio_dir / rel_path
    if candidate.exists():
        return candidate

    # 2) join by basename
    name = Path(rel_path).name
    candidate = audio_dir / name
    if candidate.exists():
        return candidate

    # 3) slow search by basename
    matches = list(audio_dir.rglob(name))
    for m in matches:
        if m.is_file():
            return m
    return None


def pick_closest_to_median(rows: list[VotRow], n: int) -> tuple[float, list[VotRow]]:
    if not rows:
        return 0.0, []
    m = median([r.vot_ms for r in rows])
    chosen = sorted(rows, key=lambda r: abs(r.vot_ms - m))[:n]
    return float(m), chosen


def copy_set(label: str, vot_csv: Path, audio_dir: Path, out_dir: Path, n: int, manifest_rows: list[dict]) -> None:
    rows = read_vot_csv(vot_csv)
    m, chosen = pick_closest_to_median(rows, n=n)

    print(f"[{label}] vot_csv={vot_csv} audio_dir={audio_dir}")
    print(f"[{label}] total_rows={len(rows)} median_vot_ms={m:.2f} selecting={len(chosen)}")

    for idx, r in enumerate(chosen):
        src = resolve_audio_path(audio_dir, r.rel_path)
        if src is None:
            print(f"[{label}] WARNING: audio not found for {r.rel_path}")
            continue

        dst_name = f"{label.upper()}_{idx:02d}_{src.name}"
        dst = out_dir / dst_name
        shutil.copy2(src, dst)

        manifest_rows.append(
            {
                "label": label,
                "selected_index": idx,
                "rel_path": r.rel_path,
                "vot_ms": f"{r.vot_ms:.4f}",
                "class_median_vot_ms": f"{m:.4f}",
                "copied_as": dst.name,
            }
        )


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["label", "selected_index", "rel_path", "vot_ms", "class_median_vot_ms", "copied_as"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Select representative samples closest to class median VOT.")
    ap.add_argument("--long-vot-csv", required=True)
    ap.add_argument("--long-audio-dir", required=True)
    ap.add_argument("--short-vot-csv", required=True)
    ap.add_argument("--short-audio-dir", required=True)
    ap.add_argument("--out", required=True, help="Output folder to copy WAVs into")
    ap.add_argument("--n", type=int, default=5, help="Number per class")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    copy_set("long", Path(args.long_vot_csv), Path(args.long_audio_dir), out_dir, args.n, manifest_rows)
    copy_set("short", Path(args.short_vot_csv), Path(args.short_audio_dir), out_dir, args.n, manifest_rows)

    manifest_path = out_dir / "representative_samples_manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"Wrote manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
