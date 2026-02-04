"""Package a lightweight deliverables bundle for sharing.

This script gathers selected artifacts (reports, CSVs, plots, sample audio) into
an output folder and creates a ZIP.

Design goals:
- Work even if some optional inputs are missing (skip with warnings)
- Avoid accidentally packaging huge training artifacts (checkpoints/tensorboard)

Example:
  python tools\package_deliverables.py --tag vietnamese_100ep --out runs\deliverables\vietnamese_100ep

By default, it will try to include:
- Root-level docs (if present): README.md, SIMILARITY_RESULTS_SUMMARY.md, detail report.md
- PDFs/DOCX (if present): COMPLETE_EVALUATION_REPORT.*, Similarity Results.*
- runs/plots/*.png (if present)
- runs/compare/*.csv (if present)
- runs/*vot*.csv and runs/*intensity*.csv (if present)
- runs/deliverables/<tag>/samples/* (if present)
- runs/deliverables/<tag>/PHONOLOGICAL_INTERPRETATION_SUPPLEMENT.md (if present)

It writes output to:
  <out>/package/
  <out>/<tag>_deliverables.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional
import zipfile


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_many(files: Iterable[Path], dst_dir: Path, base_dir: Optional[Path] = None) -> int:
    copied = 0
    for src in files:
        if not src.exists() or not src.is_file():
            continue

        if base_dir is not None:
            rel = src.relative_to(base_dir)
            dst = dst_dir / rel
        else:
            dst = dst_dir / src.name

        if safe_copy(src, dst):
            copied += 1
    return copied


def glob_existing(root: Path, pattern: str) -> list[Path]:
    return [p for p in root.glob(pattern) if p.exists() and p.is_file()]


def zip_dir(folder: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for fp in folder.rglob("*"):
            if fp.is_file():
                z.write(fp, arcname=fp.relative_to(folder).as_posix())


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Package deliverables bundle (lightweight).")
    ap.add_argument("--tag", default="", help="Bundle tag, e.g. vietnamese_100ep (defaults to out folder name)")
    ap.add_argument("--out", required=True, help="Output folder, e.g. runs\\deliverables\\vietnamese_100ep")
    ap.add_argument("--repo-root", default=".", help="Repository root (default: .)")
    ap.add_argument("--runs-dir", default="runs", help="Runs folder (default: runs)")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    runs_dir = (repo_root / args.runs_dir).resolve()
    out_dir = (repo_root / args.out).resolve()

    tag = args.tag.strip() or out_dir.name

    package_dir = out_dir / "package"
    package_dir.mkdir(parents=True, exist_ok=True)

    copied_total = 0

    # Root docs
    root_docs = [
        repo_root / "README.md",
        repo_root / "SIMILARITY_RESULTS_SUMMARY.md",
        repo_root / "detail report.md",
        repo_root / "REPORT_PRESENT.md",
    ]
    for doc in root_docs:
        if safe_copy(doc, package_dir / doc.name):
            copied_total += 1

    # PDF/DOCX deliverables
    for name in [
        "COMPLETE_EVALUATION_REPORT.pdf",
        "COMPLETE_EVALUATION_REPORT.docx",
        "Similarity Results.pdf",
        "Similarity Results.docx",
        "PhD application Writing sample-ciwGAN.pdf",
    ]:
        src = repo_root / name
        if safe_copy(src, package_dir / src.name):
            copied_total += 1

    # runs artifacts (only small patterns)
    if runs_dir.exists():
        copied_total += copy_many(glob_existing(runs_dir, "plots/*.png"), package_dir / "runs" / "plots", base_dir=runs_dir)
        copied_total += copy_many(glob_existing(runs_dir, "compare/*.csv"), package_dir / "runs" / "compare", base_dir=runs_dir)
        copied_total += copy_many(glob_existing(runs_dir, "*.csv"), package_dir / "runs", base_dir=runs_dir)

    # Supplement + samples (if present)
    supplement = out_dir / "PHONOLOGICAL_INTERPRETATION_SUPPLEMENT.md"
    if safe_copy(supplement, package_dir / supplement.name):
        copied_total += 1

    samples_dir = out_dir / "samples"
    if samples_dir.exists() and samples_dir.is_dir():
        copied_total += copy_many(samples_dir.rglob("*"), package_dir / "samples", base_dir=samples_dir)

    # Create zip
    zip_path = out_dir / f"{tag}_deliverables.zip"
    zip_dir(package_dir, zip_path)

    print(f"Packaged into: {package_dir}")
    print(f"ZIP written: {zip_path}")
    print(f"Files copied: {copied_total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
