"""Build a leakage-free train/val/test split for a CycleGAN-augmented dataset.

The original notebooks put CycleGAN images (files ending in ``_fake_B.png``)
into the class folders and then called ``random_split`` on everything, so the
validation and test sets contained generated images. This script fixes that:

1. Only **real** images are split into train / val / test (stratified by class,
   fixed seed).
2. Generated images go to **train only**.
3. A generated image whose *source* photo (same UUID prefix, e.g. the healthy
   leaf it was translated from) landed in val/test is dropped, so no leaf
   appears on both sides of the split.

Writes train.csv, val.csv and test.csv (columns: path, label, synthetic).

Usage:
    python scripts/make_splits.py --data path/to/raw/color --out splits/
"""
import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UUID_RE = re.compile(r"^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)


def is_synthetic(path: Path) -> bool:
    return "_fake" in path.stem


def source_id(path: Path) -> str | None:
    m = UUID_RE.match(path.name)
    return m.group(1).lower() if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="folder with one sub-folder per class")
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = sorted(d.name for d in args.data.iterdir() if d.is_dir())
    real, fake = defaultdict(list), defaultdict(list)
    for label, cls in enumerate(classes):
        for p in sorted((args.data / cls).iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                (fake if is_synthetic(p) else real)[label].append(p)

    rng = random.Random(args.seed)
    splits = {"train": [], "val": [], "test": []}
    held_out_sources = set()
    for label in range(len(classes)):
        items = real[label][:]
        rng.shuffle(items)
        n_test = round(len(items) * args.test)
        n_val = round(len(items) * args.val)
        for split, chunk in (("test", items[:n_test]), ("val", items[n_test:n_test + n_val]),
                             ("train", items[n_test + n_val:])):
            for p in chunk:
                splits[split].append((p, label, 0))
                if split != "train" and source_id(p):
                    held_out_sources.add(source_id(p))

    dropped = 0
    for label in range(len(classes)):
        for p in fake[label]:
            if source_id(p) in held_out_sources:
                dropped += 1
                continue
            splits["train"].append((p, label, 1))

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with open(args.out / f"{split}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "label", "synthetic"])
            w.writerows((str(p), label, syn) for p, label, syn in rows)

    n_fake = sum(len(v) for v in fake.values())
    print(f"{len(classes)} classes · real {sum(len(v) for v in real.values())} · synthetic {n_fake}")
    for split, rows in splits.items():
        print(f"  {split:5s} {len(rows):6d} images  ({sum(r[2] for r in rows)} synthetic)")
    print(f"  dropped {dropped} synthetic images whose source photo is in val/test")
    (args.out / "classes.txt").write_text("\n".join(classes) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
