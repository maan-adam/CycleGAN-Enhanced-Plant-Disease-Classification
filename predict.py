"""Classify plant leaf photos from the command line.

Examples:
    python predict.py examples/tomato_late_blight.jpg
    python predict.py examples/ --top 3
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

from src.model import CLASS_NAMES, load_model, predict, pretty

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="image file or folder")
    ap.add_argument("--weights", type=Path, default=Path("models/best_model.pth"))
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--no-clahe", action="store_true", help="skip CLAHE (the model was trained with it)")
    args = ap.parse_args()

    model = load_model(args.weights)
    paths = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_EXTS) \
        if args.source.is_dir() else [args.source]
    for path in paths:
        probs = predict(model, Image.open(path), clahe=not args.no_clahe)
        ranked = probs.argsort()[::-1][: args.top]
        print(f"{path.name}")
        for i in ranked:
            print(f"   {probs[i]:6.1%}  {pretty(CLASS_NAMES[i])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
