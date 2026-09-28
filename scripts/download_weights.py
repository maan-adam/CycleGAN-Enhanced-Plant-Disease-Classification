"""Download the trained EfficientNet-B7 weights (257 MB) into models/.

The file is attached to the repository's GitHub Release because it is too
large for a normal git repository. Override the source with WEIGHTS_URL.
"""
import os
import sys
import urllib.request
from pathlib import Path

URL = os.environ.get(
    "WEIGHTS_URL",
    "https://github.com/maan-adam/plant-disease-classification/releases/download/v1.0/best_model.pth",
)
DEST = Path(__file__).resolve().parent.parent / "models" / "best_model.pth"


def main() -> int:
    if DEST.is_file() and DEST.stat().st_size > 0:
        print(f"✓ {DEST} already present")
        return 0
    DEST.parent.mkdir(exist_ok=True)
    print(f"↓ {URL}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
