"""Evaluate the PlantVillage-trained model on real field photos from PlantDoc.

PlantVillage images are single leaves on a plain background under lab
lighting. PlantDoc (Singh et al., 2020) contains photos taken in the field:
cluttered backgrounds, several leaves, uneven light. This script measures how
much of the PlantVillage accuracy survives that shift.

Usage:
    git clone --depth 1 --filter=blob:none --sparse https://github.com/pratikkayal/PlantDoc-Dataset
    (cd PlantDoc-Dataset && git sparse-checkout set test)
    python scripts/evaluate_plantdoc.py --plantdoc PlantDoc-Dataset/test --weights models/best_model.pth
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import CLASS_NAMES, load_model, predict  # noqa: E402

# PlantDoc folder -> PlantVillage class. PlantDoc classes with no PlantVillage
# counterpart in our 21 classes (corn, potato, cherry, ...) are skipped.
PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple leaf": "Apple___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
    "Peach leaf": "Peach___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_healthy(name: str) -> bool:
    return name.endswith("healthy")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plantdoc", type=Path, required=True, help="PlantDoc test/ folder")
    ap.add_argument("--weights", type=Path, default=Path("models/best_model.pth"))
    ap.add_argument("--out", type=Path, default=Path("results/plantdoc_eval.json"))
    args = ap.parse_args()

    model = load_model(args.weights)
    samples = []
    for folder, pv_class in PLANTDOC_TO_PLANTVILLAGE.items():
        for p in sorted((args.plantdoc / folder).iterdir()):
            if p.suffix.lower() in IMAGE_EXTS:
                samples.append((p, CLASS_NAMES.index(pv_class)))
    print(f"{len(samples)} PlantDoc test images in {len(PLANTDOC_TO_PLANTVILLAGE)} matching classes")

    report = {"n_images": len(samples), "n_classes": len(PLANTDOC_TO_PLANTVILLAGE), "runs": {}}
    for clahe in (True, False):
        y_true, y_pred, top3 = [], [], []
        for path, label in samples:
            try:
                probs = predict(model, Image.open(path), clahe=clahe)
            except Exception as exc:  # a few PlantDoc files are corrupt
                print(f"skip {path.name}: {exc}")
                continue
            y_true.append(label)
            y_pred.append(int(probs.argmax()))
            top3.append(label in np.argsort(probs)[-3:])
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        healthy_true = np.array([is_healthy(CLASS_NAMES[i]) for i in y_true])
        healthy_pred = np.array([is_healthy(CLASS_NAMES[i]) for i in y_pred])
        per_class = {}
        for folder, pv_class in PLANTDOC_TO_PLANTVILLAGE.items():
            idx = CLASS_NAMES.index(pv_class)
            mask = y_true == idx
            if mask.any():
                per_class[pv_class] = {"n": int(mask.sum()), "accuracy": float((y_pred[mask] == idx).mean())}
        run = {
            "evaluated": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "top3_accuracy": float(np.mean(top3)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true)))),
            "healthy_vs_diseased_accuracy": float((healthy_true == healthy_pred).mean()),
            "per_class": per_class,
        }
        name = "with_clahe" if clahe else "without_clahe"
        report["runs"][name] = run
        print(f"\n[{name}] n={run['evaluated']}  accuracy={run['accuracy']:.3f}  top-3={run['top3_accuracy']:.3f}  "
              f"macro-F1={run['macro_f1']:.3f}  healthy-vs-diseased={run['healthy_vs_diseased_accuracy']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
