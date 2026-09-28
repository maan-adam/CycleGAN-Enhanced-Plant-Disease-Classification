"""Gradio demo: upload a leaf photo, get the top predicted diseases.

    python app.py            # http://127.0.0.1:7860

Also runs as a Hugging Face Space: put the weights at models/best_model.pth
(or set WEIGHTS_URL to download them on start-up).
"""
import os
import urllib.request
from pathlib import Path

import gradio as gr

from src.model import CLASS_NAMES, apply_clahe, load_model, predict, pretty

WEIGHTS = Path(os.environ.get("WEIGHTS_PATH", "models/best_model.pth"))
if not WEIGHTS.is_file() and os.environ.get("WEIGHTS_URL"):
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(os.environ["WEIGHTS_URL"], WEIGHTS)

MODEL = load_model(WEIGHTS)
EXAMPLES = sorted(str(p) for p in Path("examples").glob("*.jpg"))

NOTE = """
**Heads-up:** the model was trained on PlantVillage, where every photo is one leaf on a plain
background under lab lighting. On real field photos (PlantDoc) its top-1 accuracy drops to
about 31%, while it still separates healthy from diseased leaves about 80% of the time.
For best results, photograph **a single leaf against a plain background**. It only knows
the 21 classes listed below. This is a screening aid, not a diagnosis.
"""


def classify(image):
    if image is None:
        return {}, None
    probs = predict(MODEL, image, clahe=True)
    top = probs.argsort()[::-1][:5]
    return {pretty(CLASS_NAMES[i]): float(probs[i]) for i in top}, apply_clahe(image.convert("RGB"))


with gr.Blocks(title="Plant Disease Classifier") as demo:
    gr.Markdown("# Plant Disease Classifier\nEfficientNet-B7 · 21 PlantVillage classes · CLAHE preprocessing")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Leaf photo")
            btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            out = gr.Label(num_top_classes=5, label="Prediction")
            clahe_view = gr.Image(label="What the model sees (after CLAHE)", interactive=False)
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=inp, label="Field photos from PlantDoc (CC BY 4.0)")
    gr.Markdown(NOTE)
    gr.Markdown("**Classes:** " + " · ".join(pretty(c) for c in CLASS_NAMES))
    btn.click(classify, inputs=inp, outputs=[out, clahe_view])
    inp.upload(classify, inputs=inp, outputs=[out, clahe_view])

if __name__ == "__main__":
    demo.launch()
