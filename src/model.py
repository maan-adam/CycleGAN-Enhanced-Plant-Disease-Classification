"""EfficientNet-B7 plant disease classifier: model loading, preprocessing and prediction.

Preprocessing matches the training notebook exactly: CLAHE on the lightness
channel (clip limit 3.0, 8x8 tiles), resize to 224x224, normalise to [-1, 1].
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Tomato___Bacterial_spot",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]

TO_TENSOR = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def pretty(name: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato · Late blight'."""
    plant, _, condition = name.partition("___")
    return f"{plant.replace(',_', ' ').replace('_', ' ')} · {condition.replace('_', ' ')}"


def apply_clahe(image: Image.Image, clip_limit: float = 3.0, tile_grid_size: int = 8) -> Image.Image:
    """Contrast Limited Adaptive Histogram Equalization on the L channel of LAB."""
    lab = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return Image.fromarray(cv2.cvtColor(cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2RGB))


def load_model(weights: str | Path, device: str = "cpu") -> torch.nn.Module:
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_name("efficientnet-b7", num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def predict(model: torch.nn.Module, image: Image.Image, clahe: bool = True, device: str = "cpu") -> np.ndarray:
    """Return class probabilities (21,) for one PIL image."""
    image = image.convert("RGB")
    if clahe:
        image = apply_clahe(image)
    logits = model(TO_TENSOR(image).unsqueeze(0).to(device))
    return torch.softmax(logits, dim=1)[0].cpu().numpy()
