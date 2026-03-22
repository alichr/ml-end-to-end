"""Inference logic using ONNX Runtime for fast CPU prediction."""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD

CLASS_NAMES = {0: "cat", 1: "dog"}
MODEL_VERSION = "1.0.0"

# Image preprocessing constants (must match training)
IMAGE_SIZE = 256
CROP_SIZE = 224


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """Apply the same inference transform as training, using PIL + numpy.

    Steps: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)
    """
    # Convert to RGB if needed (handles grayscale, RGBA, palette, etc.)
    image = image.convert("RGB")

    # Resize so the shorter side is 256, maintaining aspect ratio
    width, height = image.size
    if width < height:
        new_width = IMAGE_SIZE
        new_height = int(height * IMAGE_SIZE / width)
    else:
        new_height = IMAGE_SIZE
        new_width = int(width * IMAGE_SIZE / height)
    image = image.resize((new_width, new_height), Image.BILINEAR)

    # Center crop to 224x224
    left = (new_width - CROP_SIZE) // 2
    top = (new_height - CROP_SIZE) // 2
    image = image.crop((left, top, left + CROP_SIZE, top + CROP_SIZE))

    # Convert to float32 numpy array, scale to [0, 1], then normalize
    arr = np.array(image, dtype=np.float32) / 255.0  # HWC, [0, 1]
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std

    # HWC -> CHW -> NCHW
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


class Predictor:
    """Loads an ONNX model and runs inference on images."""

    def __init__(self, model_path: str = "models/model.onnx") -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.model_version = MODEL_VERSION
        self.model_path = model_path

    def predict(self, image: Image.Image) -> dict:
        """Run inference on a single PIL image.

        Returns dict with predicted_class, confidence, probabilities, latency_ms.
        """
        start = time.perf_counter()

        input_tensor = _preprocess_image(image)
        logits = self.session.run(None, {"image": input_tensor})[0]
        probs = _softmax(logits)[0]

        latency_ms = (time.perf_counter() - start) * 1000

        pred_idx = int(probs.argmax())
        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 2),
        }

    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        """Run inference on a batch of PIL images."""
        start = time.perf_counter()

        # Preprocess all images and stack into a single batch
        inputs = np.concatenate([_preprocess_image(img) for img in images], axis=0)
        logits = self.session.run(None, {"image": inputs})[0]
        probs = _softmax(logits)

        total_ms = (time.perf_counter() - start) * 1000
        per_image_ms = total_ms / len(images)

        results = []
        for i in range(len(images)):
            pred_idx = int(probs[i].argmax())
            results.append(
                {
                    "predicted_class": CLASS_NAMES[pred_idx],
                    "confidence": float(probs[i][pred_idx]),
                    "probabilities": {
                        CLASS_NAMES[j]: float(probs[i][j]) for j in range(len(CLASS_NAMES))
                    },
                    "model_version": self.model_version,
                    "latency_ms": round(per_image_ms, 2),
                }
            )
        return results
