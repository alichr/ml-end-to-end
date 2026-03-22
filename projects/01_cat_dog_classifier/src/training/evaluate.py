"""Comprehensive model evaluation on the held-out test set."""

import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.data.dataset import CatDogDataset
from src.data.transforms import inference_transform
from src.model.classifier import CatDogClassifier

matplotlib.use("Agg")

CLASS_NAMES = ["cat", "dog"]


def load_model(
    checkpoint_path: str, num_classes: int = 2, unfreeze_last_n: int = 3
) -> CatDogClassifier:
    """Load a trained model from checkpoint."""
    model = CatDogClassifier(
        num_classes=num_classes,
        freeze_backbone=True,
        unfreeze_last_n=unfreeze_last_n,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def get_predictions(
    model: CatDogClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run model on all data and return labels, predictions, and probabilities."""
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []

    model.to(device)
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.concatenate(all_probs, axis=0),
    )


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, save_path: Path) -> None:
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Test Set Confusion Matrix")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, save_path: Path) -> float:
    """Plot ROC curve and return AUC score."""
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return float(roc_auc)


def plot_per_class_metrics(labels: np.ndarray, preds: np.ndarray, save_path: Path) -> None:
    """Bar chart comparing precision, recall, F1 per class."""
    report = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True)

    metrics = ["precision", "recall", "f1-score"]
    cat_vals = [report["cat"][m] for m in metrics]
    dog_vals = [report["dog"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, cat_vals, width, label="Cat", color="#4C72B0")
    ax.bar(x + width / 2, dog_vals, width, label="Dog", color="#DD8452")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics on Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.legend()
    ax.set_ylim([0.9, 1.0])
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def benchmark_latency(
    model: CatDogClassifier, device: torch.device, n_runs: int = 100
) -> dict[str, float]:
    """Measure single-image inference latency."""
    model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    for _ in range(10):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def measure_model_size(checkpoint_path: str) -> float:
    """Return model file size in MB."""
    return Path(checkpoint_path).stat().st_size / (1024 * 1024)


def evaluate(
    checkpoint_path: str = "models/best_model.pth",
    test_dir: str = "data/splits/test",
    output_dir: str = "models/evaluation",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and data
    model = load_model(checkpoint_path)
    test_dataset = CatDogDataset(test_dir, transform=inference_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    print(f"Test samples: {len(test_dataset)}")

    # Get predictions
    labels, preds, probs = get_predictions(model, test_loader, device)

    # Metrics
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted")
    rec = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")
    roc = roc_auc_score(labels, probs[:, 1])

    print(f"\n{'='*40}")
    print("TEST SET RESULTS")
    print(f"{'='*40}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {roc:.4f}")

    # Per-class report
    print(f"\n{classification_report(labels, preds, target_names=CLASS_NAMES)}")

    # Plots
    plot_confusion_matrix(labels, preds, out / "confusion_matrix.png")
    plot_roc_curve(labels, probs, out / "roc_curve.png")
    plot_per_class_metrics(labels, preds, out / "per_class_metrics.png")
    print(f"Plots saved to {out}/")

    # Performance benchmarking
    print(f"\n{'='*40}")
    print("PERFORMANCE BENCHMARKS")
    print(f"{'='*40}")

    model_size = measure_model_size(checkpoint_path)
    print(f"Model size: {model_size:.2f} MB (target: < 50 MB)")

    cpu_latency = benchmark_latency(model, torch.device("cpu"))
    print(f"CPU latency (mean):   {cpu_latency['mean_ms']:.1f} ms (target: < 200 ms)")
    print(f"CPU latency (p95):    {cpu_latency['p95_ms']:.1f} ms")
    print(f"CPU latency (p99):    {cpu_latency['p99_ms']:.1f} ms")
    print(f"CPU throughput:       {1000 / cpu_latency['mean_ms']:.1f} images/sec")

    if torch.cuda.is_available():
        gpu_latency = benchmark_latency(model, torch.device("cuda"))
        print(f"GPU latency (mean):   {gpu_latency['mean_ms']:.1f} ms")
        print(f"GPU throughput:       {1000 / gpu_latency['mean_ms']:.1f} images/sec")
    else:
        gpu_latency = None

    # Save results as JSON
    results = {
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_auc_roc": roc,
        "model_size_mb": model_size,
        "cpu_latency": cpu_latency,
        "gpu_latency": gpu_latency,
        "test_samples": len(test_dataset),
    }
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out / 'results.json'}")


if __name__ == "__main__":
    evaluate()
