"""Export PyTorch model to ONNX format and optionally quantize."""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.model.classifier import CatDogClassifier


def export_to_onnx(
    checkpoint_path: str = "models/best_model.pth",
    output_path: str = "models/model.onnx",
    num_classes: int = 2,
    unfreeze_last_n: int = 3,
) -> str:
    """Export a trained PyTorch model to ONNX format."""
    model = CatDogClassifier(
        num_classes=num_classes,
        freeze_backbone=True,
        unfreeze_last_n=unfreeze_last_n,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={"image": {0: "batch_size"}, "prediction": {0: "batch_size"}},
        opset_version=17,
    )

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"ONNX model exported to {output_path} ({size_mb:.2f} MB)")
    return output_path


def benchmark_onnx(onnx_path: str = "models/model.onnx", n_runs: int = 100) -> dict[str, float]:
    """Benchmark ONNX Runtime inference latency on CPU."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {"image": dummy})

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, {"image": dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }
    print(f"ONNX CPU latency (mean): {results['mean_ms']:.1f} ms")
    print(f"ONNX CPU latency (p95):  {results['p95_ms']:.1f} ms")
    print(f"ONNX CPU throughput:     {1000 / results['mean_ms']:.1f} images/sec")
    return results


def verify_onnx_output(
    checkpoint_path: str = "models/best_model.pth",
    onnx_path: str = "models/model.onnx",
    unfreeze_last_n: int = 3,
) -> None:
    """Verify ONNX model produces the same output as PyTorch."""
    # PyTorch inference
    model = CatDogClassifier(num_classes=2, freeze_backbone=True, unfreeze_last_n=unfreeze_last_n)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        pytorch_output = model(dummy).numpy()

    # ONNX inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"image": dummy.numpy()})[0]

    max_diff = np.abs(pytorch_output - onnx_output).max()
    print(f"Max output difference (PyTorch vs ONNX): {max_diff:.8f}")
    assert max_diff < 1e-5, f"Output mismatch: {max_diff}"
    print("ONNX output matches PyTorch output.")


if __name__ == "__main__":
    print("=== Exporting to ONNX ===")
    onnx_path = export_to_onnx()

    print("\n=== Verifying ONNX output ===")
    verify_onnx_output()

    print("\n=== Benchmarking ONNX ===")
    benchmark_onnx(onnx_path)
