# Model Card: Cat vs Dog Classifier

## Model Details

- **Model name:** cat-dog-classifier
- **Version:** v1 (production)
- **Architecture:** MobileNetV2 (pretrained on ImageNet) with fine-tuned last 3 backbone layers
- **Framework:** PyTorch
- **Model size:** 8.73 MB (PyTorch), 0.30 MB (ONNX)
- **Parameters:** 2.2M total, 1.2M trainable (last 3 backbone layers + classifier head)
- **Training run:** finetune-last3 (MLflow)

## Intended Use

- **Primary use:** Classify a photo as containing a cat or a dog
- **Intended users:** Demonstration / portfolio project
- **Out of scope:** Multi-class classification, object detection, video, edge deployment, medical/safety-critical applications

## Training Data

- **Dataset:** Microsoft Cats vs Dogs (Kaggle)
- **Total images:** 24,998 (after removing corrupted files)
- **Split:** Train 70% (17,498) / Val 15% (3,748) / Test 15% (3,752)
- **Class balance:** Perfectly balanced (50/50 cat/dog)
- **Preprocessing:** Validated for corruption, converted to RGB JPEG
- **Augmentation:** Resize(256), RandomCrop(224), RandomHorizontalFlip, ColorJitter

## Evaluation Results

Evaluated on the **held-out test set** (3,752 images, never seen during training):

| Metric | Value |
|--------|-------|
| Accuracy | 98.72% |
| Precision | 98.72% |
| Recall | 98.72% |
| F1 Score | 98.72% |
| AUC-ROC | 99.92% |

### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|-------|-----------|--------|----------|---------|
| Cat | 0.99 | 0.99 | 0.99 | 1,876 |
| Dog | 0.99 | 0.99 | 0.99 | 1,876 |

### Performance vs Design Targets

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Test accuracy | >= 95% | 98.72% | PASS |
| CPU inference latency | < 200 ms | 6.6 ms | PASS |
| Model size | < 50 MB | 8.73 MB | PASS |

## Performance Benchmarks

### Latency (single image inference)

| Platform | Mean | P95 | P99 | Throughput |
|----------|------|-----|-----|------------|
| CPU (PyTorch) | 6.6 ms | 6.8 ms | 6.9 ms | 152 img/s |
| GPU (PyTorch) | 1.0 ms | 1.0 ms | 1.1 ms | 973 img/s |
| CPU (ONNX Runtime) | 1.4 ms | 1.4 ms | — | 714 img/s |

### Model Size

| Format | Size |
|--------|------|
| PyTorch (.pth) | 8.73 MB |
| ONNX (.onnx) | 0.30 MB |

## Known Limitations

- Trained only on cats and dogs — any other animal or object will still be classified as one of the two
- Dataset consists mostly of clear, well-lit pet photos — may struggle with:
  - Very dark or overexposed images
  - Extreme close-ups or unusual angles
  - Multiple animals in one image
  - Young puppies/kittens that look similar
  - Cartoon or illustrated animals
- No confidence threshold is applied — the model always outputs cat or dog even when uncertain
- Not tested on adversarial inputs

## Ethical Considerations

- This is a demonstration project, not intended for production decision-making
- The model should not be used to make decisions that affect animal welfare
- No personally identifiable information is used in training

## How to Use

```python
from src.model.classifier import CatDogClassifier
from src.data.transforms import inference_transform
from PIL import Image
import torch

model = CatDogClassifier(num_classes=2, freeze_backbone=True, unfreeze_last_n=3)
model.load_state_dict(torch.load("models/best_model.pth", weights_only=True))
model.eval()

image = Image.open("photo.jpg").convert("RGB")
tensor = inference_transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    probs = torch.softmax(output, dim=1)
    pred = "dog" if probs[0][1] > 0.5 else "cat"
    confidence = probs.max().item()

print(f"{pred} ({confidence:.1%})")
```
