# Design Document: Cat vs Dog Image Classifier

## Problem Statement

Given a photograph, classify whether it contains a **cat** or a **dog**. This is a binary image classification service exposed via a REST API, designed for real-time inference in a production environment.

## Background & Motivation

This project uses the simplest possible ML task (binary classification) to build a **production-grade ML system** end-to-end. The engineering pipeline — data versioning, experiment tracking, serving, monitoring, and deployment — is the focus, not model complexity.

## Success Criteria

| Metric                | Target            |
|-----------------------|-------------------|
| Test set accuracy     | ≥ 95%             |
| Inference latency     | < 200ms (CPU)     |
| Model size            | < 50MB            |
| Concurrent requests   | 50                |

## Approach

- **Model:** MobileNetV2 pretrained on ImageNet, with a replaced classification head (`Linear(1280, 2)`). Fine-tuned using transfer learning.
- **Data:** Kaggle "Dogs vs Cats" dataset (~25,000 images). Split 70/15/15 (train/val/test), stratified.
- **Optimization:** Export to ONNX for faster inference; optional INT8 quantization.
- **Serving:** FastAPI with ONNX Runtime backend, behind a Streamlit frontend.
- **Monitoring:** Prometheus metrics + Grafana dashboards for latency, error rate, and prediction drift.

## Out of Scope

- Multi-class classification (breeds, other animals)
- Object detection or localization
- Video or real-time camera input
- Edge/mobile deployment
- User authentication or multi-tenancy

## Risks & Mitigations

| Risk                          | Mitigation                                      |
|-------------------------------|------------------------------------------------|
| Corrupted images in dataset   | Validation script to detect and remove bad files |
| Class imbalance               | Stratified splits; monitor per-class metrics     |
| Adversarial / ambiguous inputs| Return confidence scores; flag low-confidence predictions |
| Model drift in production     | Monitor prediction distribution and confidence drift |

## Architecture

```
User Image → Streamlit UI → FastAPI Server → ONNX Runtime (MobileNetV2)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              Structured    Prometheus       MLflow
               Logging       Metrics        Registry
                                │
                            Grafana
                           Dashboard
```

All services containerized with Docker and orchestrated via docker-compose. Deployed to Cloud Run or Railway.
