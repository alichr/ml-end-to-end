# End-to-End ML Project Plan: Cat vs Dog Image Classifier

## Goal

Build a **production-grade** image classifier using the **simplest possible ML task**
(binary classification: cat vs dog) so the focus stays on **engineering skills**, not model
complexity. By the end, you will have touched every stage of the ML lifecycle — from
design doc to live monitoring dashboard.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why cat vs dog? | Binary classification is the simplest ML task. Abundant free data. Transfer learning makes training trivial. |
| Why not something more complex? | The ML is not the point — the **pipeline** is. A simple model lets you focus 100% on engineering. |
| Will this look good on a portfolio? | Yes — interviewers care far more about "can you ship and monitor a model" than "can you beat SOTA on ImageNet." |
| What if I already know some of these tools? | Skip what you know, dive deep where you don't. The plan is modular. |

---

## Architecture Overview

```
                    ┌──────────────┐
                    │   Streamlit  │
                    │   Frontend   │
                    └──────┬───────┘
                           │ HTTP
                           ▼
                    ┌──────────────┐      ┌──────────────┐
  User Image ──────▶   FastAPI    │──────▶│  ML Model    │
                    │   Server    │       │  (PyTorch)   │
                    └──────┬───────┘      └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Logging │ │Prometheus│ │  MLflow  │
        │(Structured)│ │ Metrics │ │ Registry │
        └──────────┘ └────┬─────┘ └──────────┘
                          ▼
                    ┌──────────┐
                    │ Grafana  │
                    │Dashboard │
                    └──────────┘

Everything runs in Docker. Deployed to Cloud Run / Railway.
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | ML ecosystem standard |
| ML Framework | PyTorch + torchvision | Industry standard, best for learning |
| Experiment Tracking | MLflow | Tracks params, metrics, artifacts, model registry |
| Data Versioning | DVC | Git for data — essential for reproducibility |
| API Framework | FastAPI | Fast, modern, auto-generates OpenAPI docs |
| Frontend | Streamlit | Simplest way to build ML UIs |
| Containerization | Docker + docker-compose | Deployment standard everywhere |
| CI/CD | GitHub Actions | Free, integrated with GitHub |
| Monitoring | Prometheus + Grafana | Industry standard observability |
| Testing | pytest | Python standard |
| Linting | Ruff + mypy | Fast linting + type checking |
| Config Management | YAML + Pydantic | Clean, validated configurations |
| Model Optimization | ONNX Runtime | Faster inference in production |

---

## Project Structure

```
ml-end-to-end/
│
├── PROJECT_PLAN.md              # This file
├── DESIGN_DOC.md                # Problem statement, constraints, success criteria
├── MODEL_CARD.md                # Model documentation (what, how, limitations)
├── README.md                    # Setup instructions, architecture, quickstart
│
├── pyproject.toml               # Dependencies and project metadata
├── dvc.yaml                     # Data pipeline definition
├── dvc.lock                     # Data pipeline lock file
├── MLproject                    # MLflow project file
│
├── configs/
│   ├── train_config.yaml        # Training hyperparameters
│   ├── serve_config.yaml        # Serving configuration
│   └── data_config.yaml         # Data paths, splits, augmentation
│
├── data/                        # Git-ignored, DVC-tracked
│   ├── raw/                     # Original downloaded images
│   ├── processed/               # Resized, cleaned images
│   └── splits/                  # train/ val/ test/ folders
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_training.ipynb        # Interactive training experiments
│   └── 03_evaluation.ipynb      # Model evaluation & error analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py           # PyTorch Dataset class
│   │   ├── transforms.py        # Image preprocessing & augmentation
│   │   ├── download.py          # Download dataset script
│   │   └── validate.py          # Data integrity checks
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── classifier.py        # Model architecture (MobileNetV2 + head)
│   │   └── export.py            # Export to TorchScript / ONNX
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             # Training loop
│   │   ├── evaluate.py          # Evaluation metrics
│   │   └── callbacks.py         # Early stopping, checkpointing
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI application
│   │   ├── schemas.py           # Request/response Pydantic models
│   │   ├── middleware.py        # Rate limiting, auth, CORS
│   │   └── predict.py           # Inference logic
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Prometheus metric definitions
│   │   └── drift.py             # Prediction distribution monitoring
│   │
│   └── frontend/
│       └── app.py               # Streamlit UI
│
├── tests/
│   ├── unit/
│   │   ├── test_transforms.py   # Data transform correctness
│   │   ├── test_model.py        # Model output shape, export
│   │   └── test_schemas.py      # API schema validation
│   ├── integration/
│   │   ├── test_api.py          # Full API predict pipeline
│   │   └── test_training.py     # Training runs without error
│   └── conftest.py              # Shared fixtures (sample images, model)
│
├── docker/
│   ├── Dockerfile.api           # Multi-stage build for API
│   ├── Dockerfile.frontend      # Streamlit container
│   └── Dockerfile.training      # Training environment
│
├── docker-compose.yaml          # Orchestrate all services
│
├── .github/
│   └── workflows/
│       ├── ci.yaml              # Lint → Test → Build on PR
│       └── cd.yaml              # Deploy on merge to main
│
├── grafana/
│   └── dashboards/
│       └── ml_monitoring.json   # Pre-configured dashboard
│
├── prometheus/
│   └── prometheus.yml           # Scrape config
│
└── scripts/
    ├── setup.sh                 # One-command project setup
    ├── train.sh                 # Run training with default config
    └── deploy.sh                # Deploy to cloud
```

---

## Phase 1: Project Setup & Design Doc

**Duration:** 1–2 days
**Objective:** Define what you're building before writing any code.

### Tasks

1. ~~**Write `DESIGN_DOC.md`**~~ ✅
   - **Problem statement:** "Given a photo, classify whether it contains a cat or a dog"
   - **Success criteria:**
     - Accuracy ≥ 95% on test set
     - Inference latency < 200ms on CPU
     - Model size < 50MB
     - API handles 50 concurrent requests
   - **Out of scope:** multi-class, object detection, video, edge deployment
   - **Risks:** corrupted images in dataset, class imbalance, adversarial inputs

2. ~~**Initialize the repository**~~ ✅
   - `git init`, create `.gitignore` (data/, models/, *.pyc, .env, etc.)
   - Set up branch strategy: `main` (production), `dev` (integration), `feature/*`
   - Create `pyproject.toml` with all dependencies

3. ~~**Create the folder structure**~~ ✅ (as shown above)

4. ~~**Set up development environment**~~ ✅
   - Python virtual environment (`uv` or `venv`)
   - Pre-commit hooks: ruff, mypy, trailing whitespace
   - Editor config for consistent formatting

### Skills Learned

- Writing ML design documents (a real job requirement)
- Professional Python project setup
- Git best practices for ML projects

---

## Phase 2: Data Pipeline

**Duration:** 3–4 days
**Objective:** Get clean, versioned, reproducible data ready for training.

### Tasks

1. ~~**Download the dataset**~~ ✅
   - Kaggle "Dogs vs Cats" dataset (~25,000 images)
   - Write `src/data/download.py` — automated, idempotent download script

2. ~~**Set up DVC (Data Version Control)**~~ ✅
   - `dvc init`, configure remote storage (Google Drive or S3 free tier)
   - Track `data/raw/` with DVC
   - Learn: `dvc add`, `dvc push`, `dvc pull`
   - **Why DVC matters:** your teammate can reproduce your exact dataset with `dvc pull`

3. ~~**Exploratory Data Analysis (EDA)**~~ ✅ — `notebooks/01_eda.ipynb`
   - Total image count per class (check for imbalance)
   - Image size distribution (min, max, mean width/height)
   - Sample grid: display 16 random cats, 16 random dogs
   - Check for corrupted / unreadable images
   - File format distribution (JPEG, PNG, etc.)

4. ~~**Data preprocessing**~~ ✅ — `src/data/`
   - `validate.py`: scan for and remove corrupted images
   - `transforms.py`: define training and inference transforms
     ```python
     # Training: augmentation + normalization
     train_transform = Compose([
         Resize(256),
         RandomCrop(224),
         RandomHorizontalFlip(),
         ColorJitter(brightness=0.2, contrast=0.2),
         ToTensor(),
         Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
     ])

     # Inference: deterministic
     inference_transform = Compose([
         Resize(256),
         CenterCrop(224),
         ToTensor(),
         Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
     ])
     ```
   - Split into train (70%), validation (15%), test (15%) — **stratified**
   - `dataset.py`: PyTorch `Dataset` class with lazy loading

5. ~~**Define DVC pipeline**~~ ✅ — `dvc.yaml`
   - Stage 1: download → Stage 2: validate → Stage 3: split
   - Reproducible with `dvc repro`

### Skills Learned

- Data versioning (DVC) — reproducibility in ML
- EDA for computer vision
- Data augmentation strategies
- PyTorch Dataset / DataLoader patterns
- Building reproducible data pipelines

---

## Phase 3: Model Development & Experiment Tracking

**Duration:** 4–5 days
**Objective:** Train a model, track experiments properly, pick the best one.

### Tasks

1. ~~**Define the model**~~ ✅ — `src/model/classifier.py`
   - Use **MobileNetV2** (pretrained on ImageNet)
   - Replace the final classification head with: `Linear(1280, 2)`
   - Why MobileNetV2: small (3.4M params), fast, good accuracy
   ```python
   class CatDogClassifier(nn.Module):
       def __init__(self, num_classes=2, freeze_backbone=True):
           super().__init__()
           self.backbone = models.mobilenet_v2(pretrained=True)
           if freeze_backbone:
               for param in self.backbone.parameters():
                   param.requires_grad = False
           self.backbone.classifier[1] = nn.Linear(1280, num_classes)
   ```

2. ~~**Write training config**~~ ✅ — `configs/train_config.yaml`
   ```yaml
   model:
     name: mobilenet_v2
     pretrained: true
     freeze_backbone: true
     num_classes: 2

   training:
     epochs: 10
     batch_size: 32
     learning_rate: 0.001
     optimizer: adam
     scheduler: cosine
     early_stopping_patience: 3

   data:
     image_size: 224
     train_split: 0.7
     val_split: 0.15
     num_workers: 4
   ```

3. ~~**Write the training loop**~~ ✅ — `src/training/train.py`
   - Load config from YAML
   - Initialize model, optimizer, scheduler, loss function
   - Training loop with validation after each epoch
   - Save best checkpoint based on validation accuracy
   - Log everything to MLflow

4. ~~**Set up MLflow experiment tracking**~~ ✅
   - Log per-experiment: hyperparameters, train/val loss curves, final metrics
   - Log artifacts: best model checkpoint, confusion matrix plot
   - **Register the best model** in MLflow Model Registry
   - Learn: `mlflow.log_param()`, `mlflow.log_metric()`, `mlflow.pytorch.log_model()`

5. ~~**Run experiments**~~ ✅ (track all in MLflow)

   | Experiment | What Changes | Expected Result |
   |-----------|-------------|----------------|
   | Baseline | Frozen backbone, lr=0.001, 10 epochs | ~93-95% accuracy |
   | Fine-tune | Unfreeze last 3 layers, lr=0.0001 | ~96-97% accuracy |
   | Augmentation | Add stronger augmentation | Better generalization |
   | Batch size | Try 16, 32, 64 | Find sweet spot |
   | Scheduler | Cosine vs StepLR | Marginal improvement |

6. ~~**Pick the best model**~~ ✅ using MLflow UI
   - Compare runs side by side
   - Select best model → "Promote to Production" in registry

### Skills Learned

- Transfer learning (the most practical ML technique)
- Writing clean, configurable training loops
- Experiment tracking with MLflow (essential for any team)
- Model registry (knowing which model is in production)
- Hyperparameter tuning methodology

---

## Phase 4: Evaluation & Model Optimization

**Duration:** 2–3 days
**Objective:** Thoroughly evaluate the model and optimize it for production.

### Tasks

1. **Comprehensive evaluation** — `src/training/evaluate.py`
   - Metrics on the **held-out test set** (never seen during training):
     - Accuracy, Precision, Recall, F1 Score
     - AUC-ROC curve
     - Confusion matrix
   - **Per-class analysis:** is the model better at cats or dogs?
   - Generate all plots, save as artifacts in MLflow

2. **Error analysis** — `notebooks/03_evaluation.ipynb`
   - Display the top 20 most confident **wrong** predictions
   - Look for patterns: are failures blurry? occluded? weird angles?
   - This is where you learn what your model actually struggles with
   - Document findings in `MODEL_CARD.md`

3. **Performance benchmarking**
   - Measure inference latency: single image on CPU (must be < 200ms)
   - Measure throughput: images per second
   - Measure model file size (must be < 50MB)
   - Memory footprint during inference

4. **Model optimization for production** — `src/model/export.py`
   - Export to **ONNX** format for faster inference
   - Benchmark: PyTorch vs ONNX Runtime latency
   - Optionally: quantize to INT8 (even smaller and faster)
   ```python
   # Export to ONNX
   torch.onnx.export(model, dummy_input, "model.onnx",
                      input_names=["image"],
                      output_names=["prediction"])
   ```

5. **Write `MODEL_CARD.md`**
   - What the model does and doesn't do
   - Training data description
   - Evaluation metrics
   - Known limitations (e.g., struggles with puppies that look like kittens)
   - Intended use and misuse scenarios

### Skills Learned

- Proper ML evaluation (beyond just accuracy)
- Error analysis — the most underrated ML skill
- Model optimization (ONNX, quantization)
- Model documentation (model cards)

---

## Phase 5: API & Serving Layer

**Duration:** 3–4 days
**Objective:** Wrap the model in a production-quality REST API.

### Tasks

1. **Define API schemas** — `src/serving/schemas.py`
   ```python
   class PredictionResponse(BaseModel):
       predicted_class: str          # "cat" or "dog"
       confidence: float             # 0.0 to 1.0
       probabilities: dict[str, float]  # {"cat": 0.03, "dog": 0.97}
       model_version: str            # Which model made this prediction
       latency_ms: float             # How long inference took

   class HealthResponse(BaseModel):
       status: str                   # "healthy" or "unhealthy"
       model_loaded: bool
       model_version: str
       uptime_seconds: float
   ```

2. **Build FastAPI application** — `src/serving/app.py`
   - `POST /predict` — upload image, get prediction
   - `POST /predict/batch` — upload multiple images (batch inference)
   - `GET /health` — health check
   - `GET /metrics` — Prometheus metrics endpoint
   - Load model once at startup (not per request)

3. **Inference logic** — `src/serving/predict.py`
   - Load ONNX model with ONNX Runtime
   - Apply the same preprocessing as training (inference transforms)
   - Return probabilities via softmax
   - Handle edge cases: grayscale images, RGBA, very small images

4. **API middleware** — `src/serving/middleware.py`
   - **Input validation:** file type (JPEG/PNG only), file size (< 5MB)
   - **Rate limiting:** max 100 requests/minute per IP
   - **CORS:** allow frontend to call the API
   - **Request ID:** unique ID per request for tracing
   - **Error handling:** return clean JSON errors, never expose stack traces

5. **Build Streamlit frontend** — `src/frontend/app.py`
   - File upload widget
   - Display uploaded image
   - Show prediction, confidence bar, and latency
   - Show model version and health status

6. **API tests** — `tests/`
   - Unit tests: schema validation, transforms
   - Integration tests: full request → response with sample image
   - Edge case tests: empty file, wrong format, huge image, concurrent requests

### Skills Learned

- Building production ML APIs (FastAPI)
- Request/response schema design
- Input validation and security basics
- Batch inference patterns
- API testing strategies for ML

---

## Phase 6: Containerization

**Duration:** 2 days
**Objective:** Package everything into Docker containers.

### Tasks

1. **API Dockerfile** — `docker/Dockerfile.api`
   - Multi-stage build:
     - Stage 1 (builder): install dependencies
     - Stage 2 (runtime): copy only what's needed
   - Use slim base image (`python:3.11-slim`)
   - Target: image size < 1GB
   ```dockerfile
   # Builder stage
   FROM python:3.11-slim AS builder
   WORKDIR /app
   COPY pyproject.toml .
   RUN pip install --no-cache-dir .

   # Runtime stage
   FROM python:3.11-slim
   COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
   COPY src/ /app/src/
   COPY models/ /app/models/
   EXPOSE 8000
   CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Frontend Dockerfile** — `docker/Dockerfile.frontend`
   - Simpler single-stage build
   - Expose Streamlit port 8501

3. **Training Dockerfile** — `docker/Dockerfile.training`
   - Includes GPU support (optional)
   - Mounts data volume

4. **docker-compose.yaml** — orchestrate everything
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports: ["8000:8000"]
       environment:
         - MODEL_PATH=/app/models/model.onnx
         - LOG_LEVEL=info

     frontend:
       build:
         context: .
         dockerfile: docker/Dockerfile.frontend
       ports: ["8501:8501"]
       depends_on: [api]

     prometheus:
       image: prom/prometheus
       volumes: ["./prometheus:/etc/prometheus"]
       ports: ["9090:9090"]

     grafana:
       image: grafana/grafana
       volumes: ["./grafana:/etc/grafana/provisioning"]
       ports: ["3000:3000"]
       depends_on: [prometheus]
   ```

5. **Verify the full stack locally**
   - `docker compose up` — everything starts and connects
   - Frontend can call API, Prometheus scrapes metrics, Grafana shows dashboard

### Skills Learned

- Docker multi-stage builds for ML (keeping images small)
- docker-compose for multi-service orchestration
- Environment variable configuration
- Container networking basics

---

## Phase 7: Testing & CI/CD

**Duration:** 2–3 days
**Objective:** Automate quality checks so nothing broken reaches production.

### Tasks

1. **Write comprehensive tests**

   **Unit tests** (fast, no dependencies):
   ```
   test_transforms.py
   ├── test_training_transform_output_shape
   ├── test_inference_transform_is_deterministic
   └── test_normalize_values_in_range

   test_model.py
   ├── test_model_output_shape_is_batch_x_2
   ├── test_model_output_sums_to_1_after_softmax
   ├── test_model_loads_from_checkpoint
   └── test_onnx_export_matches_pytorch_output

   test_schemas.py
   ├── test_prediction_response_validates_confidence_range
   └── test_health_response_fields
   ```

   **Integration tests** (slower, test the full pipeline):
   ```
   test_api.py
   ├── test_predict_returns_200_on_valid_image
   ├── test_predict_returns_422_on_invalid_file
   ├── test_predict_returns_correct_schema
   ├── test_batch_predict_multiple_images
   ├── test_health_endpoint
   └── test_concurrent_requests

   test_training.py
   ├── test_training_loop_runs_one_epoch
   └── test_training_saves_checkpoint
   ```

   **ML-specific tests** (often overlooked):
   ```
   ├── test_model_is_deterministic_with_same_seed
   ├── test_preprocessing_matches_between_training_and_serving
   └── test_model_size_under_50mb
   ```

2. **Set up CI pipeline** — `.github/workflows/ci.yaml`
   ```yaml
   # Runs on every Pull Request
   name: CI
   on: [pull_request]
   jobs:
     lint:
       - ruff check .
       - mypy src/
     test:
       - pytest tests/unit/
       - pytest tests/integration/
     build:
       - docker build -f docker/Dockerfile.api .
     smoke-test:
       - Start container
       - curl /health
       - curl /predict with sample image
       - Verify response schema
   ```

3. **Set up CD pipeline** — `.github/workflows/cd.yaml`
   ```yaml
   # Runs on merge to main
   name: CD
   on:
     push:
       branches: [main]
   jobs:
     deploy:
       - Build Docker image
       - Push to container registry
       - Deploy to Cloud Run / Railway
       - Run smoke test on deployed URL
   ```

4. **Pre-commit hooks**
   - Ruff (linting + formatting)
   - mypy (type checking)
   - Trailing whitespace, YAML validation
   - Check no large files committed accidentally

### Skills Learned

- Testing ML systems (unit, integration, ML-specific)
- CI/CD pipeline design for ML projects
- GitHub Actions workflow authoring
- Pre-commit hooks for code quality

---

## Phase 8: Deployment

**Duration:** 2–3 days
**Objective:** Get the application running on the internet.

### Tasks

1. **Choose a platform** (pick one, learn the concepts)

   | Platform | Pros | Cons | Cost |
   |----------|------|------|------|
   | **Railway** | Simplest deployment | Less control | Free tier available |
   | **GCP Cloud Run** | Auto-scaling, pay-per-use | More setup | Free tier generous |
   | **AWS ECS/Fargate** | Enterprise standard | Most complex | Free tier limited |

   **Recommendation:** Start with **Railway** (simplest), then migrate to **Cloud Run** as a learning exercise.

2. **Prepare for deployment**
   - Ensure all config comes from environment variables
   - Health check endpoint works
   - Graceful shutdown handling
   - Model file is baked into the Docker image (simplest approach)

3. **Deploy the API**
   - Push Docker image to container registry
   - Deploy container with environment variables
   - Verify `/health` endpoint is reachable
   - Test `/predict` with a real image from the internet

4. **Deploy the frontend**
   - Option A: Streamlit Cloud (free, easiest)
   - Option B: Same platform as API
   - Point frontend at the deployed API URL

5. **Load testing**
   - Use `locust` or `hey` to simulate traffic
   - Target: 50 concurrent users, p95 latency < 200ms
   - Identify bottlenecks: is it CPU? memory? network?

6. **Set up basic infrastructure**
   - HTTPS (usually automatic on these platforms)
   - Custom domain (optional)
   - Environment-specific configs (staging vs production)

### Skills Learned

- Cloud deployment (containers as a service)
- Environment configuration for different stages
- Load testing and performance tuning
- Infrastructure basics (HTTPS, domains, scaling)

---

## Phase 9: Monitoring & Observability

**Duration:** 2–3 days
**Objective:** Know what your system is doing in production at all times.

### Tasks

1. **Structured logging** — `src/serving/`
   - Every prediction logged as structured JSON:
     ```json
     {
       "timestamp": "2026-03-18T10:30:00Z",
       "request_id": "abc-123",
       "predicted_class": "cat",
       "confidence": 0.97,
       "latency_ms": 45,
       "model_version": "v1.2",
       "image_size_bytes": 102400,
       "status": "success"
     }
     ```
   - Log errors with full context (but never log the image itself — privacy)

2. **Prometheus metrics** — `src/monitoring/metrics.py`
   - `prediction_requests_total` — counter by class and status
   - `prediction_latency_seconds` — histogram
   - `prediction_confidence` — histogram (track confidence distribution)
   - `model_info` — gauge with version label
   - `active_requests` — gauge (concurrent request count)

3. **Grafana dashboard** — `grafana/dashboards/ml_monitoring.json`
   - Row 1: Request rate, error rate, latency percentiles (p50, p95, p99)
   - Row 2: Prediction class distribution over time, confidence distribution
   - Row 3: System metrics (CPU, memory, active requests)

4. **ML-specific monitoring** — `src/monitoring/drift.py`
   - **Prediction distribution drift:** if your model suddenly predicts 90% cat
     (normally it's 50/50), something is wrong
   - **Confidence drift:** if average confidence drops, the model might be seeing
     out-of-distribution data
   - Simple approach: log distributions, alert on significant changes

5. **Alerting rules**
   - Error rate > 5% for 5 minutes → alert
   - p95 latency > 500ms for 5 minutes → alert
   - Prediction distribution shift > 2 standard deviations → alert
   - Health check failing → alert immediately

### What to Monitor (Reference)

```
Operational Metrics          ML-Specific Metrics
─────────────────           ────────────────────
• Request rate              • Prediction class distribution
• Error rate                • Confidence score distribution
• Latency (p50/p95/p99)    • Data drift detection
• CPU / Memory usage        • Model staleness (time since retrain)
• Active connections        • Feature distribution changes
```

### Skills Learned

- Structured logging for ML systems
- Prometheus metrics instrumentation
- Grafana dashboard design
- ML-specific monitoring (data/prediction drift)
- Alerting strategies

---

## Phase 10: Retraining Pipeline (Bonus)

**Duration:** 2 days
**Objective:** Learn how to update the model when new data arrives.

### Tasks

1. **Simulate new data arrival**
   - Set aside 10% of original data as "new data"
   - Write a script that adds new labeled images to the training set

2. **Retrain pipeline**
   - Script that: pulls latest data (DVC) → trains → evaluates → compares with current production model
   - Only promote new model if it beats current production metrics
   - Log everything in MLflow

3. **Model versioning strategy**
   - Production model is always the one tagged "Production" in MLflow registry
   - Rollback = change the tag back to previous version
   - API loads model version from config / environment variable

4. **Automate with GitHub Actions** (optional)
   - Trigger retraining on schedule (weekly) or on new data push
   - Auto-create PR with new model metrics for human review

### Skills Learned

- ML retraining strategies
- Model versioning and rollback
- Automated ML pipelines
- Human-in-the-loop model promotion

---

## Phase 11: Documentation & Polish

**Duration:** 1–2 days
**Objective:** Make the project presentable and reproducible.

### Tasks

1. **README.md** — the first thing anyone sees
   - One-paragraph project description
   - Architecture diagram (Mermaid)
   - Quickstart: clone → install → run in 3 commands
   - API usage examples with curl
   - Development guide: how to train, test, deploy

2. **API documentation**
   - Auto-generated by FastAPI (Swagger UI at `/docs`)
   - Add examples for each endpoint

3. **Architecture diagram**
   - Use Mermaid (renders in GitHub markdown)
   - Show data flow: user → frontend → API → model → response
   - Show monitoring flow: API → Prometheus → Grafana

4. **Lessons learned document** (for your own growth)
   - What was harder than expected?
   - What would you do differently?
   - What tool surprised you (positively or negatively)?

---

## Timeline Summary

```
Week 1  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 1: Setup          (2 days)
        Phase 2: Data Pipeline  (3 days)

Week 2  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 3: Model Development  (5 days)

Week 3  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 4: Evaluation     (2 days)
        Phase 5: API & Serving  (3 days)

Week 4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 6: Docker         (2 days)
        Phase 7: CI/CD          (3 days)

Week 5  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 8: Deployment     (2 days)
        Phase 9: Monitoring     (3 days)

Week 6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 10: Retraining    (2 days)
        Phase 11: Documentation (2 days)
        Buffer / catch-up       (1 day)
```

**Total: ~30 days (6 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] Writing ML design documents
- [ ] Python project setup (pyproject.toml, virtual environments)
- [ ] Git workflow for ML (branching, .gitignore, large files)
- [ ] Data versioning with DVC
- [ ] Exploratory data analysis for computer vision
- [ ] PyTorch Dataset, DataLoader, transforms
- [ ] Transfer learning with pretrained models
- [ ] Training loop with validation, early stopping, checkpointing
- [ ] Experiment tracking with MLflow
- [ ] Model registry and versioning
- [ ] ML evaluation metrics (precision, recall, F1, AUC-ROC)
- [ ] Error analysis and model debugging
- [ ] Model optimization (ONNX export, quantization)
- [ ] Building REST APIs with FastAPI
- [ ] API input validation and security
- [ ] Batch and real-time inference
- [ ] Streamlit frontend development
- [ ] Docker multi-stage builds
- [ ] docker-compose for multi-service apps
- [ ] Writing unit, integration, and ML-specific tests
- [ ] CI/CD with GitHub Actions
- [ ] Cloud deployment (Railway / GCP Cloud Run)
- [ ] Load testing
- [ ] Prometheus metrics instrumentation
- [ ] Grafana dashboard creation
- [ ] ML monitoring (prediction drift, confidence drift)
- [ ] Structured logging
- [ ] Model retraining pipelines
- [ ] Technical documentation (README, model cards)

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Initialize the project
cd ml-end-to-end
git init

# 2. Create the folder structure
mkdir -p configs data/{raw,processed,splits} notebooks src/{data,model,training,serving,monitoring,frontend} tests/{unit,integration} docker .github/workflows grafana/dashboards prometheus scripts

# 3. Start writing DESIGN_DOC.md
```
