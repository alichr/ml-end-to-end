# ML End-to-End: Cat vs Dog Classifier

Production-grade image classifier built with PyTorch, FastAPI, MLflow, Docker, and Prometheus/Grafana. The ML is simple (binary classification) — the focus is on the **engineering pipeline**.

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

---

## Phase 1: Project Setup

### Step 1 — Write the Design Doc

Create `DESIGN_DOC.md` with:
- Problem statement
- Success criteria (accuracy ≥ 95%, latency < 200ms, model < 50MB, 50 concurrent requests)
- Out of scope items
- Risks and mitigations

### Step 2 — Initialize the Repository

```bash
# Initialize git
git init

# Create .gitignore to exclude data, models, python artifacts, env files
# See .gitignore in this repo for the full list

# Create pyproject.toml with all dependencies
# See pyproject.toml in this repo

# Set up branch strategy
git add .
git commit -m "Initial commit"
git branch -m master main        # rename default branch to main
git checkout -b dev              # create dev branch

# Push to GitHub (create an empty repo on github.com first)
git remote add origin https://github.com/<your-username>/ml-end-to-end.git
git push -u origin main
git push -u origin dev
```

### Step 3 — Create the Folder Structure

```bash
mkdir -p configs \
  data/{raw,processed,splits} \
  notebooks \
  src/{data,model,training,serving,monitoring,frontend} \
  tests/{unit,integration} \
  docker \
  .github/workflows \
  grafana/dashboards \
  prometheus \
  scripts

# Add __init__.py so Python recognizes packages
touch src/__init__.py \
  src/{data,model,training,serving,monitoring}/__init__.py \
  tests/__init__.py \
  tests/{unit,integration}/__init__.py
```

### Step 4 — Set Up Development Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install project + dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (ruff, mypy, trailing whitespace, etc.)
pre-commit install
```

The pre-commit config (`.pre-commit-config.yaml`) runs on every commit:
- **ruff** — linting and formatting
- **mypy** — type checking
- **trailing-whitespace** — removes trailing spaces
- **check-yaml** — validates YAML files
- **check-added-large-files** — prevents accidental large file commits (> 5MB)

The `.editorconfig` ensures consistent formatting across editors (4-space indent, UTF-8, LF line endings).

---

## Phase 2: Data Pipeline

### Step 1 — Download the Dataset

The dataset is the Microsoft Cats vs Dogs dataset (~25,000 images) from Kaggle.

**Prerequisites:**
1. Create a Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings → API → Create New Token
3. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
4. Run: `chmod 600 ~/.kaggle/kaggle.json`

**Download:**
```bash
python -m src.data.download
```

The script is **idempotent** — if `data/raw/` already has 25,000 images, it skips the download. Images are saved as `cat.0.jpg`, `cat.1.jpg`, ..., `dog.0.jpg`, `dog.1.jpg`, etc.

### Step 2 — Set Up DVC (Data Version Control)

DVC tracks large data files outside of git. A small `.dvc` pointer file is committed to git, while the actual data lives in remote storage.

```bash
# Initialize DVC
dvc init

# Track the raw data
dvc add data/raw

# Configure remote storage (Google Drive with service account)
dvc remote add -d gdrive gdrive://<folder-id>
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path ~/.gdrive/credentials.json

# Or use local storage for quick setup
dvc remote add -d local /tmp/dvc-storage

# Push data to remote
dvc push
```

To reproduce the dataset on another machine:
```bash
dvc pull
```

### Step 3 — Exploratory Data Analysis

Run the EDA notebook to explore the dataset:

```bash
cd notebooks
jupyter notebook 01_eda.ipynb
```

The notebook covers:
- Class distribution (cats vs dogs count)
- Image size distribution (width, height, scatter plot)
- Sample image grid (16 cats, 16 dogs)
- Corrupted image detection
- File format and color mode distribution
- File size distribution

### Step 4 — Data Preprocessing

```bash
# Validate images: removes corrupted files, converts all to RGB JPEG
python -m src.data.validate

# Split into train (70%), val (15%), test (15%) — stratified by class
python -m src.data.split
```

This creates:
- `data/processed/` — cleaned images (corrupted ones removed, all converted to RGB)
- `data/splits/{train,val,test}/{cat,dog}/` — stratified split ready for training

Key files:
- `src/data/validate.py` — scans and removes corrupted images
- `src/data/transforms.py` — training (with augmentation) and inference transforms
- `src/data/split.py` — stratified train/val/test split
- `src/data/dataset.py` — PyTorch `Dataset` class with lazy loading

### Step 5 — DVC Pipeline

The full data pipeline is defined in `dvc.yaml`: download → validate → split.

```bash
# Run the entire pipeline (or re-run if inputs change)
dvc repro
```

---

## Phase 3: Model Development & Experiment Tracking

### Step 1 — Define the Model

The classifier uses **MobileNetV2** pretrained on ImageNet with a custom classification head (`Linear(1280, 2)`). The backbone can be fully frozen or partially unfrozen for fine-tuning.

Key file: `src/model/classifier.py`

### Step 2 — Training Config

Training hyperparameters are defined in YAML config files under `configs/`. The default config is `configs/train_config.yaml`.

### Step 3 — Train the Model

```bash
# Train with default config
python -m src.training.train

# Train with a specific experiment config
python -m src.training.train --config configs/experiment_finetune.yaml
```

The training loop includes:
- Validation after each epoch
- Early stopping (patience=3)
- Best model checkpoint saved to `models/best_model.pth`
- MLflow logging (params, metrics per epoch, confusion matrix, model artifact)

### Step 4 — MLflow Experiment Tracking

All training runs are logged to MLflow. To view results:

```bash
mlflow ui
# Open http://localhost:5000
```

Each run logs: hyperparameters, per-epoch train/val loss & accuracy, best val accuracy, confusion matrix plot, and the model artifact.

### Step 5 — Run Experiments

Five experiments are configured to compare different training strategies:

```bash
bash scripts/run_experiments.sh
```

| Experiment | Config | What Changes | Result |
|-----------|--------|-------------|--------|
| Baseline | `experiment_baseline.yaml` | Frozen backbone, lr=0.001 | 97.65% |
| Fine-tune | `experiment_finetune.yaml` | Unfreeze last 3 layers, lr=0.0001 | **98.40%** |
| Augmentation | `experiment_augmentation.yaml` | Stronger augmentation | 96.93% |
| Batch size | `experiment_batch_size.yaml` | Batch size 64 | 97.28% |
| Scheduler | `experiment_scheduler.yaml` | StepLR instead of cosine | 97.52% |

### Step 6 — Promote the Best Model

```bash
# Auto-selects the run with highest val accuracy and registers it
python scripts/promote_model.py
```

The best model (`finetune-last3`, 98.4% val accuracy) is registered as `cat-dog-classifier` with the `production` alias in MLflow Model Registry.

---

## Phase 4: Evaluation & Model Optimization

### Step 1 — Comprehensive Evaluation

```bash
python -m src.training.evaluate
```

Evaluates the best model on the held-out test set (3,752 images):

| Metric | Value | Target |
|--------|-------|--------|
| Accuracy | 98.72% | >= 95% |
| Precision | 98.72% | — |
| Recall | 98.72% | — |
| F1 Score | 98.72% | — |
| AUC-ROC | 99.92% | — |

Generates plots (confusion matrix, ROC curve, per-class metrics) in `models/evaluation/`.

### Step 2 — Error Analysis

```bash
cd notebooks
jupyter notebook 03_evaluation.ipynb
```

The notebook displays the top 20 most confident wrong predictions, analyzes error patterns by class, and compares confidence distributions for correct vs wrong predictions.

### Step 3 — Performance Benchmarks

| Platform | Latency (mean) | Throughput | Target |
|----------|---------------|------------|--------|
| CPU (PyTorch) | 6.6 ms | 152 img/s | < 200 ms |
| GPU (PyTorch) | 1.0 ms | 973 img/s | — |
| CPU (ONNX Runtime) | 1.4 ms | 714 img/s | — |

Model size: 8.73 MB (PyTorch), 0.30 MB (ONNX) — target < 50 MB.

### Step 4 — ONNX Export

```bash
python -m src.model.export
```

Exports the model to ONNX format, verifies output matches PyTorch, and benchmarks ONNX Runtime inference. The ONNX model is ~4.7x faster than PyTorch on CPU.

### Step 5 — Model Card

See [`MODEL_CARD.md`](MODEL_CARD.md) for full model documentation including intended use, training data, evaluation metrics, known limitations, and usage examples.

---

## Phase 5: API & Serving Layer

### Step 1 — API Schemas

Pydantic models define the API contract in `src/serving/schemas.py`:

- `PredictionResponse` — predicted class, confidence, probabilities, model version, latency
- `BatchPredictionResponse` — list of predictions with total latency
- `HealthResponse` — status, model loaded, version, uptime
- `ErrorResponse` — standardized error format with request ID

### Step 2 — Inference Engine

The `Predictor` class (`src/serving/predict.py`) loads the ONNX model and runs inference:

- Preprocesses images with the same pipeline as training (Resize → CenterCrop → Normalize)
- Handles edge cases: grayscale, RGBA, very small, and very large images
- Supports both single and batch prediction
- Pure numpy preprocessing — no PyTorch dependency at serving time

### Step 3 — FastAPI Application

```bash
# Start the API server
uvicorn src.serving.app:app --reload
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Classify a single image |
| `POST` | `/predict/batch` | Classify up to 16 images at once |
| `GET` | `/health` | Health check (model status, uptime) |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Interactive Swagger UI (auto-generated) |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@cat.jpg" | python -m json.tool
```

```json
{
    "predicted_class": "cat",
    "confidence": 0.9847,
    "probabilities": {"cat": 0.9847, "dog": 0.0153},
    "model_version": "1.0.0",
    "latency_ms": 3.81
}
```

### Step 4 — Middleware

`src/serving/middleware.py` provides production hardening:

- **Input validation** — JPEG/PNG only, max 5 MB file size
- **Rate limiting** — 100 requests/minute per IP (in-memory)
- **Request ID** — unique UUID per request via `X-Request-ID` header
- **CORS** — configured for frontend access
- **Error handling** — clean JSON errors, no stack traces exposed

### Step 5 — Streamlit Frontend

```bash
# Start the frontend (API must be running)
streamlit run src/frontend/app.py
```

Features:
- Dark glassmorphism UI with gradient accents
- Single image and batch upload tabs
- Animated confidence bars with per-class probabilities
- Live API health status in sidebar
- Model specs display (architecture, accuracy, latency)

### Step 6 — Tests

```bash
# Run all tests (33 tests)
pytest tests/ -v
```

| Suite | File | Tests |
|-------|------|-------|
| Unit | `tests/unit/test_schemas.py` | Schema validation, confidence range, error models |
| Unit | `tests/unit/test_predict.py` | Preprocessing shapes, softmax, edge cases (grayscale, RGBA, tiny/large) |
| Integration | `tests/integration/test_api.py` | Full API pipeline, invalid inputs, batch limits, health, metrics, request ID |

---

## Tech Stack

| Category | Tool |
|----------|------|
| ML Framework | PyTorch + torchvision |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| API | FastAPI |
| Frontend | Streamlit |
| Containers | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Testing | pytest |
| Linting | Ruff + mypy |
| Model Optimization | ONNX Runtime |
