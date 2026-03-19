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
