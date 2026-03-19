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
