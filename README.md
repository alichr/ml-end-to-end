# ML Engineer Roadmap: 12 Production-Grade Projects

A comprehensive, hands-on roadmap to becoming a job-ready ML Engineer. Each project focuses on **engineering best practices** — not just model accuracy — covering the full ML lifecycle from design doc to production monitoring.

## The Roadmap

| # | Project | Difficulty | Domain | Key Skills |
|---|---------|-----------|--------|------------|
| 1 | [**Cat vs Dog Classifier**](projects/01_cat_dog_classifier/) | Beginner | Computer Vision | PyTorch, FastAPI, Docker, MLflow, Prometheus |
| 2 | [**Sentiment Analysis API**](projects/02_sentiment_analysis/) | Beginner+ | NLP | Hugging Face Transformers, tokenization, GPU training |
| 3 | [**House Price Prediction**](projects/03_house_price/) | Intermediate | Tabular Data | XGBoost, SHAP interpretability, feature engineering |
| 4 | [**Energy Demand Forecasting**](projects/04_energy_forecasting/) | Intermediate | Time Series | LSTM, Prophet, Airflow, scheduled retraining |
| 5 | [**Movie Recommendation Engine**](projects/05_recommendation/) | Intermediate+ | RecSys | Embeddings, collaborative filtering, A/B testing, Redis |
| 6 | [**Object Detection & Tracking**](projects/06_object_detection/) | Hard | Advanced CV | YOLOv8, DeepSORT, TensorRT, edge deployment |
| 7 | [**Real-Time Fraud Detection**](projects/07_fraud_detection/) | Hard | Streaming ML | Kafka, Feast feature store, online serving |
| 8 | [**RAG-Powered Q&A System**](projects/08_rag_qa/) | Hard+ | LLMs | Vector databases, retrieval, prompt engineering |
| 9 | [**Multi-Modal Content Moderation**](projects/09_content_moderation/) | Expert | Multi-Modal | Text + image fusion, Celery, policy engine, bias auditing |
| 10 | [**Autonomous AI Agent**](projects/10_ai_agent/) | Expert | GenAI / Agentic | Tool use, ReAct loops, sandboxed execution, guardrails |
| 11 | [**Multi-Agent Collaboration Platform**](projects/11_multi_agent/) | Expert+ | GenAI / Multi-Agent | Agent orchestration, distributed tracing, RabbitMQ |
| 12 | [**Internal ML Platform**](projects/12_ml_platform/) | Expert+ | MLOps / Infra | Kubernetes, Terraform, feature store, multi-tenant serving |

## Progress

- [x] **Project 1** — Cat vs Dog Classifier (Phases 1-6 complete)
- [ ] Project 2 — Sentiment Analysis API
- [ ] Project 3 — House Price Prediction
- [ ] Project 4 — Energy Demand Forecasting
- [ ] Project 5 — Movie Recommendation Engine
- [ ] Project 6 — Object Detection & Tracking
- [ ] Project 7 — Real-Time Fraud Detection
- [ ] Project 8 — RAG-Powered Q&A System
- [ ] Project 9 — Multi-Modal Content Moderation
- [ ] Project 10 — Autonomous AI Agent
- [ ] Project 11 — Multi-Agent Collaboration Platform
- [ ] Project 12 — Internal ML Platform

## Repository Structure

```
ml-end-to-end/
├── README.md                          # This file — roadmap overview
├── .gitignore
├── .pre-commit-config.yaml
├── .editorconfig
│
├── projects/
│   ├── 01_cat_dog_classifier/         # ✅ In progress
│   │   ├── README.md                  # Project-specific guide
│   │   ├── pyproject.toml
│   │   ├── dvc.yaml
│   │   ├── docker-compose.yaml
│   │   ├── configs/
│   │   ├── data/
│   │   ├── models/
│   │   ├── notebooks/
│   │   ├── src/
│   │   ├── tests/
│   │   ├── docker/
│   │   ├── prometheus/
│   │   ├── grafana/
│   │   ├── scripts/
│   │   └── doc/
│   │       ├── DESIGN_DOC.md
│   │       ├── MODEL_CARD.md
│   │       └── PROJECT_PLAN.md
│   │
│   ├── 02_sentiment_analysis/         # 📋 Planned
│   │   ├── README.md
│   │   └── doc/PROJECT_PLAN.md
│   │
│   ├── ...                            # Projects 3-11
│   │
│   └── 12_ml_platform/               # 📋 Planned
│       ├── README.md
│       └── doc/PROJECT_PLAN.md
```

## How to Use This Repo

Each project is self-contained in its own folder under `projects/`. To work on a project:

```bash
# Clone the repo
git clone https://github.com/alichr/ml-end-to-end.git
cd ml-end-to-end

# Navigate to the project you want to work on
cd projects/01_cat_dog_classifier

# Read the project-specific README
cat README.md

# Read the detailed project plan
cat doc/PROJECT_PLAN.md
```

Each project follows the same pattern:
1. **Design Doc** — define the problem, constraints, and success criteria
2. **Data Pipeline** — download, validate, preprocess, version with DVC
3. **Model Development** — train, experiment, track with MLflow
4. **Evaluation** — metrics, error analysis, benchmarks
5. **API & Serving** — FastAPI, inference optimization
6. **Containerization** — Docker, docker-compose
7. **Testing & CI/CD** — pytest, GitHub Actions
8. **Deployment** — cloud deployment
9. **Monitoring** — Prometheus, Grafana, drift detection

## Skills You'll Master

By completing all 12 projects, you will have production experience with:

**ML Domains:** Computer Vision, NLP, Tabular Data, Time Series, Recommendation Systems, Object Detection, Streaming ML, LLMs/RAG, Multi-Modal AI, Agentic AI, ML Platform Engineering

**Core Tools:** PyTorch, Hugging Face, scikit-learn, XGBoost, ONNX Runtime, FastAPI, Docker, Kubernetes, Prometheus, Grafana, MLflow, DVC, Airflow, Kafka, Redis, PostgreSQL, Terraform

**Engineering Skills:** API design, microservices, CI/CD, monitoring, A/B testing, feature stores, model optimization, edge deployment, distributed tracing, infrastructure as code

## Tech Stack Overview

| Category | Tools |
|----------|-------|
| ML Frameworks | PyTorch, Hugging Face, scikit-learn, XGBoost, LightGBM |
| LLM / GenAI | Claude API, OpenAI API, LangChain, LangGraph, ChromaDB |
| Serving | FastAPI, ONNX Runtime, TensorRT, Seldon Core |
| Data | DVC, Feast, Pandas, Apache Kafka, Apache Flink |
| Infrastructure | Docker, Kubernetes, Terraform, Helm |
| Monitoring | Prometheus, Grafana, OpenTelemetry, Evidently |
| CI/CD | GitHub Actions, pytest, Ruff, mypy |
| Databases | PostgreSQL, Redis, ChromaDB, Pinecone |
| Frontend | Streamlit, React/Next.js |
| Orchestration | Airflow, Celery, RabbitMQ |
