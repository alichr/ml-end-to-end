# End-to-End ML Project Plan: Sentiment Analysis API

## Goal

Build a **production-grade** sentiment analysis API that classifies text reviews as
**positive, negative, or neutral** using a fine-tuned **DistilBERT** model. This project
introduces NLP, the Hugging Face ecosystem, and the unique challenges of deploying
language models -- variable-length inputs, tokenization pipelines, and GPU-accelerated
training.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why sentiment analysis? | It is the "hello world" of NLP -- simple enough to focus on engineering, complex enough to be realistic. Every company with user reviews needs this. |
| Why DistilBERT? | 40% smaller than BERT, 60% faster, retains 97% of performance. Perfect balance of quality and efficiency for production. |
| Why Hugging Face? | Industry standard for NLP. Learning `transformers`, `datasets`, and `tokenizers` is a direct job skill. |
| How is this different from Project 1? | NLP instead of CV. Variable-length text inputs, subword tokenization, attention mechanisms, and a new ecosystem (Hugging Face). |

---

## Architecture Overview

```
                    ┌──────────────┐
                    │   Streamlit  │
                    │   Frontend   │
                    │  (text input)│
                    └──────┬───────┘
                           │ HTTP
                           ▼
                    ┌──────────────┐      ┌──────────────────┐
  Review Text ─────▶   FastAPI    │──────▶│  DistilBERT      │
                    │   Server    │       │  (Fine-tuned)    │
                    └──────┬───────┘      └──────────────────┘
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
| ML Framework | PyTorch + Hugging Face Transformers | Industry standard for NLP |
| Tokenization | Hugging Face Tokenizers | Rust-backed, fast subword tokenization |
| Datasets | Hugging Face Datasets | Efficient dataset loading, built-in streaming |
| Experiment Tracking | MLflow | Tracks params, metrics, artifacts, model registry |
| Data Versioning | DVC | Git for data -- essential for reproducibility |
| API Framework | FastAPI | Fast, modern, auto-generates OpenAPI docs |
| Frontend | Streamlit | Simplest way to build ML UIs |
| Containerization | Docker + docker-compose | Deployment standard everywhere |
| CI/CD | GitHub Actions | Free, integrated with GitHub |
| Monitoring | Prometheus + Grafana | Industry standard observability |
| Testing | pytest | Python standard |
| Linting | Ruff + mypy | Fast linting + type checking |
| Config Management | YAML + Pydantic | Clean, validated configurations |

---

## Project Structure

```
sentiment-analysis/
│
├── doc/
│   ├── PROJECT_PLAN.md              # This file
│   ├── DESIGN_DOC.md                # Problem statement, constraints, success criteria
│   └── MODEL_CARD.md                # Model documentation (what, how, limitations)
│
├── README.md                        # Setup instructions, architecture, quickstart
├── pyproject.toml                   # Dependencies and project metadata
├── dvc.yaml                         # Data pipeline definition
├── dvc.lock                         # Data pipeline lock file
│
├── configs/
│   ├── train_config.yaml            # Training hyperparameters
│   ├── serve_config.yaml            # Serving configuration
│   └── data_config.yaml             # Data paths, splits, preprocessing
│
├── data/                            # Git-ignored, DVC-tracked
│   ├── raw/                         # Original downloaded reviews
│   ├── processed/                   # Cleaned, tokenized data
│   └── splits/                      # train/ val/ test/ splits
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_training.ipynb            # Interactive training experiments
│   └── 03_evaluation.ipynb          # Model evaluation & error analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py              # Download dataset script
│   │   ├── preprocess.py            # Text cleaning, label mapping
│   │   ├── tokenize_data.py         # Hugging Face tokenization pipeline
│   │   └── validate.py              # Data integrity checks
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── classifier.py            # DistilBERT + classification head
│   │   └── export.py                # Export to ONNX
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                 # Fine-tuning loop (Hugging Face Trainer)
│   │   ├── evaluate.py              # Evaluation metrics
│   │   └── callbacks.py             # Early stopping, logging callbacks
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application
│   │   ├── schemas.py               # Request/response Pydantic models
│   │   ├── middleware.py            # Rate limiting, auth, CORS
│   │   └── predict.py               # Inference logic with tokenization
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Prometheus metric definitions
│   │   └── drift.py                 # Text-specific drift monitoring
│   │
│   └── frontend/
│       └── app.py                   # Streamlit UI
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py    # Text cleaning tests
│   │   ├── test_tokenization.py     # Tokenizer output validation
│   │   ├── test_model.py            # Model output shape, export
│   │   └── test_schemas.py          # API schema validation
│   ├── integration/
│   │   ├── test_api.py              # Full API predict pipeline
│   │   └── test_training.py         # Training runs without error
│   └── conftest.py                  # Shared fixtures (sample texts, model)
│
├── docker/
│   ├── Dockerfile.api               # Multi-stage build for API
│   ├── Dockerfile.frontend          # Streamlit container
│   └── Dockerfile.training          # Training environment (GPU-ready)
│
├── docker-compose.yaml              # Orchestrate all services
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                  # Lint -> Test -> Build on PR
│       └── cd.yaml                  # Deploy on merge to main
│
├── grafana/
│   └── dashboards/
│       └── nlp_monitoring.json      # Pre-configured dashboard
│
├── prometheus/
│   └── prometheus.yml               # Scrape config
│
└── scripts/
    ├── setup.sh                     # One-command project setup
    ├── train.sh                     # Run training with default config
    └── deploy.sh                    # Deploy to cloud
```

---

## Phase 1: Project Setup & Design Doc

**Duration:** 1-2 days
**Objective:** Define what you are building before writing any code.

### Tasks

1. **Write `DESIGN_DOC.md`**
   - **Problem statement:** "Given a text review, classify its sentiment as positive, negative, or neutral"
   - **Success criteria:**
     - Macro F1 score >= 0.85 on test set
     - Inference latency < 100ms per review on CPU
     - Model size < 300MB (DistilBERT is ~260MB)
     - API handles 100 concurrent requests
     - Supports reviews up to 512 tokens
   - **Out of scope:** aspect-based sentiment, sarcasm detection, multilingual support, real-time streaming
   - **Risks:** class imbalance (neutral is often underrepresented), ambiguous reviews, domain shift between training data and real inputs

2. **Initialize the repository**
   - `git init`, create `.gitignore` (data/, models/, *.pyc, .env, __pycache__/, wandb/, mlruns/)
   - Set up branch strategy: `main` (production), `dev` (integration), `feature/*`
   - Create `pyproject.toml` with all dependencies:
     ```toml
     [project]
     name = "sentiment-analysis"
     version = "0.1.0"
     requires-python = ">=3.11"
     dependencies = [
         "transformers>=4.36.0",
         "datasets>=2.16.0",
         "tokenizers>=0.15.0",
         "torch>=2.1.0",
         "accelerate>=0.25.0",
         "scikit-learn>=1.3.0",
         "mlflow>=2.9.0",
         "fastapi>=0.108.0",
         "uvicorn>=0.25.0",
         "streamlit>=1.29.0",
         "prometheus-client>=0.19.0",
         "pydantic>=2.5.0",
         "pyyaml>=6.0",
         "structlog>=23.2.0",
         "dvc>=3.30.0",
         "pytest>=7.4.0",
         "ruff>=0.1.9",
         "mypy>=1.8.0",
     ]
     ```

3. **Create the folder structure** (as shown above)

4. **Set up development environment**
   - Python virtual environment (`uv` or `venv`)
   - Pre-commit hooks: ruff, mypy, trailing whitespace
   - Verify GPU availability (recommended for fine-tuning):
     ```python
     import torch
     print(f"CUDA available: {torch.cuda.is_available()}")
     print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
     ```

### Skills Learned

- Writing ML design documents for NLP tasks
- Professional Python project setup for Hugging Face projects
- Understanding GPU requirements for transformer models

---

## Phase 2: Data Pipeline

**Duration:** 3-5 days
**Objective:** Get clean, versioned, tokenized data ready for fine-tuning.

### Tasks

1. **Download the dataset** -- `src/data/download.py`
   - Use the Hugging Face `datasets` library to download Yelp Reviews or Amazon Reviews:
     ```python
     from datasets import load_dataset

     # Option A: Yelp Reviews (5 classes -> map to 3)
     dataset = load_dataset("yelp_review_full")

     # Option B: Amazon Reviews
     dataset = load_dataset("amazon_polarity")
     ```
   - Map star ratings to sentiment labels:
     - 1-2 stars -> negative
     - 3 stars -> neutral
     - 4-5 stars -> positive
   - Subsample if needed (full Yelp has 650K reviews -- start with 50K for faster iteration)
   - Write an automated, idempotent download script

2. **Set up DVC (Data Version Control)**
   - `dvc init`, configure remote storage
   - Track `data/raw/` with DVC
   - Learn: `dvc add`, `dvc push`, `dvc pull`
   - **Why DVC matters for NLP:** text datasets can be large, and you need to track which version of the data produced which model

3. **Exploratory Data Analysis (EDA)** -- `notebooks/01_eda.ipynb`
   - Class distribution (positive / negative / neutral counts and percentages)
   - Review length distribution (word count and character count histograms)
   - Average review length per class (are negative reviews longer?)
   - Word frequency analysis per class (top 30 words per sentiment)
   - Sample reviews from each class (5-10 examples each)
   - Identify potential issues:
     - Duplicate reviews
     - Empty or very short reviews (< 5 words)
     - Reviews with HTML artifacts, URLs, special characters
     - Language detection (filter non-English if needed)
   ```python
   import matplotlib.pyplot as plt
   import pandas as pd

   # Class distribution
   df["sentiment"].value_counts().plot(kind="bar")
   plt.title("Sentiment Class Distribution")

   # Review length distribution
   df["word_count"] = df["text"].str.split().str.len()
   df["word_count"].hist(bins=50)
   plt.title("Review Length Distribution (words)")
   plt.xlabel("Number of words")
   ```

4. **Data preprocessing** -- `src/data/preprocess.py`
   - Text cleaning pipeline:
     ```python
     def clean_text(text: str) -> str:
         """Clean a review text for model input."""
         # Remove HTML tags
         text = re.sub(r"<[^>]+>", "", text)
         # Remove URLs
         text = re.sub(r"http\S+|www\.\S+", "", text)
         # Remove excessive whitespace
         text = re.sub(r"\s+", " ", text).strip()
         # Keep punctuation (important for sentiment!)
         # Keep case (DistilBERT is uncased, but log original for analysis)
         return text
     ```
   - Label encoding: positive=2, neutral=1, negative=0
   - Handle class imbalance:
     - Option A: undersample majority class
     - Option B: use class weights in loss function (recommended)
     - Option C: oversample minority class with random duplication
   - Split into train (70%), validation (15%), test (15%) -- **stratified by class**
   - Validate splits: ensure class distribution is maintained in each split

5. **Tokenization pipeline** -- `src/data/tokenize_data.py`
   - Use Hugging Face tokenizers with DistilBERT vocabulary:
     ```python
     from transformers import AutoTokenizer

     tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

     def tokenize_function(examples):
         return tokenizer(
             examples["text"],
             padding="max_length",
             truncation=True,
             max_length=256,  # Most reviews fit in 256 tokens
             return_tensors="pt",
         )

     tokenized_dataset = dataset.map(tokenize_function, batched=True)
     ```
   - Key decisions:
     - `max_length`: analyze token length distribution from EDA to pick the right value (128, 256, or 512)
     - `padding`: "max_length" for batched training, "longest" for dynamic batching at inference
     - `truncation`: True -- reviews longer than max_length are truncated
   - Understand subword tokenization:
     ```python
     # "unhappiness" -> ["un", "##happi", "##ness"]
     tokenizer.tokenize("The food was absolutely terrible and unhygienic")
     # -> ['the', 'food', 'was', 'absolutely', 'terrible', 'and', 'un', '##hy', '##gi', '##enic']
     ```
   - Save tokenized datasets to disk for fast loading during training

6. **Define DVC pipeline** -- `dvc.yaml`
   - Stage 1: download -> Stage 2: clean -> Stage 3: tokenize -> Stage 4: split
   - Reproducible with `dvc repro`
   ```yaml
   stages:
     download:
       cmd: python -m src.data.download
       outs:
         - data/raw/
     preprocess:
       cmd: python -m src.data.preprocess
       deps:
         - data/raw/
         - src/data/preprocess.py
       outs:
         - data/processed/
     tokenize:
       cmd: python -m src.data.tokenize_data
       deps:
         - data/processed/
         - src/data/tokenize_data.py
       outs:
         - data/splits/
   ```

### Skills Learned

- Hugging Face `datasets` library for loading and processing data
- Subword tokenization (BPE/WordPiece) -- how transformers see text
- Text preprocessing for NLP (different from CV -- punctuation matters!)
- Handling class imbalance in classification
- EDA for text data (length distributions, word frequencies)
- DVC pipelines for NLP data

---

## Phase 3: Model Development & Experiment Tracking

**Duration:** 4-5 days
**Objective:** Fine-tune DistilBERT, track experiments, pick the best model.

### Tasks

1. **Define the model** -- `src/model/classifier.py`
   - Use DistilBERT with a classification head:
     ```python
     from transformers import AutoModelForSequenceClassification

     class SentimentClassifier:
         def __init__(self, num_labels=3, model_name="distilbert-base-uncased"):
             self.model = AutoModelForSequenceClassification.from_pretrained(
                 model_name,
                 num_labels=num_labels,
                 problem_type="single_label_classification",
             )
             self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

         def freeze_base(self, num_layers_to_freeze=4):
             """Freeze early transformer layers, fine-tune later ones."""
             for name, param in self.model.distilbert.named_parameters():
                 layer_num = int(name.split(".")[2]) if "layer" in name else -1
                 if layer_num < num_layers_to_freeze:
                     param.requires_grad = False
     ```
   - Why DistilBERT:
     - 66M parameters (vs 110M for BERT-base)
     - 6 transformer layers (vs 12 for BERT-base)
     - 2x faster inference
     - Retains 97% of BERT's language understanding

2. **Write training config** -- `configs/train_config.yaml`
   ```yaml
   model:
     name: distilbert-base-uncased
     num_labels: 3
     freeze_layers: 0  # Fine-tune all layers

   training:
     epochs: 5
     batch_size: 32
     learning_rate: 2e-5      # Standard for transformer fine-tuning
     weight_decay: 0.01
     warmup_ratio: 0.1        # Warm up LR over first 10% of steps
     fp16: true               # Mixed precision (2x faster on GPU)
     gradient_accumulation_steps: 2
     max_length: 256
     seed: 42

   data:
     train_split: 0.7
     val_split: 0.15
     max_samples: 50000       # Start small, scale up later
     class_weights: auto       # Compute from class distribution
   ```

3. **Write the training loop** -- `src/training/train.py`
   - Use Hugging Face `Trainer` for clean, production-quality training:
     ```python
     from transformers import Trainer, TrainingArguments
     import mlflow

     training_args = TrainingArguments(
         output_dir="./results",
         num_train_epochs=5,
         per_device_train_batch_size=32,
         per_device_eval_batch_size=64,
         learning_rate=2e-5,
         weight_decay=0.01,
         warmup_ratio=0.1,
         evaluation_strategy="epoch",
         save_strategy="epoch",
         load_best_model_at_end=True,
         metric_for_best_model="f1_macro",
         fp16=True,
         report_to="mlflow",
     )

     def compute_metrics(eval_pred):
         predictions, labels = eval_pred
         preds = predictions.argmax(axis=-1)
         f1 = f1_score(labels, preds, average="macro")
         acc = accuracy_score(labels, preds)
         return {"accuracy": acc, "f1_macro": f1}

     trainer = Trainer(
         model=model,
         args=training_args,
         train_dataset=tokenized_train,
         eval_dataset=tokenized_val,
         compute_metrics=compute_metrics,
     )

     with mlflow.start_run():
         trainer.train()
     ```
   - Handle class imbalance with weighted loss:
     ```python
     from torch.nn import CrossEntropyLoss

     # Compute class weights from training distribution
     class_counts = [num_negative, num_neutral, num_positive]
     total = sum(class_counts)
     class_weights = torch.tensor([total / (3 * c) for c in class_counts])

     # Custom Trainer with weighted loss
     class WeightedTrainer(Trainer):
         def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
             labels = inputs.pop("labels")
             outputs = model(**inputs)
             loss_fn = CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
             loss = loss_fn(outputs.logits, labels)
             return (loss, outputs) if return_outputs else loss
     ```

4. **Set up MLflow experiment tracking**
   - Log per-experiment: hyperparameters, train/val loss curves, final metrics
   - Log artifacts: best model checkpoint, tokenizer, confusion matrix plot
   - **Register the best model** in MLflow Model Registry
   - Key metrics to track:
     - `f1_macro` (primary metric -- handles class imbalance)
     - `accuracy`
     - `f1_per_class` (positive, negative, neutral separately)
     - `train_loss`, `val_loss` per epoch
     - `learning_rate` schedule

5. **Run experiments** (track all in MLflow)

   | Experiment | What Changes | Expected Result |
   |-----------|-------------|----------------|
   | Baseline | All defaults, lr=2e-5, 3 epochs | ~0.80 F1 macro |
   | More epochs | 5 epochs with early stopping | ~0.83 F1 macro |
   | Unweighted | Remove class weights | Lower neutral F1 |
   | Longer context | max_length=512 | Marginal improvement, 2x slower |
   | Lower LR | lr=5e-6, 8 epochs | Potentially better convergence |
   | Frozen layers | Freeze first 4 layers | Faster training, slightly lower F1 |
   | Larger batch | batch_size=64, lr=4e-5 | Faster training, comparable F1 |

6. **Pick the best model** using MLflow UI
   - Compare runs side by side on F1 macro
   - Check for overfitting (gap between train and val metrics)
   - Select best model -> "Promote to Production" in registry
   - Save the tokenizer alongside the model (they are paired)

### Skills Learned

- Fine-tuning pretrained transformer models
- Hugging Face Trainer API (the standard for NLP training)
- Learning rate scheduling for transformers (warmup is critical)
- Mixed precision training (fp16) for GPU efficiency
- Handling class imbalance in NLP
- Experiment tracking for NLP models

---

## Phase 4: Evaluation & Error Analysis

**Duration:** 2-3 days
**Objective:** Thoroughly evaluate the model and understand its failure modes.

### Tasks

1. **Comprehensive evaluation** -- `src/training/evaluate.py`
   - Metrics on the **held-out test set** (never seen during training):
     - Accuracy, Macro F1, Weighted F1
     - Per-class Precision, Recall, F1
     - Confusion matrix (3x3)
     - Classification report
   ```python
   from sklearn.metrics import classification_report, confusion_matrix

   report = classification_report(
       y_true, y_pred,
       target_names=["negative", "neutral", "positive"],
       output_dict=True,
   )

   cm = confusion_matrix(y_true, y_pred)
   # Visualize with seaborn heatmap
   sns.heatmap(cm, annot=True, fmt="d",
               xticklabels=["neg", "neu", "pos"],
               yticklabels=["neg", "neu", "pos"])
   ```
   - Generate all plots, save as artifacts in MLflow

2. **Error analysis** -- `notebooks/03_evaluation.ipynb`
   - Display the top 20 most confident **wrong** predictions per class:
     ```python
     # Find confident misclassifications
     wrong_mask = (y_pred != y_true)
     wrong_confidences = max_probs[wrong_mask]
     wrong_indices = np.where(wrong_mask)[0]

     # Sort by confidence (highest first)
     sorted_idx = wrong_confidences.argsort()[::-1][:20]
     for i in sorted_idx:
         idx = wrong_indices[i]
         print(f"True: {y_true[idx]}, Predicted: {y_pred[idx]}, "
               f"Confidence: {max_probs[idx]:.3f}")
         print(f"Text: {texts[idx][:200]}...")
         print("---")
     ```
   - Look for patterns in misclassifications:
     - Sarcastic reviews ("Great, just great. The worst meal ever.")
     - Mixed sentiment ("The food was amazing but the service was terrible.")
     - Very short reviews ("ok" -- is this neutral or negative?)
     - Reviews with unusual vocabulary or slang
   - Confusion matrix analysis:
     - Is neutral often confused with positive or negative?
     - Are negative reviews misclassified as positive (most dangerous error)?
   - Document findings in `MODEL_CARD.md`

3. **Text-specific evaluation**
   - Performance by review length (do short reviews perform worse?)
   - Performance by confidence threshold (precision-recall tradeoff)
   - Analyze attention weights for interpretability:
     ```python
     # Visualize which words the model attends to
     outputs = model(**inputs, output_attentions=True)
     attention = outputs.attentions[-1]  # Last layer attention
     # Average over heads
     avg_attention = attention.mean(dim=1)
     ```
   - Test on out-of-domain data (e.g., movie reviews if trained on restaurant reviews)

4. **Performance benchmarking**
   - Measure inference latency: single review on CPU (must be < 100ms)
   - Measure throughput: reviews per second (single and batched)
   - Measure model file size (DistilBERT ~260MB, check with head)
   - Memory footprint during inference
   - Tokenization time vs inference time breakdown
   ```python
   import time

   # Single review latency
   times = []
   for _ in range(100):
       start = time.perf_counter()
       _ = predict("This restaurant was absolutely fantastic!")
       times.append(time.perf_counter() - start)

   print(f"Mean latency: {np.mean(times)*1000:.1f}ms")
   print(f"P95 latency: {np.percentile(times, 95)*1000:.1f}ms")
   ```

5. **Model optimization for production** -- `src/model/export.py`
   - Export to **ONNX** format for faster CPU inference:
     ```python
     from transformers import convert_graph_to_onnx
     from pathlib import Path

     # Or manual export
     dummy_input = {
         "input_ids": torch.randint(0, 30522, (1, 256)),
         "attention_mask": torch.ones(1, 256, dtype=torch.long),
     }
     torch.onnx.export(
         model,
         (dummy_input["input_ids"], dummy_input["attention_mask"]),
         "model.onnx",
         input_names=["input_ids", "attention_mask"],
         output_names=["logits"],
         dynamic_axes={
             "input_ids": {0: "batch", 1: "sequence"},
             "attention_mask": {0: "batch", 1: "sequence"},
         },
     )
     ```
   - Benchmark: PyTorch vs ONNX Runtime latency (expect 2-3x speedup on CPU)
   - Optionally: quantize ONNX model to INT8

6. **Write `MODEL_CARD.md`**
   - What the model does: classifies text reviews into positive/negative/neutral
   - Training data description and preprocessing
   - Evaluation metrics with per-class breakdown
   - Known limitations:
     - Struggles with sarcasm
     - Mixed-sentiment reviews default to the dominant sentiment
     - Reviews shorter than 5 words have lower accuracy
     - English only
   - Intended use and misuse scenarios

### Skills Learned

- Per-class evaluation for multi-class classification
- Error analysis for NLP (understanding text misclassifications)
- Attention visualization for model interpretability
- ONNX export for transformer models
- Benchmarking NLP inference (tokenization + model time)

---

## Phase 5: API & Serving Layer

**Duration:** 3-4 days
**Objective:** Wrap the model in a production-quality REST API with text-specific features.

### Tasks

1. **Define API schemas** -- `src/serving/schemas.py`
   ```python
   from pydantic import BaseModel, Field

   class ReviewInput(BaseModel):
       text: str = Field(..., min_length=1, max_length=5000,
                         description="Review text to analyze")

   class BatchReviewInput(BaseModel):
       reviews: list[ReviewInput] = Field(..., max_length=32,
                                           description="Batch of reviews (max 32)")

   class SentimentPrediction(BaseModel):
       sentiment: str                     # "positive", "negative", "neutral"
       confidence: float                  # 0.0 to 1.0
       probabilities: dict[str, float]    # {"positive": 0.85, "neutral": 0.10, "negative": 0.05}
       model_version: str
       latency_ms: float
       token_count: int                   # Number of tokens in the input

   class BatchPredictionResponse(BaseModel):
       predictions: list[SentimentPrediction]
       total_latency_ms: float
       batch_size: int

   class HealthResponse(BaseModel):
       status: str
       model_loaded: bool
       model_version: str
       uptime_seconds: float
       tokenizer_loaded: bool
   ```

2. **Build FastAPI application** -- `src/serving/app.py`
   - `POST /predict` -- single review sentiment analysis
   - `POST /predict/batch` -- batch inference (up to 32 reviews)
   - `GET /health` -- health check
   - `GET /metrics` -- Prometheus metrics endpoint
   - Load model AND tokenizer once at startup:
     ```python
     from fastapi import FastAPI
     from contextlib import asynccontextmanager

     @asynccontextmanager
     async def lifespan(app: FastAPI):
         # Load model and tokenizer at startup
         app.state.model = load_model("models/sentiment_model.onnx")
         app.state.tokenizer = AutoTokenizer.from_pretrained("models/tokenizer/")
         yield
         # Cleanup on shutdown

     app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

     @app.post("/predict", response_model=SentimentPrediction)
     async def predict(review: ReviewInput):
         start = time.perf_counter()
         tokens = app.state.tokenizer(
             review.text, padding=True, truncation=True,
             max_length=256, return_tensors="np"
         )
         logits = app.state.model.run(None, dict(tokens))
         probs = softmax(logits[0])
         latency = (time.perf_counter() - start) * 1000

         return SentimentPrediction(
             sentiment=LABEL_MAP[probs.argmax()],
             confidence=float(probs.max()),
             probabilities={
                 "negative": float(probs[0][0]),
                 "neutral": float(probs[0][1]),
                 "positive": float(probs[0][2]),
             },
             model_version=MODEL_VERSION,
             latency_ms=latency,
             token_count=len(tokens["input_ids"][0]),
         )
     ```

3. **Inference logic** -- `src/serving/predict.py`
   - Load ONNX model with ONNX Runtime
   - Apply the same tokenization as training
   - Dynamic padding for batch inference (pad to longest in batch, not max_length)
   - Handle edge cases:
     - Empty string after cleaning
     - Text exceeding max_length (truncate gracefully)
     - Non-UTF8 characters
     - Extremely long inputs (reject > 5000 chars before tokenization)

4. **API middleware** -- `src/serving/middleware.py`
   - **Input validation:** text length limits, character encoding
   - **Rate limiting:** max 200 requests/minute per IP
   - **CORS:** allow frontend to call the API
   - **Request ID:** unique ID per request for tracing
   - **Error handling:** return clean JSON errors, never expose stack traces
   - **Request logging:** log input length (but NOT the input text -- privacy)

5. **Build Streamlit frontend** -- `src/frontend/app.py`
   - Text area for review input (with placeholder example)
   - Submit button
   - Display results:
     - Sentiment label with color coding (green/yellow/red)
     - Confidence bar chart for all three classes
     - Token count and latency
   - Batch mode: upload a CSV of reviews
   - Example reviews buttons for quick testing
   ```python
   import streamlit as st
   import requests

   st.title("Sentiment Analysis")
   text = st.text_area("Enter a review:", placeholder="The food was amazing...")

   if st.button("Analyze"):
       response = requests.post(f"{API_URL}/predict", json={"text": text})
       result = response.json()

       col1, col2, col3 = st.columns(3)
       col1.metric("Sentiment", result["sentiment"])
       col2.metric("Confidence", f"{result['confidence']:.1%}")
       col3.metric("Tokens", result["token_count"])

       # Probability bar chart
       st.bar_chart(result["probabilities"])
   ```

6. **API tests** -- `tests/`
   - Unit tests: schema validation, text cleaning, tokenization
   - Integration tests: full request -> response with sample reviews
   - Edge case tests: empty text, very long text, special characters, non-English text
   - Batch endpoint tests: various batch sizes, mixed lengths
   - Latency tests: verify p95 < 100ms for single reviews

### Skills Learned

- Building NLP-specific APIs (text input, tokenization at inference time)
- Dynamic batching for variable-length inputs
- Privacy-aware logging (log metadata, not content)
- Frontend for text-based ML applications
- Batch inference patterns for NLP

---

## Phase 6: Containerization

**Duration:** 2 days
**Objective:** Package everything into Docker containers.

### Tasks

1. **API Dockerfile** -- `docker/Dockerfile.api`
   - Multi-stage build with model and tokenizer baked in:
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
     COPY models/ /app/models/        # Includes model.onnx + tokenizer/
     EXPOSE 8000
     CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
     ```
   - NLP-specific note: the tokenizer files (vocab.txt, tokenizer_config.json) must be included alongside the ONNX model
   - Target: image size < 2GB (transformer models are larger than CV models)

2. **Frontend Dockerfile** -- `docker/Dockerfile.frontend`
   - Simpler single-stage build
   - Expose Streamlit port 8501

3. **Training Dockerfile** -- `docker/Dockerfile.training`
   - Based on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` for GPU support
   - Includes Hugging Face libraries
   - Mounts data volume

4. **docker-compose.yaml** -- orchestrate everything
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports: ["8000:8000"]
       environment:
         - MODEL_PATH=/app/models/model.onnx
         - TOKENIZER_PATH=/app/models/tokenizer/
         - LOG_LEVEL=info
         - MAX_BATCH_SIZE=32
         - MAX_SEQUENCE_LENGTH=256

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
   - `docker compose up` -- everything starts and connects
   - Frontend can submit reviews, API returns predictions
   - Prometheus scrapes metrics, Grafana shows dashboard

### Skills Learned

- Docker multi-stage builds for NLP models (larger images)
- Packaging tokenizer + model together
- Environment variable configuration for NLP serving
- Container networking for multi-service NLP applications

---

## Phase 7: Testing & CI/CD

**Duration:** 2-3 days
**Objective:** Automate quality checks so nothing broken reaches production.

### Tasks

1. **Write comprehensive tests**

   **Unit tests** (fast, no heavy model loading):
   ```
   test_preprocessing.py
   ├── test_clean_text_removes_html
   ├── test_clean_text_removes_urls
   ├── test_clean_text_preserves_punctuation
   ├── test_clean_text_handles_empty_string
   └── test_label_mapping_is_correct

   test_tokenization.py
   ├── test_tokenizer_output_has_input_ids_and_attention_mask
   ├── test_tokenizer_respects_max_length
   ├── test_tokenizer_handles_special_characters
   └── test_tokenizer_padding_is_correct

   test_model.py
   ├── test_model_output_shape_is_batch_x_3
   ├── test_model_output_sums_to_1_after_softmax
   ├── test_model_loads_from_checkpoint
   └── test_onnx_export_matches_pytorch_output

   test_schemas.py
   ├── test_review_input_rejects_empty_text
   ├── test_review_input_rejects_too_long_text
   ├── test_prediction_response_validates_confidence_range
   └── test_batch_input_respects_max_size
   ```

   **Integration tests** (slower, test the full pipeline):
   ```
   test_api.py
   ├── test_predict_returns_200_on_valid_text
   ├── test_predict_returns_422_on_empty_text
   ├── test_predict_returns_correct_schema
   ├── test_batch_predict_multiple_reviews
   ├── test_health_endpoint
   └── test_concurrent_requests

   test_training.py
   ├── test_training_loop_runs_one_epoch_on_tiny_data
   └── test_training_saves_checkpoint
   ```

   **NLP-specific tests**:
   ```
   ├── test_tokenizer_matches_between_training_and_serving
   ├── test_model_is_deterministic_with_same_input
   ├── test_known_positive_review_is_classified_positive
   ├── test_known_negative_review_is_classified_negative
   └── test_model_handles_very_long_input_gracefully
   ```

2. **Set up CI pipeline** -- `.github/workflows/ci.yaml`
   ```yaml
   name: CI
   on: [pull_request]
   jobs:
     lint:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - run: pip install ruff mypy
         - run: ruff check .
         - run: mypy src/
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - run: pip install -e ".[dev]"
         - run: pytest tests/unit/ -v
         - run: pytest tests/integration/ -v
     build:
       runs-on: ubuntu-latest
       steps:
         - run: docker build -f docker/Dockerfile.api .
   ```

3. **Set up CD pipeline** -- `.github/workflows/cd.yaml`

4. **Pre-commit hooks**
   - Ruff (linting + formatting)
   - mypy (type checking)
   - Check no large model files committed accidentally (> 100MB)

### Skills Learned

- Testing NLP pipelines (tokenization consistency is critical)
- Behavioral tests for ML models (known inputs -> expected outputs)
- CI/CD pipeline design for NLP projects
- Handling large model files in CI (download or mock)

---

## Phase 8: Deployment

**Duration:** 2-3 days
**Objective:** Get the application running on the internet.

### Tasks

1. **Choose a platform** (same options as Project 1)

   | Platform | Pros | Cons | Cost |
   |----------|------|------|------|
   | **Railway** | Simplest deployment | Less control, memory limits | Free tier available |
   | **GCP Cloud Run** | Auto-scaling, pay-per-use | More setup | Free tier generous |
   | **AWS ECS/Fargate** | Enterprise standard | Most complex | Free tier limited |

   **NLP-specific consideration:** transformer models need more memory (~1GB RAM minimum). Ensure your platform supports this.

2. **Prepare for deployment**
   - Ensure all config comes from environment variables
   - Health check endpoint works
   - Model and tokenizer files baked into Docker image
   - Set appropriate memory limits (at least 1GB for DistilBERT)

3. **Deploy the API and frontend**

4. **Load testing**
   - Use `locust` to simulate traffic
   - Target: 50 concurrent users, p95 latency < 200ms
   - Test with varying review lengths (short, medium, long)
   - Identify bottleneck: tokenization or model inference?

5. **Set up basic infrastructure**
   - HTTPS, custom domain (optional)
   - Environment-specific configs (staging vs production)

### Skills Learned

- Deploying NLP models (memory requirements, larger images)
- Load testing text-based APIs
- Performance tuning for transformer inference

---

## Phase 9: Monitoring & Observability

**Duration:** 2-3 days
**Objective:** Know what your system is doing in production, with text-specific monitoring.

### Tasks

1. **Structured logging** -- `src/serving/`
   - Every prediction logged as structured JSON:
     ```json
     {
       "timestamp": "2026-03-22T10:30:00Z",
       "request_id": "abc-123",
       "predicted_sentiment": "positive",
       "confidence": 0.92,
       "latency_ms": 67,
       "model_version": "v1.0",
       "input_length_chars": 342,
       "input_length_tokens": 78,
       "status": "success"
     }
     ```
   - **Privacy:** never log the review text itself -- only metadata (length, token count)
   - Log errors with full context

2. **Prometheus metrics** -- `src/monitoring/metrics.py`
   - `prediction_requests_total` -- counter by sentiment class and status
   - `prediction_latency_seconds` -- histogram
   - `prediction_confidence` -- histogram (track confidence distribution)
   - `input_length_tokens` -- histogram (track input length distribution)
   - `input_length_chars` -- histogram
   - `tokenization_latency_seconds` -- histogram (tokenization vs inference split)
   - `model_info` -- gauge with version label
   - `active_requests` -- gauge

3. **Grafana dashboard** -- `grafana/dashboards/nlp_monitoring.json`
   - Row 1: Request rate, error rate, latency percentiles (p50, p95, p99)
   - Row 2: Sentiment class distribution over time, confidence distribution
   - Row 3: Input length distribution (are users sending longer texts?)
   - Row 4: Tokenization time vs inference time breakdown
   - Row 5: System metrics (CPU, memory, active requests)

4. **NLP-specific monitoring** -- `src/monitoring/drift.py`
   - **Class distribution drift:** if sentiment predictions suddenly skew heavily positive or negative, something may have changed
   - **Confidence drift:** if average confidence drops, the model may be seeing out-of-distribution text
   - **Input length drift:** if average input length changes significantly, the user population or use case may have shifted
   - **Token distribution monitoring:** track the frequency of [UNK] tokens -- a spike means the model is seeing unfamiliar vocabulary
   ```python
   class TextDriftDetector:
       def __init__(self, window_size=1000):
           self.recent_lengths = deque(maxlen=window_size)
           self.recent_confidences = deque(maxlen=window_size)
           self.recent_sentiments = deque(maxlen=window_size)
           self.baseline_stats = None

       def update(self, token_count, confidence, sentiment):
           self.recent_lengths.append(token_count)
           self.recent_confidences.append(confidence)
           self.recent_sentiments.append(sentiment)

       def check_drift(self):
           """Compare recent window to baseline using statistical tests."""
           if self.baseline_stats is None:
               return {"drift_detected": False}
           # KS test for length distribution
           # Chi-squared test for sentiment distribution
           ...
   ```

5. **Alerting rules**
   - Error rate > 5% for 5 minutes -> alert
   - p95 latency > 300ms for 5 minutes -> alert
   - Sentiment distribution shift > 2 standard deviations -> alert
   - Average confidence drop > 10% vs baseline -> alert
   - [UNK] token rate > 5% -> alert (vocabulary drift)

### Skills Learned

- NLP-specific monitoring (input length, vocabulary drift, class distribution)
- Privacy-preserving logging for text data
- Statistical drift detection for categorical outputs
- Building NLP-specific Grafana dashboards

---

## Timeline Summary

```
Week 1  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 1: Setup              (2 days)
        Phase 2: Data Pipeline      (3 days)

Week 2  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 2: Data (cont.)       (2 days)
        Phase 3: Model Development  (3 days)

Week 3  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 3: Model (cont.)      (2 days)
        Phase 4: Evaluation         (3 days)

Week 4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 5: API & Serving      (4 days)
        Phase 6: Docker             (1 day)

Week 5  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 6: Docker (cont.)     (1 day)
        Phase 7: CI/CD              (3 days)
        Phase 8: Deployment         (1 day)

Week 6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 8: Deployment (cont.) (2 days)
        Phase 9: Monitoring         (3 days)
```

**Total: ~33 days (6-7 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] Hugging Face Transformers library (loading, fine-tuning, saving models)
- [ ] Hugging Face Datasets library (loading, mapping, filtering)
- [ ] Subword tokenization (BPE/WordPiece) and its implications
- [ ] Transfer learning for NLP (fine-tuning pretrained language models)
- [ ] Hugging Face Trainer API for production training
- [ ] Mixed precision training (fp16) for GPU efficiency
- [ ] Handling class imbalance with weighted loss functions
- [ ] Text preprocessing pipelines (cleaning, normalizing)
- [ ] EDA for text data (length distributions, word frequencies)
- [ ] Per-class evaluation metrics for multi-class classification
- [ ] Error analysis for NLP (sarcasm, ambiguity, mixed sentiment)
- [ ] Attention visualization for model interpretability
- [ ] ONNX export for transformer models
- [ ] Building text-input REST APIs with FastAPI
- [ ] Dynamic batching for variable-length inputs
- [ ] Privacy-aware logging (metadata only)
- [ ] Streamlit frontend for NLP applications
- [ ] Docker containerization for NLP models (larger images)
- [ ] Testing NLP pipelines (tokenization consistency)
- [ ] CI/CD with GitHub Actions
- [ ] Cloud deployment with memory-aware configuration
- [ ] NLP-specific monitoring (vocabulary drift, input length drift)
- [ ] Prometheus metrics instrumentation
- [ ] Grafana dashboard creation for NLP
- [ ] Data versioning with DVC

---

## Key Differences from Project 1 (Cat vs Dog Classifier)

| Aspect | Project 1 (CV) | Project 2 (NLP) |
|--------|----------------|-----------------|
| Input type | Fixed-size images (224x224) | Variable-length text |
| Preprocessing | Resize, normalize, augment | Tokenize (subword), pad/truncate |
| Model | MobileNetV2 (3.4M params) | DistilBERT (66M params) |
| Framework | PyTorch + torchvision | Hugging Face Transformers |
| Training | Custom loop | Hugging Face Trainer |
| GPU need | Nice to have | Strongly recommended |
| Inference size | ~15MB (ONNX) | ~260MB (ONNX) |
| Batch handling | Fixed-size tensors | Dynamic padding needed |
| Monitoring | Image size, pixel distribution | Text length, vocabulary drift |
| Privacy | Don't log images | Don't log review text |

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Create the project directory
mkdir sentiment-analysis && cd sentiment-analysis
git init

# 2. Create the folder structure
mkdir -p configs data/{raw,processed,splits} notebooks \
  src/{data,model,training,serving,monitoring,frontend} \
  tests/{unit,integration} docker .github/workflows \
  grafana/dashboards prometheus scripts

# 3. Verify Hugging Face is working
python -c "from transformers import AutoModel; print('Transformers OK')"
python -c "from datasets import load_dataset; print('Datasets OK')"

# 4. Start writing DESIGN_DOC.md
```
