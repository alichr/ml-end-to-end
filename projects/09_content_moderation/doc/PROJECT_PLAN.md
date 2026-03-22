# Project 9: Multi-Modal Content Moderation (Expert)

## Goal

Build a **production-grade content moderation system** that analyzes social media posts
containing both text and images, detects harmful content across multiple categories
(hate speech, violence, NSFW, harassment), and routes decisions through a configurable
policy engine with human review workflows. The system processes content asynchronously
at scale using a distributed task queue.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why content moderation? | Every platform with user-generated content needs moderation. It is a multi-billion dollar problem combining ML, policy, and systems engineering. |
| Why multi-modal? | Real harmful content often relies on the combination of text and image (e.g., hateful memes). Single-modality models miss these cases. |
| Will this look good on a portfolio? | Extremely. This project demonstrates multi-model orchestration, async processing, bias auditing, and human-in-the-loop systems -- all highly valued in industry. |
| What makes this "Expert"? | You must coordinate multiple ML models, handle bias and fairness, build policy systems, implement async pipelines, and design human review workflows. |

---

## Architecture Overview

```
                     ┌──────────────────────┐
                     │   Content Submission  │
                     │   (API Endpoint)      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │     RabbitMQ          │
                     │   Message Queue       │
                     │  (priority routing)   │
                     └──┬────────┬────────┬─┘
                        │        │        │
              ┌─────────▼──┐ ┌──▼──────┐ ┌▼───────────┐
              │ Text Worker │ │ Image   │ │  Fusion    │
              │ (BERT       │ │ Worker  │ │  Worker    │
              │  toxicity)  │ │ (ViT/   │ │  (CLIP     │
              │             │ │  CLIP)  │ │  combined) │
              └──────┬──────┘ └────┬────┘ └─────┬──────┘
                     │             │             │
                     └─────────────┼─────────────┘
                                   ▼
                     ┌──────────────────────┐
                     │    Policy Engine      │
                     │  (rules + ML scores)  │
                     │  severity → decision  │
                     └──────────┬───────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌────────┐ ┌──────────┐
              │  Auto     │ │ Flag   │ │  Auto    │
              │  Approve  │ │ for    │ │  Remove  │
              │           │ │ Review │ │  + Ban   │
              └──────────┘ └───┬────┘ └──────────┘
                               ▼
                     ┌──────────────────────┐
                     │  Human Review        │
                     │  Dashboard           │
                     │  (appeal workflow)   │
                     └──────────────────────┘

     ┌──────────────┐  ┌──────────┐  ┌──────────┐
     │  PostgreSQL   │  │  Redis   │  │ Grafana  │
     │  (decisions,  │  │ (cache,  │  │(monitor) │
     │   appeals)    │  │  counts) │  │          │
     └──────────────┘  └──────────┘  └──────────┘

Everything runs in Docker. Models run on CPU/GPU workers.
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | ML ecosystem standard |
| Text Model | BERT (fine-tuned for toxicity) | Proven text classification architecture |
| Image Model | ViT / CLIP | State-of-the-art image understanding |
| Fusion Model | CLIP + custom head | Best multi-modal alignment model available |
| Task Queue | Celery + RabbitMQ | Industry standard distributed task processing |
| Database | PostgreSQL | Relational data: decisions, appeals, user history |
| Cache | Redis | Fast lookups: rate limits, duplicate detection, model cache |
| API Framework | FastAPI | Async support, background tasks, auto-docs |
| Frontend | Streamlit / React | Review dashboard for human moderators |
| Containerization | Docker + docker-compose | Consistent environments, scalable workers |
| CI/CD | GitHub Actions | Free, integrated with GitHub |
| Monitoring | Prometheus + Grafana | Volume, latency, bias tracking |
| ML Framework | Hugging Face Transformers + PyTorch | Pre-trained models, easy fine-tuning |
| Bias Auditing | Fairlearn / custom | Demographic parity, equalized odds |

---

## Project Structure

```
content-moderation/
│
├── doc/
│   ├── DESIGN_DOC.md              # Moderation policies, severity definitions
│   ├── PROJECT_PLAN.md            # This file
│   ├── ANNOTATION_GUIDELINES.md   # How to label training data
│   └── BIAS_AUDIT_REPORT.md       # Fairness analysis results
│
├── pyproject.toml                 # Dependencies and project metadata
├── README.md                      # Setup instructions, architecture
│
├── configs/
│   ├── moderation_config.yaml     # Policy thresholds, severity levels
│   ├── model_config.yaml          # Model paths, batch sizes, devices
│   ├── celery_config.yaml         # Queue settings, concurrency, retries
│   └── serve_config.yaml          # API settings, rate limits
│
├── data/                          # Git-ignored
│   ├── raw/                       # Original datasets
│   ├── processed/                 # Preprocessed text + images
│   ├── annotations/               # Human labels
│   └── evaluation/                # Bias evaluation datasets
│
├── notebooks/
│   ├── 01_eda_hateful_memes.ipynb     # Dataset exploration
│   ├── 02_text_model_training.ipynb   # Text classifier experiments
│   ├── 03_image_model_training.ipynb  # Image classifier experiments
│   ├── 04_fusion_experiments.ipynb    # Multi-modal fusion experiments
│   └── 05_bias_audit.ipynb            # Fairness analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Multi-modal PyTorch Dataset
│   │   ├── text_preprocessing.py  # Text cleaning, tokenization
│   │   ├── image_preprocessing.py # Image transforms, augmentation
│   │   └── download.py            # Dataset download scripts
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── text_classifier.py     # BERT toxicity classifier
│   │   ├── image_classifier.py    # ViT/CLIP image classifier
│   │   ├── fusion_model.py        # Multi-modal fusion model
│   │   └── export.py              # ONNX export for serving
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_text.py          # Text model training loop
│   │   ├── train_image.py         # Image model training loop
│   │   ├── train_fusion.py        # Fusion model training loop
│   │   └── evaluate.py            # Per-model and combined evaluation
│   │
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── engine.py              # Policy decision engine
│   │   ├── rules.py               # Rule-based filters (blocklists, regex)
│   │   ├── severity.py            # Severity scoring logic
│   │   └── actions.py             # Decision actions (approve/flag/remove/ban)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── celery_app.py          # Celery application configuration
│   │   ├── tasks.py               # Celery task definitions
│   │   ├── priority.py            # Priority queue routing
│   │   └── dedup.py               # Duplicate content detection
│   │
│   ├── bias/
│   │   ├── __init__.py
│   │   ├── auditor.py             # Bias audit framework
│   │   ├── metrics.py             # Fairness metrics implementation
│   │   └── report.py              # Generate bias audit reports
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application
│   │   ├── schemas.py             # Request/response models
│   │   ├── middleware.py          # Rate limiting, auth, logging
│   │   └── review_api.py         # Human review endpoints
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Prometheus metrics
│   │   ├── bias_monitor.py        # Real-time bias tracking
│   │   └── queue_monitor.py       # Queue depth and throughput
│   │
│   └── frontend/
│       ├── review_dashboard.py    # Human review interface
│       └── admin_dashboard.py     # Admin monitoring dashboard
│
├── tests/
│   ├── unit/
│   │   ├── test_text_preprocessing.py
│   │   ├── test_image_preprocessing.py
│   │   ├── test_policy_engine.py
│   │   ├── test_severity_scoring.py
│   │   └── test_schemas.py
│   ├── integration/
│   │   ├── test_text_model.py
│   │   ├── test_image_model.py
│   │   ├── test_fusion_model.py
│   │   ├── test_celery_pipeline.py
│   │   ├── test_api.py
│   │   └── test_review_workflow.py
│   ├── bias/
│   │   ├── test_demographic_parity.py
│   │   └── test_equalized_odds.py
│   └── conftest.py
│
├── docker/
│   ├── Dockerfile.api             # API server
│   ├── Dockerfile.worker          # Celery worker with models
│   ├── Dockerfile.frontend        # Review dashboard
│   └── Dockerfile.training        # Training environment (GPU)
│
├── docker-compose.yaml
│
├── .github/
│   └── workflows/
│       ├── ci.yaml
│       └── cd.yaml
│
├── grafana/
│   └── dashboards/
│       └── moderation_monitoring.json
│
├── prometheus/
│   └── prometheus.yml
│
└── scripts/
    ├── setup.sh
    ├── train_all.sh               # Train text → image → fusion
    ├── evaluate_bias.sh           # Run full bias audit
    └── seed_review_queue.sh       # Populate review queue for testing
```

---

## Phase 1: Setup & Design Doc

**Duration:** 2-3 days
**Objective:** Define moderation policies, severity levels, and system boundaries.

### Tasks

1. **Write `DESIGN_DOC.md`**
   - **Problem statement:** "Given a social media post (text + optional image), classify
     whether it violates content policies and determine the appropriate action"
   - **Content categories to detect:**
     - Hate speech / discrimination
     - Violence / graphic content
     - NSFW / sexual content
     - Harassment / bullying
     - Spam / scams
   - **Severity levels:**
     - `LOW` -- borderline content, auto-approve with logging
     - `MEDIUM` -- likely violation, flag for human review
     - `HIGH` -- clear violation, auto-remove
     - `CRITICAL` -- illegal content, auto-remove + escalate
   - **Success criteria:**
     - Per-category precision >= 85% (minimize false positives that silence legitimate speech)
     - Per-category recall >= 90% (minimize harmful content that slips through)
     - False positive rate < 5% across all demographics
     - Processing latency < 2 seconds per post (async pipeline)
     - Queue throughput: 1000 posts/minute
   - **Out of scope:** video moderation, audio, real-time chat, non-English languages
   - **Ethical considerations:** bias across demographics, appeal process, transparency

2. **Write `ANNOTATION_GUIDELINES.md`**
   - Clear definitions for each content category with examples
   - Edge cases and how to handle them
   - Severity assignment rubric
   - Inter-annotator agreement targets (Cohen's kappa >= 0.7)

3. **Initialize the repository**
   - Dependencies in `pyproject.toml`:
     ```toml
     [project]
     name = "content-moderation"
     dependencies = [
         "torch>=2.1.0",
         "transformers>=4.36.0",
         "clip-interrogator>=0.6.0",
         "open-clip-torch>=2.24.0",
         "celery[rabbitmq]>=5.3.0",
         "psycopg2-binary>=2.9.0",
         "sqlalchemy>=2.0.0",
         "redis>=5.0.0",
         "fastapi>=0.109.0",
         "uvicorn>=0.27.0",
         "streamlit>=1.30.0",
         "Pillow>=10.0.0",
         "prometheus-client>=0.20.0",
         "fairlearn>=0.9.0",
         "pydantic>=2.5.0",
         "python-multipart>=0.0.6",
     ]
     ```

4. **Set up development environment**
   - Python virtual environment
   - Docker Compose for RabbitMQ, PostgreSQL, Redis (development dependencies)
   - Pre-commit hooks: ruff, mypy

### Skills Learned

- Designing content moderation policies (a real-world requirement)
- Defining severity levels and action matrices
- Writing annotation guidelines for ML datasets
- Ethical considerations in ML system design

---

## Phase 2: Data Pipeline

**Duration:** 4-5 days
**Objective:** Prepare multi-modal dataset with text and image pairs.

### Tasks

1. **Download and explore dataset** -- `src/data/download.py`
   - Primary: Facebook Hateful Memes dataset (10,000+ text+image pairs)
   - Supplementary: Jigsaw Toxic Comment dataset (text only), NSFW image datasets
   - Understand label distribution, annotation quality, known biases
   ```python
   def download_hateful_memes(data_dir: Path):
       """Download and extract the Hateful Memes dataset."""
       # Requires accepting license agreement
       # Dataset structure: image file + JSONL with text and labels
       pass
   ```

2. **EDA notebook** -- `notebooks/01_eda_hateful_memes.ipynb`
   - Class distribution: hateful vs benign (expect heavy imbalance)
   - Text length distribution
   - Image resolution distribution
   - Sample harmful vs benign examples (be cautious with display)
   - Identify confounders: memes that are only hateful in text+image combination

3. **Text preprocessing** -- `src/data/text_preprocessing.py`
   - Lowercasing and normalization
   - Handle social media conventions: emojis, hashtags, mentions, URLs
   - Profanity normalization (handle obfuscation: "h@te", "sh1t")
   - BERT tokenization with special tokens
   ```python
   class TextPreprocessor:
       def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
           self.tokenizer = AutoTokenizer.from_pretrained(model_name)
           self.max_length = max_length

       def preprocess(self, text: str) -> dict:
           text = self._normalize_social_media(text)
           text = self._normalize_profanity(text)
           return self.tokenizer(
               text,
               padding="max_length",
               truncation=True,
               max_length=self.max_length,
               return_tensors="pt",
           )

       def _normalize_social_media(self, text: str) -> str:
           text = re.sub(r"http\S+", "[URL]", text)
           text = re.sub(r"@\w+", "[USER]", text)
           text = re.sub(r"#(\w+)", r"\1", text)
           return text
   ```

4. **Image preprocessing** -- `src/data/image_preprocessing.py`
   - Resize to model input size (224x224 for ViT, 336x336 for CLIP)
   - Normalization with ImageNet statistics
   - Training augmentation: random crop, horizontal flip, color jitter
   - Handle corrupted images, unusual aspect ratios, transparency
   ```python
   class ImagePreprocessor:
       def __init__(self, image_size: int = 224):
           self.train_transform = Compose([
               Resize(image_size + 32),
               RandomCrop(image_size),
               RandomHorizontalFlip(),
               ColorJitter(brightness=0.1, contrast=0.1),
               ToTensor(),
               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
           ])
           self.eval_transform = Compose([
               Resize(image_size + 32),
               CenterCrop(image_size),
               ToTensor(),
               Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
           ])
   ```

5. **Multi-modal dataset** -- `src/data/dataset.py`
   - PyTorch Dataset that returns (text_tokens, image_tensor, label) tuples
   - Handle missing modalities: some posts are text-only, some image-only
   - Stratified train/val/test split (70/15/15) preserving class distribution
   - Weighted sampling for class imbalance
   ```python
   class MultiModalDataset(Dataset):
       def __init__(self, annotations: list[dict], text_preprocessor, image_preprocessor):
           self.annotations = annotations
           self.text_pp = text_preprocessor
           self.image_pp = image_preprocessor

       def __getitem__(self, idx):
           item = self.annotations[idx]
           text_features = self.text_pp.preprocess(item["text"])
           image = Image.open(item["image_path"]).convert("RGB")
           image_features = self.image_pp(image)
           label = item["label"]
           return {
               "text": text_features,
               "image": image_features,
               "label": torch.tensor(label, dtype=torch.long),
               "has_image": torch.tensor(1.0),
           }
   ```

### Skills Learned

- Working with multi-modal datasets
- Social media text preprocessing (a specialized skill)
- Handling class imbalance in moderation datasets
- Building PyTorch Datasets for multi-modal inputs
- Writing annotation guidelines

---

## Phase 3: Individual Models

**Duration:** 5-6 days
**Objective:** Train separate text and image classifiers before building the fusion model.

### Tasks

1. **Text toxicity classifier** -- `src/models/text_classifier.py`
   - Fine-tune `bert-base-uncased` on toxicity detection
   - Multi-label classification head (a post can be hateful AND violent)
   - Freeze lower BERT layers, fine-tune top 4 layers + classification head
   ```python
   class TextToxicityClassifier(nn.Module):
       def __init__(
           self, model_name: str = "bert-base-uncased",
           num_labels: int = 5, freeze_layers: int = 8,
       ):
           super().__init__()
           self.bert = AutoModel.from_pretrained(model_name)
           # Freeze lower layers
           for layer in self.bert.encoder.layer[:freeze_layers]:
               for param in layer.parameters():
                   param.requires_grad = False
           self.dropout = nn.Dropout(0.3)
           self.classifier = nn.Sequential(
               nn.Linear(768, 256),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(256, num_labels),
           )

       def forward(self, input_ids, attention_mask):
           outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
           pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
           return self.classifier(self.dropout(pooled))
   ```

2. **Train text model** -- `src/training/train_text.py`
   - Binary cross-entropy loss (multi-label)
   - AdamW optimizer with linear warmup
   - Evaluate per-label: precision, recall, F1
   - Track in MLflow: per-epoch metrics, confusion matrices
   ```python
   def train_text_model(config):
       model = TextToxicityClassifier(num_labels=config.num_labels)
       optimizer = AdamW(
           model.parameters(), lr=config.learning_rate, weight_decay=0.01
       )
       scheduler = get_linear_schedule_with_warmup(
           optimizer,
           num_warmup_steps=config.warmup_steps,
           num_training_steps=config.total_steps,
       )
       criterion = nn.BCEWithLogitsLoss(
           pos_weight=torch.tensor(config.class_weights)  # Handle imbalance
       )
       # Training loop with validation...
   ```

3. **Image classifier** -- `src/models/image_classifier.py`
   - Fine-tune ViT-Base or use CLIP image encoder
   - Classification head: NSFW, violence, graphic content
   - Apply strong augmentation to prevent overfitting on small dataset
   ```python
   class ImageContentClassifier(nn.Module):
       def __init__(
           self, model_name: str = "google/vit-base-patch16-224",
           num_labels: int = 3,
       ):
           super().__init__()
           self.vit = ViTModel.from_pretrained(model_name)
           self.classifier = nn.Sequential(
               nn.Linear(768, 256),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(256, num_labels),
           )

       def forward(self, pixel_values):
           outputs = self.vit(pixel_values=pixel_values)
           cls_output = outputs.last_hidden_state[:, 0, :]
           return self.classifier(cls_output)
   ```

4. **Train image model** -- `src/training/train_image.py`
   - Similar training loop to text model
   - Heavier augmentation (images are more scarce)
   - Per-category metrics: NSFW precision/recall, violence precision/recall

5. **Per-model evaluation** -- `src/training/evaluate.py`
   - Confusion matrix per label
   - Precision-recall curves (critical for setting thresholds)
   - ROC-AUC per label
   - Analyze failure modes: what does each model miss?
   - Document performance in experiment tracking

6. **Experiment notebooks** -- `notebooks/02_*.ipynb`, `notebooks/03_*.ipynb`
   - Text model: compare BERT vs RoBERTa vs DistilBERT
   - Image model: compare ViT vs CLIP image encoder vs ResNet
   - Document trade-offs: accuracy vs latency vs model size

### Skills Learned

- Fine-tuning BERT for text classification
- Fine-tuning ViT for image classification
- Multi-label classification (different from multi-class)
- Handling class imbalance with weighted losses
- Per-model evaluation and failure analysis

---

## Phase 4: Multi-Modal Fusion

**Duration:** 5-6 days
**Objective:** Combine text and image models into a single multi-modal classifier.

### Tasks

1. **Understand fusion strategies**
   - **Early fusion:** concatenate raw features before any model processing
   - **Late fusion:** run models independently, combine predictions
   - **Cross-attention fusion:** let text and image attend to each other
   - **CLIP-based fusion:** use CLIP's pre-trained multi-modal alignment

2. **Implement late fusion baseline** -- `src/models/fusion_model.py`
   - Run text and image models independently
   - Combine scores with learned weights
   - Simple but effective baseline
   ```python
   class LateFusionClassifier(nn.Module):
       """Combine text and image model outputs with learned weights."""

       def __init__(self, text_model, image_model, num_labels: int = 5):
           super().__init__()
           self.text_model = text_model
           self.image_model = image_model
           self.combiner = nn.Sequential(
               nn.Linear(num_labels * 2, 128),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(128, num_labels),
           )

       def forward(self, text_inputs, image_inputs, has_image):
           text_scores = self.text_model(**text_inputs)
           image_scores = self.image_model(image_inputs)
           # Mask image scores if post has no image
           image_scores = image_scores * has_image.unsqueeze(-1)
           combined = torch.cat([text_scores, image_scores], dim=-1)
           return self.combiner(combined)
   ```

3. **Implement CLIP-based fusion** -- `src/models/fusion_model.py`
   - Use CLIP to embed text and image into shared space
   - Compute cross-modal similarity as a feature
   - Add classification head on top of CLIP features
   ```python
   class CLIPFusionClassifier(nn.Module):
       """Use CLIP embeddings for multi-modal classification."""

       def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32",
                    num_labels: int = 5):
           super().__init__()
           self.clip = CLIPModel.from_pretrained(clip_model_name)
           self.processor = CLIPProcessor.from_pretrained(clip_model_name)
           # CLIP features: text (512) + image (512) + similarity (1)
           self.classifier = nn.Sequential(
               nn.Linear(512 + 512 + 1, 256),
               nn.ReLU(),
               nn.Dropout(0.3),
               nn.Linear(256, 64),
               nn.ReLU(),
               nn.Linear(64, num_labels),
           )

       def forward(self, input_ids, pixel_values, attention_mask):
           outputs = self.clip(
               input_ids=input_ids,
               pixel_values=pixel_values,
               attention_mask=attention_mask,
           )
           text_embeds = outputs.text_embeds        # (batch, 512)
           image_embeds = outputs.image_embeds      # (batch, 512)
           similarity = (text_embeds * image_embeds).sum(dim=-1, keepdim=True)
           combined = torch.cat([text_embeds, image_embeds, similarity], dim=-1)
           return self.classifier(combined)
   ```

4. **Implement cross-attention fusion** (advanced)
   - Text features attend to image patches
   - Image features attend to text tokens
   - Most powerful but most compute-intensive
   ```python
   class CrossAttentionFusion(nn.Module):
       def __init__(self, embed_dim: int = 768, num_heads: int = 8):
           super().__init__()
           self.text_to_image_attn = nn.MultiheadAttention(embed_dim, num_heads)
           self.image_to_text_attn = nn.MultiheadAttention(embed_dim, num_heads)
           self.fusion_norm = nn.LayerNorm(embed_dim * 2)
           self.classifier = nn.Linear(embed_dim * 2, 5)

       def forward(self, text_features, image_features):
           # Text attends to image
           t2i, _ = self.text_to_image_attn(
               text_features, image_features, image_features
           )
           # Image attends to text
           i2t, _ = self.image_to_text_attn(
               image_features, text_features, text_features
           )
           # Pool and combine
           t2i_pooled = t2i.mean(dim=0)
           i2t_pooled = i2t.mean(dim=0)
           combined = self.fusion_norm(torch.cat([t2i_pooled, i2t_pooled], dim=-1))
           return self.classifier(combined)
   ```

5. **Train and compare fusion models** -- `src/training/train_fusion.py`
   - Train late fusion, CLIP-based, and cross-attention approaches
   - Compare: combined accuracy, per-category performance, latency
   - Key question: does fusion beat individual models? By how much?
   - Focus on cases where text-only or image-only models fail

6. **Fusion experiments notebook** -- `notebooks/04_fusion_experiments.ipynb`
   - Side-by-side comparison of fusion strategies
   - Analyze: which fusion catches "hateful memes" that single-modality misses?
   - Latency comparison: late fusion is fastest, cross-attention is slowest
   - Select best approach based on accuracy-latency trade-off

### Skills Learned

- Multi-modal fusion architectures (a cutting-edge skill)
- CLIP model usage and fine-tuning
- Cross-attention mechanisms
- Comparing model architectures systematically
- Trade-off analysis: accuracy vs latency

---

## Phase 5: Policy Engine

**Duration:** 3-4 days
**Objective:** Build a configurable policy system that combines ML scores with rules.

### Tasks

1. **Rule-based filters** -- `src/policy/rules.py`
   - Blocklist: known harmful terms, URLs, patterns
   - Regex patterns: phone numbers, email addresses (PII detection)
   - Rate limiting: flag users posting too frequently
   - Duplicate detection: flag identical content posted multiple times
   ```python
   class RuleBasedFilter:
       def __init__(self, config_path: str):
           self.config = load_config(config_path)
           self.blocklist = set(self.config["blocklist_terms"])
           self.patterns = [re.compile(p) for p in self.config["patterns"]]

       def check(self, text: str) -> list[RuleViolation]:
           violations = []
           text_lower = text.lower()
           for term in self.blocklist:
               if term in text_lower:
                   violations.append(RuleViolation(
                       rule="blocklist",
                       term=term,
                       severity="HIGH",
                   ))
           for pattern in self.patterns:
               if pattern.search(text):
                   violations.append(RuleViolation(
                       rule="pattern_match",
                       pattern=pattern.pattern,
                       severity="MEDIUM",
                   ))
           return violations
   ```

2. **Severity scoring** -- `src/policy/severity.py`
   - Combine ML model scores with rule violations
   - Weighted scoring: text_score * 0.4 + image_score * 0.3 + fusion_score * 0.3
   - Configurable thresholds per category
   - Override: rule-based violations can force minimum severity
   ```python
   class SeverityScorer:
       def __init__(self, config: dict):
           self.thresholds = config["thresholds"]
           # Example: {"hate_speech": {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.85}}

       def score(self, ml_scores: dict, rule_violations: list) -> SeverityResult:
           max_severity = "NONE"
           category_results = {}

           for category, score in ml_scores.items():
               thresholds = self.thresholds[category]
               if score >= thresholds["HIGH"]:
                   severity = "HIGH"
               elif score >= thresholds["MEDIUM"]:
                   severity = "MEDIUM"
               elif score >= thresholds["LOW"]:
                   severity = "LOW"
               else:
                   severity = "NONE"
               category_results[category] = CategoryResult(
                   score=score, severity=severity
               )
               max_severity = max(max_severity, severity, key=SEVERITY_ORDER.index)

           # Rule violations can escalate severity
           for violation in rule_violations:
               max_severity = max(
                   max_severity, violation.severity, key=SEVERITY_ORDER.index
               )

           return SeverityResult(
               overall_severity=max_severity,
               category_results=category_results,
               rule_violations=rule_violations,
           )
   ```

3. **Decision engine** -- `src/policy/engine.py`
   - Map severity to action: NONE-->approve, LOW-->approve+log, MEDIUM-->flag,
     HIGH-->remove, CRITICAL-->remove+escalate
   - Configurable per-category actions
   - User history context: repeat offenders get stricter treatment
   ```python
   class PolicyEngine:
       def __init__(self, config: dict):
           self.rule_filter = RuleBasedFilter(config["rules"])
           self.severity_scorer = SeverityScorer(config["severity"])
           self.action_map = config["actions"]

       def decide(self, content: Content, ml_scores: dict) -> Decision:
           # Step 1: Apply rule-based filters
           rule_violations = self.rule_filter.check(content.text)

           # Step 2: Score severity
           severity = self.severity_scorer.score(ml_scores, rule_violations)

           # Step 3: Determine action
           action = self.action_map[severity.overall_severity]

           # Step 4: Check user history for escalation
           if self._is_repeat_offender(content.user_id):
               action = self._escalate_action(action)

           return Decision(
               content_id=content.id,
               severity=severity,
               action=action,
               requires_review=action in ("FLAG", "ESCALATE"),
               explanation=self._generate_explanation(severity, rule_violations),
           )
   ```

4. **Action handlers** -- `src/policy/actions.py`
   - `APPROVE`: log and pass through
   - `FLAG`: add to human review queue with priority
   - `REMOVE`: hide content, notify user
   - `BAN`: remove content, suspend user account
   - `ESCALATE`: notify senior moderator, preserve evidence
   - Each action creates a database record for audit trail

5. **Configuration** -- `configs/moderation_config.yaml`
   ```yaml
   severity:
     thresholds:
       hate_speech:
         LOW: 0.3
         MEDIUM: 0.6
         HIGH: 0.85
       violence:
         LOW: 0.4
         MEDIUM: 0.7
         HIGH: 0.9
       nsfw:
         LOW: 0.3
         MEDIUM: 0.5
         HIGH: 0.8

   actions:
     NONE: APPROVE
     LOW: APPROVE_WITH_LOG
     MEDIUM: FLAG_FOR_REVIEW
     HIGH: AUTO_REMOVE
     CRITICAL: REMOVE_AND_ESCALATE

   repeat_offender:
     lookback_days: 30
     violation_threshold: 3
     escalation_levels: 1  # Bump severity by 1 level
   ```

### Skills Learned

- Building configurable policy engines (ML + rules hybrid)
- Severity scoring and threshold tuning
- Action routing and escalation logic
- Configuration-driven system design
- Audit trail and compliance requirements

---

## Phase 6: Evaluation & Bias Auditing

**Duration:** 4-5 days
**Objective:** Thoroughly evaluate model performance and audit for demographic biases.

### Tasks

1. **Per-modality evaluation**
   - Text model: precision/recall/F1 per category
   - Image model: precision/recall/F1 per category
   - Fusion model: precision/recall/F1 per category
   - Compare: where does fusion outperform individual models?

2. **Combined system evaluation**
   - End-to-end: content in --> decision out
   - Include rule-based filters + ML models + policy engine
   - Measure: overall precision, recall, F1 at each severity level
   - Threshold analysis: precision-recall curves for threshold tuning
   ```python
   def evaluate_full_pipeline(test_data, policy_engine, models):
       results = []
       for sample in test_data:
           ml_scores = predict_all_models(sample, models)
           decision = policy_engine.decide(sample.content, ml_scores)
           results.append({
               "true_label": sample.label,
               "predicted_severity": decision.severity.overall_severity,
               "action": decision.action,
               "scores": ml_scores,
           })
       return compute_metrics(results)
   ```

3. **Bias auditing** -- `src/bias/auditor.py`
   - Test model performance across demographic groups:
     - Race/ethnicity-related terms
     - Gender-related terms
     - Religious references
     - Nationality references
   - Metrics:
     - **Demographic parity:** does the positive rate differ across groups?
     - **Equalized odds:** do TPR and FPR differ across groups?
     - **False positive rate parity:** are innocent posts from some groups flagged more?
   ```python
   class BiasAuditor:
       def __init__(self, demographic_groups: dict):
           self.groups = demographic_groups
           # Example: {"african_american": [...texts...], "caucasian": [...texts...]}

       def audit(self, model, test_data) -> BiasReport:
           group_results = {}
           for group_name, group_data in self.groups.items():
               predictions = model.predict(group_data)
               group_results[group_name] = {
                   "positive_rate": predictions.mean(),
                   "false_positive_rate": self._fpr(predictions, group_data.labels),
                   "false_negative_rate": self._fnr(predictions, group_data.labels),
               }

           disparities = self._compute_disparities(group_results)
           return BiasReport(
               group_results=group_results,
               demographic_parity_gap=disparities["parity_gap"],
               equalized_odds_gap=disparities["odds_gap"],
               recommendations=self._generate_recommendations(disparities),
           )
   ```

4. **False positive analysis**
   - Deep-dive into false positives: what benign content gets flagged?
   - Categorize FPs: sarcasm, reclaimed language, news reporting, education
   - Quantify the cost: how many legitimate posts would be removed per day?
   - This is the most important metric for content moderation

5. **Bias audit notebook** -- `notebooks/05_bias_audit.ipynb`
   - Visualize: per-group false positive rates
   - Identify the most biased categories
   - Document findings in `BIAS_AUDIT_REPORT.md`
   - Propose mitigations: threshold adjustments, training data rebalancing

6. **Write `BIAS_AUDIT_REPORT.md`**
   - Demographic groups tested
   - Disparities found (with confidence intervals)
   - Mitigations applied
   - Remaining known biases
   - Recommendations for future improvement

### Skills Learned

- Bias auditing for ML systems (an essential professional skill)
- Fairness metrics (demographic parity, equalized odds)
- False positive analysis for high-stakes systems
- Writing bias audit reports
- Ethical ML evaluation

---

## Phase 7: API & Serving

**Duration:** 4-5 days
**Objective:** Build the moderation API with batch review and appeal workflows.

### Tasks

1. **Define API schemas** -- `src/serving/schemas.py`
   ```python
   class ModerationRequest(BaseModel):
       content_id: str
       text: str | None = None
       image_url: str | None = None
       user_id: str
       priority: str = "NORMAL"  # NORMAL, HIGH, URGENT

   class ModerationResult(BaseModel):
       content_id: str
       decision: str              # APPROVE, FLAG, REMOVE, BAN, ESCALATE
       severity: str              # NONE, LOW, MEDIUM, HIGH, CRITICAL
       category_scores: dict[str, float]
       rule_violations: list[str]
       explanation: str
       requires_review: bool
       processing_time_ms: float

   class ReviewDecision(BaseModel):
       content_id: str
       reviewer_id: str
       decision: str              # APPROVE, REMOVE, ESCALATE
       reason: str
       overrides_ml: bool         # Did the human override the ML decision?

   class AppealRequest(BaseModel):
       content_id: str
       user_id: str
       reason: str

   class QueueStats(BaseModel):
       pending_reviews: int
       avg_wait_time_seconds: float
       reviews_today: int
       override_rate: float       # How often humans override ML
   ```

2. **Build FastAPI application** -- `src/serving/app.py`
   - `POST /moderate` -- submit content for moderation (async, returns immediately)
   - `GET /moderate/{content_id}` -- check moderation status/result
   - `POST /moderate/batch` -- submit batch of content
   - `GET /review/queue` -- get items pending human review
   - `POST /review/{content_id}` -- submit human review decision
   - `POST /appeal/{content_id}` -- submit user appeal
   - `GET /stats` -- queue statistics and system health
   - `GET /health` -- health check
   - `GET /metrics` -- Prometheus metrics

3. **Human review dashboard** -- `src/frontend/review_dashboard.py`
   - Queue view: sorted by severity and wait time
   - Content display: show text and image side by side
   - ML explanation: show model scores and triggered rules
   - Action buttons: Approve, Remove, Escalate with required reason
   - Appeal view: show original decision, user appeal text, allow re-review
   ```python
   # Streamlit review dashboard
   st.title("Content Moderation Review Queue")

   # Filters
   severity_filter = st.sidebar.multiselect(
       "Severity", ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
   )

   # Queue
   queue = fetch_review_queue(severity=severity_filter)
   for item in queue:
       with st.expander(f"[{item.severity}] Content {item.content_id}"):
           col1, col2 = st.columns(2)
           with col1:
               st.write("**Text:**", item.text)
           with col2:
               if item.image_url:
                   st.image(item.image_url)

           st.write("**ML Scores:**", item.category_scores)
           st.write("**Rule Violations:**", item.rule_violations)

           decision = st.radio("Decision", ["Approve", "Remove", "Escalate"])
           reason = st.text_input("Reason (required)")
           if st.button("Submit", key=item.content_id):
               submit_review(item.content_id, decision, reason)
   ```

4. **Appeal workflow**
   - Users can appeal moderation decisions
   - Appeals go to a separate queue with higher priority
   - Different reviewer than original decision (where possible)
   - Track appeal outcomes: overturn rate, common appeal reasons

5. **Database models** (PostgreSQL via SQLAlchemy)
   - `moderation_decisions`: content_id, decision, severity, scores, timestamp
   - `human_reviews`: content_id, reviewer_id, decision, reason, timestamp
   - `appeals`: content_id, user_id, reason, status, outcome
   - `user_violations`: user_id, violation_count, last_violation, status

### Skills Learned

- Designing moderation APIs with async processing
- Human review workflows and dashboard design
- Appeal process implementation
- Database schema design for audit trails
- Building operational dashboards

---

## Phase 8: Async Processing Pipeline

**Duration:** 4-5 days
**Objective:** Build a distributed task queue for high-throughput content processing.

### Tasks

1. **Celery application** -- `src/pipeline/celery_app.py`
   - Configure Celery with RabbitMQ as broker and Redis as result backend
   - Define queues: `text_analysis`, `image_analysis`, `fusion`, `policy`
   - Configure concurrency based on resource requirements
   ```python
   from celery import Celery

   app = Celery("moderation")
   app.config_from_object({
       "broker_url": "amqp://guest:guest@rabbitmq:5672//",
       "result_backend": "redis://redis:6379/0",
       "task_serializer": "json",
       "result_serializer": "json",
       "task_routes": {
           "tasks.analyze_text": {"queue": "text_analysis"},
           "tasks.analyze_image": {"queue": "image_analysis"},
           "tasks.fuse_results": {"queue": "fusion"},
           "tasks.apply_policy": {"queue": "policy"},
       },
       "worker_prefetch_multiplier": 1,  # For long-running ML tasks
       "task_acks_late": True,           # Re-queue on worker crash
   })
   ```

2. **Task definitions** -- `src/pipeline/tasks.py`
   - Chain: `analyze_text | analyze_image | fuse_results | apply_policy`
   - Each task is independently retryable
   - Handle failures gracefully: if image analysis fails, proceed with text only
   ```python
   @app.task(bind=True, max_retries=3, default_retry_delay=10)
   def analyze_text(self, content_id: str, text: str) -> dict:
       try:
           model = get_text_model()
           scores = model.predict(text)
           return {"content_id": content_id, "text_scores": scores}
       except Exception as exc:
           self.retry(exc=exc)

   @app.task(bind=True, max_retries=3)
   def analyze_image(self, content_id: str, image_url: str) -> dict:
       try:
           image = download_image(image_url)
           model = get_image_model()
           scores = model.predict(image)
           return {"content_id": content_id, "image_scores": scores}
       except Exception as exc:
           self.retry(exc=exc)

   @app.task
   def moderate_content(content_id: str, text: str, image_url: str | None):
       """Orchestrate the full moderation pipeline."""
       chain = (
           analyze_text.s(content_id, text)
           | analyze_image.s(content_id, image_url)
           | fuse_results.s(content_id)
           | apply_policy.s(content_id)
       )
       chain.apply_async()
   ```

3. **Priority queue routing** -- `src/pipeline/priority.py`
   - URGENT: reported content, high-profile accounts --> process within 30 seconds
   - HIGH: content from flagged users --> process within 2 minutes
   - NORMAL: standard content --> process within 5 minutes
   - LOW: bulk re-moderation --> process during off-peak hours
   ```python
   class PriorityRouter:
       PRIORITY_MAP = {
           "URGENT": 9,
           "HIGH": 6,
           "NORMAL": 3,
           "LOW": 0,
       }

       def route_task(self, content: Content) -> dict:
           priority = self._determine_priority(content)
           return {
               "queue": f"moderation_{priority.lower()}",
               "priority": self.PRIORITY_MAP[priority],
           }

       def _determine_priority(self, content: Content) -> str:
           if content.is_reported:
               return "URGENT"
           if content.user_is_flagged:
               return "HIGH"
           return "NORMAL"
   ```

4. **Duplicate content detection** -- `src/pipeline/dedup.py`
   - Compute perceptual hash for images (pHash)
   - Compute SimHash for text
   - Check Redis for recent duplicates before full analysis
   - If duplicate found, copy the previous decision (much cheaper)

5. **Worker configuration and scaling**
   - Text workers: CPU-bound, high concurrency (4-8 workers)
   - Image workers: GPU-preferred, lower concurrency (1-2 workers)
   - Fusion workers: CPU-bound, medium concurrency
   - Policy workers: lightweight, high concurrency
   - Auto-scaling rules based on queue depth

### Skills Learned

- Distributed task queues (Celery + RabbitMQ)
- Task chaining and error recovery
- Priority queue routing
- Duplicate detection at scale
- Worker scaling strategies for ML workloads

---

## Phase 9: Containerization

**Duration:** 2-3 days
**Objective:** Package all components into Docker containers.

### Tasks

1. **Worker Dockerfile** -- `docker/Dockerfile.worker`
   - Include all ML model weights
   - Configure for CPU (with optional GPU support)
   - Separate Dockerfiles for text workers vs image workers (different dependencies)
   ```dockerfile
   FROM python:3.11-slim AS builder
   WORKDIR /app
   COPY pyproject.toml .
   RUN pip install --no-cache-dir .[worker]

   FROM python:3.11-slim
   WORKDIR /app
   COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
   COPY --from=builder /usr/local/bin /usr/local/bin
   COPY src/ /app/src/
   COPY configs/ /app/configs/
   COPY models/ /app/models/

   CMD ["celery", "-A", "src.pipeline.celery_app", "worker", \
        "--loglevel=info", "--concurrency=4", "-Q", "text_analysis"]
   ```

2. **docker-compose.yaml** -- orchestrate everything
   ```yaml
   services:
     api:
       build: { context: ., dockerfile: docker/Dockerfile.api }
       ports: ["8000:8000"]
       depends_on: [rabbitmq, postgres, redis]

     text-worker:
       build: { context: ., dockerfile: docker/Dockerfile.worker }
       command: celery -A src.pipeline.celery_app worker -Q text_analysis -c 4
       depends_on: [rabbitmq, redis]

     image-worker:
       build: { context: ., dockerfile: docker/Dockerfile.worker }
       command: celery -A src.pipeline.celery_app worker -Q image_analysis -c 2
       depends_on: [rabbitmq, redis]

     policy-worker:
       build: { context: ., dockerfile: docker/Dockerfile.worker }
       command: celery -A src.pipeline.celery_app worker -Q policy -c 8
       depends_on: [rabbitmq, redis, postgres]

     frontend:
       build: { context: ., dockerfile: docker/Dockerfile.frontend }
       ports: ["8501:8501"]
       depends_on: [api]

     rabbitmq:
       image: rabbitmq:3-management
       ports: ["5672:5672", "15672:15672"]

     postgres:
       image: postgres:16
       environment:
         POSTGRES_DB: moderation
         POSTGRES_USER: mod_user
         POSTGRES_PASSWORD: mod_pass
       volumes: [postgres_data:/var/lib/postgresql/data]
       ports: ["5432:5432"]

     redis:
       image: redis:7
       ports: ["6379:6379"]

     prometheus:
       image: prom/prometheus
       volumes: ["./prometheus:/etc/prometheus"]
       ports: ["9090:9090"]

     grafana:
       image: grafana/grafana
       volumes: ["./grafana:/etc/grafana/provisioning"]
       ports: ["3000:3000"]

   volumes:
     postgres_data:
   ```

3. **Verify the full stack**
   - `docker compose up` -- all services start
   - Submit test content via API, verify it flows through the pipeline
   - Check review dashboard shows flagged content
   - Verify metrics in Prometheus/Grafana

### Skills Learned

- Docker for multi-worker ML systems
- Celery worker containers with proper resource allocation
- Docker Compose for complex multi-service architectures
- Managing stateful services (PostgreSQL, RabbitMQ, Redis)

---

## Phase 10: Testing & CI/CD

**Duration:** 3-4 days
**Objective:** Build comprehensive tests for a multi-model, multi-service system.

### Tasks

1. **Unit tests**
   ```
   test_text_preprocessing.py
   ├── test_social_media_normalization
   ├── test_profanity_normalization
   └── test_tokenizer_max_length

   test_policy_engine.py
   ├── test_severity_scoring_low
   ├── test_severity_scoring_high
   ├── test_rule_violation_escalation
   ├── test_repeat_offender_escalation
   └── test_configurable_thresholds

   test_severity_scoring.py
   ├── test_all_categories_below_threshold
   ├── test_single_category_above_high
   └── test_rule_override
   ```

2. **Integration tests**
   ```
   test_celery_pipeline.py
   ├── test_text_analysis_task
   ├── test_image_analysis_task
   ├── test_full_pipeline_chain
   ├── test_task_retry_on_failure
   └── test_priority_routing

   test_api.py
   ├── test_moderate_text_only
   ├── test_moderate_text_and_image
   ├── test_review_queue_endpoint
   ├── test_submit_review_decision
   └── test_appeal_workflow
   ```

3. **Bias regression tests** -- `tests/bias/`
   - Run bias audit on every PR
   - Fail if demographic parity gap exceeds threshold
   - Fail if false positive rate for any group exceeds 2x the overall rate
   ```python
   def test_demographic_parity():
       auditor = BiasAuditor(DEMOGRAPHIC_GROUPS)
       report = auditor.audit(model, test_data)
       assert report.demographic_parity_gap < 0.1, (
           f"Parity gap {report.demographic_parity_gap} exceeds threshold 0.1"
       )

   def test_false_positive_rate_parity():
       auditor = BiasAuditor(DEMOGRAPHIC_GROUPS)
       report = auditor.audit(model, test_data)
       overall_fpr = report.overall_false_positive_rate
       for group, metrics in report.group_results.items():
           assert metrics["false_positive_rate"] < overall_fpr * 2, (
               f"Group {group} FPR {metrics['false_positive_rate']:.3f} "
               f"exceeds 2x overall FPR {overall_fpr:.3f}"
           )
   ```

4. **CI pipeline** -- `.github/workflows/ci.yaml`
   ```yaml
   name: CI
   on: [pull_request]
   jobs:
     lint:
       steps:
         - run: ruff check .
         - run: mypy src/
     unit-test:
       steps:
         - run: pytest tests/unit/ -v
     integration-test:
       services:
         rabbitmq: { image: rabbitmq:3 }
         postgres: { image: postgres:16 }
         redis: { image: redis:7 }
       steps:
         - run: pytest tests/integration/ -v --timeout=120
     bias-test:
       steps:
         - run: pytest tests/bias/ -v --timeout=300
     build:
       steps:
         - run: docker compose build
   ```

5. **CD pipeline** -- `.github/workflows/cd.yaml`
   - Build and push all Docker images
   - Run integration tests against staging
   - Run bias regression tests before deploy
   - Deploy with canary strategy (route 5% of traffic to new version)

### Skills Learned

- Testing multi-model systems
- Bias regression testing in CI/CD
- Integration testing with Docker services (RabbitMQ, PostgreSQL, Redis)
- Canary deployment strategies

---

## Phase 11: Monitoring

**Duration:** 3-4 days
**Objective:** Track moderation volume, quality, bias, and queue health in production.

### Tasks

1. **Prometheus metrics** -- `src/monitoring/metrics.py`
   - `moderation_requests_total` -- counter by decision (approve/flag/remove)
   - `moderation_latency_seconds` -- histogram (per pipeline stage)
   - `moderation_scores` -- histogram per category
   - `review_queue_depth` -- gauge (items pending review)
   - `review_decisions_total` -- counter by decision
   - `human_override_rate` -- gauge (how often humans disagree with ML)
   - `appeals_total` -- counter by outcome (upheld/overturned)
   - `active_workers` -- gauge per queue type

2. **Bias monitoring** -- `src/monitoring/bias_monitor.py`
   - Track false positive rates by demographic group in real-time
   - Alert if disparity exceeds threshold
   - Log demographic breakdowns hourly
   ```python
   class BiasMonitor:
       def __init__(self, window_minutes: int = 60):
           self.window = window_minutes
           self.group_counters = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})

       def record(self, prediction: str, ground_truth: str, demographic_group: str):
           counter = self.group_counters[demographic_group]
           if prediction == "REMOVE" and ground_truth == "BENIGN":
               counter["fp"] += 1
           elif prediction == "REMOVE" and ground_truth == "HARMFUL":
               counter["tp"] += 1
           # ...

       def check_disparity(self) -> list[BiasAlert]:
           alerts = []
           fprs = {g: c["fp"] / max(c["fp"] + c["tn"], 1)
                   for g, c in self.group_counters.items()}
           avg_fpr = sum(fprs.values()) / len(fprs)
           for group, fpr in fprs.items():
               if fpr > avg_fpr * 1.5:
                   alerts.append(BiasAlert(group=group, fpr=fpr, avg_fpr=avg_fpr))
           return alerts
   ```

3. **Queue monitoring** -- `src/monitoring/queue_monitor.py`
   - Track queue depth per priority level
   - Track processing time per task type
   - Alert if queue depth exceeds threshold (workers not keeping up)
   - Track worker health: crash rate, restart frequency

4. **Grafana dashboard** -- `grafana/dashboards/moderation_monitoring.json`
   - **Row 1 -- Volume:** moderation requests/second, decisions breakdown, error rate
   - **Row 2 -- Quality:** human override rate, appeal overturn rate, false positive rate
   - **Row 3 -- Bias:** per-group false positive rates, demographic parity gap over time
   - **Row 4 -- Queue:** queue depth by priority, processing latency, worker utilization
   - **Row 5 -- System:** CPU, memory, RabbitMQ connection count, PostgreSQL connections

5. **Alerting rules**
   - Queue depth > 1000 for 5 minutes --> alert (scale up workers)
   - Human override rate > 30% --> alert (model may need retraining)
   - False positive rate disparity > 2x for any group --> alert
   - Error rate > 5% --> alert
   - Worker crash rate > 1/hour --> alert
   - Appeal overturn rate > 50% --> alert (policy may need adjustment)

### Skills Learned

- Monitoring multi-service ML systems
- Real-time bias tracking in production
- Queue-based system observability
- Setting meaningful alerts for content moderation
- Human-override rate as a model quality signal

---

## Timeline Summary

```
Week 1   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 1: Setup & Design Doc       (3 days)
         Phase 2: Data Pipeline            (2 days)

Week 2   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 2: Data Pipeline            (3 days)
         Phase 3: Individual Models        (2 days)

Week 3   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 3: Individual Models        (4 days)
         Buffer                            (1 day)

Week 4   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 4: Multi-Modal Fusion       (5 days)

Week 5   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 4: Fusion (continued)       (1 day)
         Phase 5: Policy Engine            (4 days)

Week 6   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 6: Evaluation & Bias Audit  (5 days)

Week 7   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 7: API & Serving            (5 days)

Week 8   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 8: Async Processing         (5 days)

Week 9   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 9: Containerization         (3 days)
         Phase 10: Testing & CI/CD         (2 days)

Week 10  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 10: Testing & CI/CD         (2 days)
         Phase 11: Monitoring              (3 days)

Week 11  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 11: Monitoring              (1 day)
         Buffer / polish                   (4 days)
```

**Total: ~50 days (10-11 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] Designing content moderation systems and policies
- [ ] Writing annotation guidelines for ML datasets
- [ ] Multi-modal dataset preparation (text + image)
- [ ] Social media text preprocessing (emojis, hashtags, obfuscation)
- [ ] Fine-tuning BERT for text classification
- [ ] Fine-tuning ViT for image classification
- [ ] Multi-label classification with class imbalance
- [ ] Multi-modal fusion architectures (late, CLIP-based, cross-attention)
- [ ] CLIP model usage and fine-tuning
- [ ] Building configurable policy engines (ML + rules hybrid)
- [ ] Severity scoring and threshold tuning
- [ ] Bias auditing across demographic groups
- [ ] Fairness metrics (demographic parity, equalized odds)
- [ ] False positive analysis for high-stakes systems
- [ ] Distributed task queues (Celery + RabbitMQ)
- [ ] Priority queue routing and task chaining
- [ ] Duplicate content detection (perceptual hashing)
- [ ] Human review workflows and dashboard design
- [ ] Appeal process implementation
- [ ] PostgreSQL schema design for audit trails
- [ ] Redis for caching and rate limiting
- [ ] Docker for multi-worker ML systems
- [ ] Integration testing with Docker services
- [ ] Bias regression testing in CI/CD
- [ ] Real-time bias monitoring in production
- [ ] Queue-based system observability
- [ ] Monitoring human override rates as a quality signal
- [ ] Ethical ML system design and documentation

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Initialize the project
mkdir content-moderation && cd content-moderation
git init

# 2. Create the folder structure
mkdir -p configs data/{raw,processed,annotations,evaluation} notebooks \
  src/{data,models,training,policy,pipeline,bias,serving,monitoring,frontend} \
  tests/{unit,integration,bias} docker .github/workflows grafana/dashboards prometheus scripts

# 3. Start infrastructure dependencies
cat > docker-compose.dev.yaml << 'EOF'
services:
  rabbitmq:
    image: rabbitmq:3-management
    ports: ["5672:5672", "15672:15672"]
  postgres:
    image: postgres:16
    environment: { POSTGRES_DB: moderation, POSTGRES_USER: dev, POSTGRES_PASSWORD: dev }
    ports: ["5432:5432"]
  redis:
    image: redis:7
    ports: ["6379:6379"]
EOF
docker compose -f docker-compose.dev.yaml up -d

# 4. Start writing DESIGN_DOC.md
```
