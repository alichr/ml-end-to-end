# Project 5: Movie Recommendation Engine (Intermediate+)

## Goal

Build a **production-grade movie recommendation system** that suggests personalized
movies to users based on their viewing history and preferences. The system combines
collaborative filtering, neural embeddings, and content-based features into a hybrid
model, served through a low-latency API with Redis caching. By the end, you will
understand how recommendation systems work at companies like Netflix and Spotify --
from offline training to real-time serving with A/B testing.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why recommendations? | Recommendation systems power the most valuable features at tech companies (Netflix, Amazon, Spotify, YouTube). Understanding them is a massive career advantage. |
| Why MovieLens? | MovieLens 25M is the gold standard academic dataset for recommendations -- 25 million ratings from 162,000 users on 62,000 movies. Clean, well-documented, and free. |
| How is this different from classification? | Instead of predicting a label, you are predicting **which items a user will like** from a catalog of 62,000 movies. This requires user-item interaction modeling, not just feature-to-label mapping. |
| What new skills will I learn? | Embedding models, user-item interaction data, Redis caching for ML, A/B testing frameworks, and the critical distinction between ML metrics and business metrics. |
| Will this look good on a portfolio? | Absolutely. Recommendation systems are one of the most asked-about ML topics in industry interviews. A working system with A/B testing shows real engineering maturity. |

---

## Architecture Overview

```
                         ┌────────────────┐
                         │   Streamlit    │
                         │   Frontend     │
                         └───────┬────────┘
                                 │ HTTP
                                 ▼
                         ┌────────────────┐
    User Request ───────▶│   FastAPI      │
                         │   Server       │
                         └───┬───┬───┬────┘
                             │   │   │
                ┌────────────┘   │   └────────────┐
                ▼                ▼                 ▼
         ┌────────────┐  ┌────────────┐   ┌────────────┐
         │   Redis    │  │  Hybrid    │   │ PostgreSQL │
         │   Cache    │  │  RecModel  │   │ User Data  │
         │ (hot recs) │  │ (PyTorch)  │   │ & Ratings  │
         └────────────┘  └─────┬──────┘   └────────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
           ┌──────────┐ ┌──────────┐ ┌──────────┐
           │  MLflow  │ │Prometheus│ │ A/B Test │
           │ Registry │ │ Metrics  │ │ Framework│
           └──────────┘ └────┬─────┘ └──────────┘
                             ▼
                       ┌──────────┐
                       │ Grafana  │
                       │Dashboard │
                       └──────────┘

Everything runs in Docker. API + Redis + PostgreSQL + Frontend + Monitoring.
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | ML ecosystem standard |
| ML Framework | PyTorch | Neural collaborative filtering, embedding layers |
| Baseline Library | Surprise | Fast implementation of SVD, KNN, and other classic RecSys algorithms |
| Database | PostgreSQL | Relational store for users, movies, ratings -- supports complex queries |
| Cache | Redis | Sub-millisecond lookups for precomputed recommendations |
| Experiment Tracking | MLflow | Track offline metrics across model variants |
| API Framework | FastAPI | Async support critical for I/O-heavy recommendation serving |
| Frontend | Streamlit | Quick interactive UI for browsing recommendations |
| Containerization | Docker + docker-compose | Orchestrate API, Redis, PostgreSQL, frontend, monitoring |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Monitoring | Prometheus + Grafana | Track recommendation quality and system health |
| Testing | pytest | Unit and integration testing |
| Data Processing | pandas, scipy | Sparse matrix operations, data manipulation |

---

## Project Structure

```
movie-recommendation-engine/
│
├── doc/
│   ├── DESIGN_DOC.md                # Problem statement, constraints, success criteria
│   ├── MODEL_CARD.md                # Model documentation
│   └── PROJECT_PLAN.md              # This file
│
├── pyproject.toml                   # Dependencies and project metadata
├── dvc.yaml                         # Data pipeline definition
│
├── configs/
│   ├── train_config.yaml            # Training hyperparameters
│   ├── serve_config.yaml            # Serving configuration (Redis TTL, batch sizes)
│   └── ab_test_config.yaml          # A/B test definitions and traffic splits
│
├── data/                            # Git-ignored, DVC-tracked
│   ├── raw/                         # Original MovieLens files
│   │   ├── ratings.csv              # 25M ratings (userId, movieId, rating, timestamp)
│   │   ├── movies.csv               # Movie metadata (movieId, title, genres)
│   │   ├── tags.csv                 # User-generated tags
│   │   └── genome-scores.csv        # Tag relevance scores
│   ├── processed/                   # Cleaned, encoded data
│   └── splits/                      # train / val / test (temporal split)
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Rating distributions, user activity, item popularity
│   ├── 02_cold_start_analysis.ipynb # New user / new item analysis
│   ├── 03_model_experiments.ipynb   # Interactive model development
│   └── 04_evaluation.ipynb          # Offline evaluation, diversity analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py              # Download MovieLens 25M
│   │   ├── preprocess.py            # Clean, encode user/item IDs
│   │   ├── dataset.py               # PyTorch Dataset for interactions
│   │   ├── features.py              # Feature engineering (user profiles, item features)
│   │   └── split.py                 # Temporal train/val/test split
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── svd_baseline.py          # Surprise SVD model
│   │   ├── ncf.py                   # Neural Collaborative Filtering (PyTorch)
│   │   ├── hybrid.py                # Hybrid model (collaborative + content)
│   │   └── export.py                # Export models for serving
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_svd.py             # Train SVD baseline
│   │   ├── train_ncf.py             # Train neural collaborative filtering
│   │   ├── train_hybrid.py          # Train hybrid model
│   │   ├── evaluate.py              # Offline evaluation metrics
│   │   └── callbacks.py             # Early stopping, checkpointing
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application
│   │   ├── schemas.py               # Request/response Pydantic models
│   │   ├── recommend.py             # Recommendation logic (cache-first)
│   │   ├── cache.py                 # Redis caching layer
│   │   └── db.py                    # PostgreSQL connection and queries
│   │
│   ├── ab_testing/
│   │   ├── __init__.py
│   │   ├── router.py                # Route users to champion or challenger
│   │   ├── tracker.py               # Track impressions, clicks, engagement
│   │   └── analyzer.py              # Statistical significance testing
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Prometheus metric definitions
│   │   ├── quality.py               # Recommendation quality tracking
│   │   └── bias.py                  # Popularity bias monitoring
│   │
│   └── frontend/
│       └── app.py                   # Streamlit UI
│
├── tests/
│   ├── unit/
│   │   ├── test_dataset.py          # Dataset encoding, shape checks
│   │   ├── test_models.py           # Model output shapes, embedding dims
│   │   ├── test_cache.py            # Redis cache hit/miss logic
│   │   ├── test_schemas.py          # API schema validation
│   │   └── test_ab_router.py        # A/B test routing determinism
│   ├── integration/
│   │   ├── test_api.py              # Full recommendation request pipeline
│   │   ├── test_cache_db.py         # Cache + DB interaction
│   │   └── test_training.py         # Training runs without error
│   └── conftest.py                  # Shared fixtures
│
├── docker/
│   ├── Dockerfile.api               # Multi-stage build for API
│   ├── Dockerfile.frontend          # Streamlit container
│   └── Dockerfile.training          # Training environment
│
├── docker-compose.yaml              # API + Redis + PostgreSQL + Frontend + Monitoring
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                  # Lint -> Test -> Build on PR
│       └── cd.yaml                  # Deploy on merge to main
│
├── grafana/
│   └── dashboards/
│       └── rec_monitoring.json      # Recommendation quality dashboard
│
├── prometheus/
│   └── prometheus.yml               # Scrape config
│
└── scripts/
    ├── setup.sh                     # One-command project setup
    ├── seed_db.sh                   # Load MovieLens data into PostgreSQL
    ├── warm_cache.sh                # Pre-populate Redis with top recommendations
    └── run_ab_analysis.sh           # Analyze A/B test results
```

---

## Phase 1: Project Setup & Design Doc

**Duration:** 1-2 days
**Objective:** Define the recommendation problem clearly and set up the development environment.

### Tasks

1. **Write `DESIGN_DOC.md`**
   - **Problem statement:** "Given a user's rating history, recommend the top-K movies
     they are most likely to enjoy, ranked by predicted preference."
   - **Success criteria:**
     - NDCG@10 >= 0.35 on the held-out test set
     - Precision@10 >= 0.20
     - Recommendation diversity (ILS) >= 0.60
     - API latency < 100ms for cached requests, < 500ms for cold requests
     - System handles 200 concurrent users
   - **Out of scope:** real-time streaming updates, social features, review text analysis
   - **Risks:** cold start for new users (< 5 ratings), popularity bias dominating
     recommendations, scalability beyond single-node

2. **Initialize the repository**
   - `git init`, create `.gitignore` (data/, models/, *.pyc, .env, mlruns/, etc.)
   - Set up branch strategy: `main` (production), `dev` (integration), `feature/*`
   - Create `pyproject.toml` with all dependencies:
     ```toml
     [project]
     name = "movie-recommendation-engine"
     version = "0.1.0"
     requires-python = ">=3.11"
     dependencies = [
         "torch>=2.0",
         "scikit-surprise>=1.1",
         "fastapi>=0.100",
         "uvicorn>=0.23",
         "redis>=5.0",
         "psycopg2-binary>=2.9",
         "sqlalchemy>=2.0",
         "mlflow>=2.8",
         "streamlit>=1.28",
         "pandas>=2.0",
         "scipy>=1.11",
         "prometheus-client>=0.19",
         "pydantic>=2.0",
         "httpx>=0.25",
     ]
     ```

3. **Create the folder structure** (as shown above)

4. **Set up development environment**
   - Python virtual environment
   - Pre-commit hooks: ruff, mypy
   - Install and verify PostgreSQL and Redis locally (or via Docker)
   - Verify GPU availability for PyTorch (optional but helpful for NCF training)

### Skills Learned

- Designing recommendation system architectures
- Understanding the recommendation problem formulation (implicit vs explicit feedback)
- Setting up multi-service development environments (API + DB + Cache)

---

## Phase 2: Data Pipeline

**Duration:** 4-5 days
**Objective:** Download, explore, and prepare the MovieLens 25M dataset for model training.

### Tasks

1. **Download MovieLens 25M** -- `src/data/download.py`
   - Download from https://grouplens.org/datasets/movielens/25m/
   - Extract and validate file integrity (check row counts, column names)
   - Files: `ratings.csv` (25M rows), `movies.csv` (62K rows), `tags.csv`, `genome-scores.csv`
   - Make the script idempotent -- skip download if files already exist
   ```python
   MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

   def download_movielens(data_dir: Path) -> None:
       """Download and extract MovieLens 25M dataset."""
       zip_path = data_dir / "ml-25m.zip"
       if (data_dir / "raw" / "ratings.csv").exists():
           logger.info("Dataset already downloaded, skipping.")
           return
       logger.info(f"Downloading MovieLens 25M from {MOVIELENS_URL}...")
       urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
       with zipfile.ZipFile(zip_path, "r") as z:
           z.extractall(data_dir / "raw")
       zip_path.unlink()  # Clean up zip file
   ```

2. **Set up DVC for data versioning**
   - `dvc init`, track `data/raw/` with DVC
   - Configure remote storage
   - This dataset is large (250MB+ compressed) -- DVC is essential

3. **Exploratory Data Analysis** -- `notebooks/01_eda.ipynb`

   This is where recommendations get interesting. Key analyses:

   - **Rating distribution:** histogram of ratings (1-5 stars). MovieLens is famously
     skewed toward higher ratings (most ratings are 3-5).
   - **User activity distribution:** how many ratings per user? Plot the long tail.
     Most users rate very few movies, some rate thousands.
     ```python
     user_counts = ratings.groupby("userId").size()
     print(f"Median ratings per user: {user_counts.median()}")
     print(f"Mean ratings per user: {user_counts.mean():.0f}")
     print(f"Users with < 20 ratings: {(user_counts < 20).sum()}")
     # Expect: heavy long-tail distribution
     ```
   - **Item popularity distribution:** how many ratings per movie? The top 1% of movies
     receive a huge fraction of all ratings (popularity bias).
   - **Temporal patterns:** are ratings increasing over time? Are there seasonal patterns?
     Plot ratings per month.
   - **Genre analysis:** which genres are most rated? What is the average rating per genre?
   - **Sparsity analysis:** the user-item interaction matrix is extremely sparse.
     Calculate: `num_ratings / (num_users * num_movies)` -- expect ~0.02% density.
     ```python
     n_users = ratings["userId"].nunique()
     n_movies = ratings["movieId"].nunique()
     n_ratings = len(ratings)
     sparsity = 1 - (n_ratings / (n_users * n_movies))
     print(f"Matrix sparsity: {sparsity:.4%}")
     # Expect: ~99.97% sparse
     ```

4. **Cold Start Analysis** -- `notebooks/02_cold_start_analysis.ipynb`
   - Identify users with fewer than 5 ratings (cold start users)
   - Identify movies with fewer than 10 ratings (cold start items)
   - Analyze: can we use genre preferences or popularity as a fallback?
   - This analysis directly informs the hybrid model design

5. **Data Preprocessing** -- `src/data/preprocess.py`
   - Re-index user IDs and movie IDs to contiguous integers (0, 1, 2, ...)
     This is critical for embedding layers.
     ```python
     def encode_ids(ratings: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
         """Map original IDs to contiguous integers for embeddings."""
         user_encoder = {uid: idx for idx, uid in enumerate(ratings["userId"].unique())}
         item_encoder = {mid: idx for idx, mid in enumerate(ratings["movieId"].unique())}
         ratings["user_idx"] = ratings["userId"].map(user_encoder)
         ratings["item_idx"] = ratings["movieId"].map(item_encoder)
         return ratings, user_encoder, item_encoder
     ```
   - Convert explicit ratings to implicit feedback (rating >= 4 = positive interaction)
   - Extract genre features as multi-hot vectors for the content-based component
   - Save encoder mappings for serving (you need to map back to real movie IDs)

6. **Feature Engineering** -- `src/data/features.py`
   - **User features:** average rating, number of ratings, genre preference vector,
     rating variance (picky vs generous raters)
   - **Item features:** average rating, number of ratings, genre vector, year released,
     tag genome features (if using genome-scores.csv)
   - **Interaction features:** time since user's first rating, recency of this rating

7. **Temporal Train/Val/Test Split** -- `src/data/split.py`
   - **Do NOT use random split** for recommendation data. Use temporal split:
     - Train: first 80% of each user's ratings (by timestamp)
     - Validation: next 10%
     - Test: final 10%
   - Why? Random splits leak future information. In production, you only have past data.
   ```python
   def temporal_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
       """Split each user's ratings chronologically."""
       ratings = ratings.sort_values(["userId", "timestamp"])
       train, val, test = [], [], []
       for _, user_ratings in ratings.groupby("userId"):
           n = len(user_ratings)
           train.append(user_ratings.iloc[:int(0.8 * n)])
           val.append(user_ratings.iloc[int(0.8 * n):int(0.9 * n)])
           test.append(user_ratings.iloc[int(0.9 * n):])
       return pd.concat(train), pd.concat(val), pd.concat(test)
   ```

8. **PyTorch Dataset** -- `src/data/dataset.py`
   - Dataset class for user-item interactions with negative sampling
   - For each positive interaction, sample N negative items (items the user has not rated)
   - This is essential for training the neural model
   ```python
   class InteractionDataset(Dataset):
       def __init__(self, interactions: pd.DataFrame, n_items: int, n_negatives: int = 4):
           self.users = interactions["user_idx"].values
           self.items = interactions["item_idx"].values
           self.ratings = interactions["rating"].values
           self.n_items = n_items
           self.n_negatives = n_negatives
           self.user_positive_items = (
               interactions.groupby("user_idx")["item_idx"].apply(set).to_dict()
           )

       def __getitem__(self, idx):
           user = self.users[idx]
           pos_item = self.items[idx]
           rating = self.ratings[idx]
           # Sample negative items
           neg_items = []
           while len(neg_items) < self.n_negatives:
               neg = random.randint(0, self.n_items - 1)
               if neg not in self.user_positive_items[user]:
                   neg_items.append(neg)
           return user, pos_item, neg_items, rating
   ```

9. **Define DVC pipeline** -- `dvc.yaml`
   - Stage 1: download -> Stage 2: preprocess -> Stage 3: features -> Stage 4: split
   - Reproducible with `dvc repro`

### Skills Learned

- Working with large-scale user-item interaction data
- Understanding recommendation data characteristics (sparsity, long tails, cold start)
- Temporal splitting for recommendation evaluation (avoiding data leakage)
- Negative sampling for implicit feedback
- Feature engineering for users and items

---

## Phase 3: Model Development

**Duration:** 5-7 days
**Objective:** Build three recommendation models of increasing complexity and track all experiments.

### Tasks

1. **SVD Baseline** -- `src/model/svd_baseline.py`
   - Use the Surprise library's SVD implementation (matrix factorization)
   - This is your baseline -- every other model must beat this
   ```python
   from surprise import SVD, Dataset, Reader
   from surprise.model_selection import cross_validate

   def train_svd(ratings_df: pd.DataFrame, n_factors: int = 100) -> SVD:
       """Train SVD baseline using Surprise library."""
       reader = Reader(rating_scale=(0.5, 5.0))
       data = Dataset.load_from_df(
           ratings_df[["userId", "movieId", "rating"]], reader
       )
       model = SVD(n_factors=n_factors, n_epochs=20, lr_all=0.005, reg_all=0.02)
       # Cross-validate for reliable estimates
       cv_results = cross_validate(model, data, measures=["RMSE", "MAE"], cv=5)
       logger.info(f"SVD RMSE: {cv_results['test_rmse'].mean():.4f}")
       # Train on full training set
       trainset = data.build_full_trainset()
       model.fit(trainset)
       return model
   ```
   - Log RMSE, MAE, and training time to MLflow
   - Experiment with n_factors: 50, 100, 200

2. **Neural Collaborative Filtering (NCF)** -- `src/model/ncf.py`
   - Implement the NCF architecture from the seminal He et al. 2017 paper
   - Combines Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP)
   ```python
   class NeuralCollaborativeFiltering(nn.Module):
       def __init__(
           self,
           n_users: int,
           n_items: int,
           embed_dim: int = 64,
           mlp_dims: list[int] = [128, 64, 32],
       ):
           super().__init__()
           # GMF pathway
           self.gmf_user_embed = nn.Embedding(n_users, embed_dim)
           self.gmf_item_embed = nn.Embedding(n_items, embed_dim)

           # MLP pathway
           self.mlp_user_embed = nn.Embedding(n_users, embed_dim)
           self.mlp_item_embed = nn.Embedding(n_items, embed_dim)

           mlp_layers = []
           input_dim = embed_dim * 2
           for dim in mlp_dims:
               mlp_layers.extend([
                   nn.Linear(input_dim, dim),
                   nn.ReLU(),
                   nn.BatchNorm1d(dim),
                   nn.Dropout(0.2),
               ])
               input_dim = dim
           self.mlp = nn.Sequential(*mlp_layers)

           # Fusion layer
           self.output = nn.Linear(embed_dim + mlp_dims[-1], 1)

       def forward(self, user_ids, item_ids):
           # GMF pathway
           gmf_user = self.gmf_user_embed(user_ids)
           gmf_item = self.gmf_item_embed(item_ids)
           gmf_out = gmf_user * gmf_item  # Element-wise product

           # MLP pathway
           mlp_user = self.mlp_user_embed(user_ids)
           mlp_item = self.mlp_item_embed(item_ids)
           mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
           mlp_out = self.mlp(mlp_input)

           # Fuse and predict
           concat = torch.cat([gmf_out, mlp_out], dim=-1)
           prediction = self.output(concat).squeeze(-1)
           return prediction
   ```
   - Key hyperparameters to tune: embedding dimension, MLP depth, learning rate, dropout

3. **Train NCF** -- `src/training/train_ncf.py`
   - Training loop with BPR loss (Bayesian Personalized Ranking) for implicit feedback
   - Or MSE loss for explicit rating prediction
   ```python
   def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
       """Bayesian Personalized Ranking loss."""
       return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
   ```
   - Use AdamW optimizer with cosine annealing scheduler
   - Validate every epoch with NDCG@10 on the validation set
   - Early stopping based on validation NDCG

4. **Hybrid Model** -- `src/model/hybrid.py`
   - Combine collaborative filtering embeddings with content features
   - Content features: genre vectors, movie year, average rating, popularity
   - Architecture: NCF embeddings concatenated with content features, fed through
     additional MLP layers
   ```python
   class HybridRecommender(nn.Module):
       def __init__(
           self,
           n_users: int,
           n_items: int,
           n_content_features: int,
           embed_dim: int = 64,
       ):
           super().__init__()
           self.ncf = NeuralCollaborativeFiltering(n_users, n_items, embed_dim)
           self.content_mlp = nn.Sequential(
               nn.Linear(n_content_features, 32),
               nn.ReLU(),
               nn.Linear(32, 16),
           )
           # Replace NCF output layer with fusion layer
           self.ncf.output = nn.Identity()
           fusion_dim = embed_dim + 32 + 16  # GMF + MLP + content
           self.fusion = nn.Sequential(
               nn.Linear(fusion_dim, 64),
               nn.ReLU(),
               nn.Dropout(0.2),
               nn.Linear(64, 1),
           )

       def forward(self, user_ids, item_ids, content_features):
           ncf_features = self.ncf(user_ids, item_ids)
           content_out = self.content_mlp(content_features)
           fused = torch.cat([ncf_features, content_out], dim=-1)
           return self.fusion(fused).squeeze(-1)
   ```
   - The hybrid model should handle cold-start items better because it can fall
     back on content features when collaborative signals are weak

5. **Offline Evaluation with Cross-Validation** -- `src/training/evaluate.py`
   - Implement leave-one-out evaluation (standard in RecSys literature)
   - For each user, hold out the last interaction, predict rankings, measure metrics
   - Compare all three models on the same evaluation protocol

6. **Run Experiments** (track all in MLflow)

   | Experiment | Model | Key Change | Expected Result |
   |-----------|-------|-----------|----------------|
   | Baseline | SVD (100 factors) | -- | RMSE ~0.86, NDCG@10 ~0.30 |
   | SVD-200 | SVD (200 factors) | More latent factors | Slight improvement |
   | NCF-64 | NCF (embed=64) | Neural approach | NDCG@10 ~0.33 |
   | NCF-128 | NCF (embed=128) | Larger embeddings | Possible overfitting |
   | NCF-BPR | NCF + BPR loss | Ranking loss | Better ranking metrics |
   | Hybrid | Hybrid (NCF + content) | Add content features | Best cold-start, NDCG@10 ~0.36 |

7. **Pick the best model** using MLflow UI
   - Compare runs side by side on NDCG@10, Precision@10, and training time
   - Promote best model to "Production" in MLflow registry
   - Keep SVD as a fallback model (fast, no GPU needed)

### Skills Learned

- Matrix factorization for recommendations (SVD)
- Neural collaborative filtering with embeddings
- Hybrid model design (combining collaborative and content signals)
- BPR loss for implicit feedback learning
- Negative sampling strategies
- Leave-one-out evaluation protocol

---

## Phase 4: Evaluation

**Duration:** 3-4 days
**Objective:** Evaluate recommendation quality beyond simple accuracy metrics.

### Tasks

1. **Ranking Metrics** -- `src/training/evaluate.py`
   - **Precision@K:** of the top-K recommended items, how many did the user actually interact with?
   - **Recall@K:** of all items the user interacted with, how many appeared in the top-K?
   - **NDCG@K (Normalized Discounted Cumulative Gain):** measures ranking quality --
     relevant items appearing earlier in the list get higher scores
   ```python
   def ndcg_at_k(predicted_ranking: list[int], actual_items: set[int], k: int) -> float:
       """Calculate NDCG@K for a single user."""
       dcg = 0.0
       for i, item in enumerate(predicted_ranking[:k]):
           if item in actual_items:
               dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
       # Ideal DCG: all relevant items at the top
       ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_items), k)))
       return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

   def precision_at_k(predicted_ranking: list[int], actual_items: set[int], k: int) -> float:
       """Fraction of top-K recommendations that are relevant."""
       top_k = set(predicted_ranking[:k])
       return len(top_k & actual_items) / k

   def recall_at_k(predicted_ranking: list[int], actual_items: set[int], k: int) -> float:
       """Fraction of relevant items that appear in top-K."""
       top_k = set(predicted_ranking[:k])
       return len(top_k & actual_items) / len(actual_items) if actual_items else 0.0
   ```

2. **Beyond-Accuracy Metrics** -- critical for real recommendation systems
   - **Coverage:** what fraction of the total catalog does the system ever recommend?
     Low coverage means the system only recommends popular items.
   - **Diversity (Intra-List Similarity):** how different are the recommended items from
     each other? Users do not want 10 very similar movies.
     ```python
     def intra_list_diversity(recommended_items: list[int], item_features: dict) -> float:
         """Average pairwise distance between recommended items."""
         if len(recommended_items) < 2:
             return 0.0
         distances = []
         for i in range(len(recommended_items)):
             for j in range(i + 1, len(recommended_items)):
                 fi = item_features[recommended_items[i]]
                 fj = item_features[recommended_items[j]]
                 distances.append(1 - cosine_similarity(fi, fj))
         return np.mean(distances)
     ```
   - **Novelty:** are we recommending obscure items or just blockbusters?
     Measured by the negative log of item popularity.
   - **Serendipity:** are recommendations surprising yet relevant?

3. **Popularity Bias Analysis**
   - Split items into popularity buckets (head, mid, tail)
   - Measure: does the model disproportionately recommend head items?
   - Compare: SVD vs NCF vs Hybrid on tail-item recommendation quality
   - This is a major real-world problem -- popularity bias creates filter bubbles

4. **Cold Start Evaluation**
   - Evaluate separately on users with < 5 ratings vs users with 50+ ratings
   - Evaluate separately on items with < 10 ratings vs items with 100+ ratings
   - The hybrid model should significantly outperform pure collaborative models
     on cold-start scenarios

5. **A/B Testing Framework Design** -- `src/ab_testing/`
   - Design the framework now, implement it with serving (Phase 5)
   - Define: what business metrics matter? (click-through rate, watch time, diversity
     of consumption)
   - Key insight: **ML metrics (NDCG) do not always correlate with business metrics**
     A model with higher NDCG might recommend safer, more obvious choices, leading
     to lower user engagement

6. **Generate Evaluation Report**
   - Create comprehensive plots: metric comparison across models, per-user-segment
     performance, popularity bias visualization
   - Log all artifacts to MLflow
   - Document findings and model selection rationale

### Skills Learned

- Ranking metrics for recommendation systems (NDCG, Precision@K, Recall@K)
- Beyond-accuracy evaluation (diversity, coverage, novelty, serendipity)
- Popularity bias analysis and mitigation
- Cold start evaluation methodology
- The critical gap between ML metrics and business metrics

---

## Phase 5: API & Serving

**Duration:** 4-5 days
**Objective:** Build a production recommendation API with Redis caching and a Streamlit frontend.

### Tasks

1. **Define API Schemas** -- `src/serving/schemas.py`
   ```python
   class RecommendationRequest(BaseModel):
       user_id: int
       n_recommendations: int = 10
       exclude_seen: bool = True
       genre_filter: list[str] | None = None

   class MovieRecommendation(BaseModel):
       movie_id: int
       title: str
       genres: list[str]
       predicted_score: float
       popularity_rank: int

   class RecommendationResponse(BaseModel):
       user_id: int
       recommendations: list[MovieRecommendation]
       model_version: str
       served_from_cache: bool
       latency_ms: float
       ab_test_group: str | None = None

   class SimilarMoviesRequest(BaseModel):
       movie_id: int
       n_similar: int = 10

   class SimilarMoviesResponse(BaseModel):
       movie_id: int
       title: str
       similar_movies: list[MovieRecommendation]
       latency_ms: float
   ```

2. **Redis Caching Layer** -- `src/serving/cache.py`
   - Cache precomputed recommendations per user (TTL: 1 hour)
   - Cache similar-movies results per movie (TTL: 24 hours -- these change less)
   - Cache popular movies as fallback for cold-start users (TTL: 6 hours)
   ```python
   class RecommendationCache:
       def __init__(self, redis_client: redis.Redis, default_ttl: int = 3600):
           self.redis = redis_client
           self.default_ttl = default_ttl

       async def get_recommendations(self, user_id: int) -> list[dict] | None:
           key = f"rec:user:{user_id}"
           cached = self.redis.get(key)
           if cached:
               return json.loads(cached)
           return None

       async def set_recommendations(
           self, user_id: int, recs: list[dict], ttl: int | None = None
       ) -> None:
           key = f"rec:user:{user_id}"
           self.redis.setex(key, ttl or self.default_ttl, json.dumps(recs))

       async def invalidate_user(self, user_id: int) -> None:
           """Invalidate when user rates a new movie."""
           self.redis.delete(f"rec:user:{user_id}")
   ```

3. **Build FastAPI Application** -- `src/serving/app.py`
   - `POST /recommend` -- get personalized recommendations for a user
   - `POST /similar` -- get similar movies for a given movie
   - `POST /rate` -- record a new rating (invalidates cache)
   - `GET /popular` -- get most popular movies (cold-start fallback)
   - `GET /health` -- health check (model loaded, Redis connected, DB connected)
   - `GET /metrics` -- Prometheus metrics endpoint
   - Load model once at startup, keep embeddings in memory

4. **Recommendation Logic** -- `src/serving/recommend.py`
   - Serving flow with cache-first strategy:
     1. Check Redis cache for precomputed recommendations
     2. If cache miss: compute recommendations from model
     3. Store result in Redis for future requests
     4. Return recommendations
   - For cold-start users (no history): return popular movies filtered by any
     genre preference they have expressed
   - For similar movies: use item embedding cosine similarity (precomputed at
     model load time)
   ```python
   async def get_recommendations(
       user_id: int,
       n_recs: int,
       cache: RecommendationCache,
       model: HybridRecommender,
       db: Database,
   ) -> tuple[list[MovieRecommendation], bool]:
       """Get recommendations with cache-first strategy."""
       # Try cache first
       cached = await cache.get_recommendations(user_id)
       if cached:
           return [MovieRecommendation(**r) for r in cached[:n_recs]], True

       # Cache miss -- compute recommendations
       user_history = await db.get_user_ratings(user_id)
       if len(user_history) < 3:
           # Cold start: return popular movies
           recs = await db.get_popular_movies(n_recs)
       else:
           # Score all unseen items
           seen_items = {r.movie_id for r in user_history}
           scores = model.predict_all_items(user_id)
           # Filter seen items, sort by score, take top-N
           candidates = [
               (item_id, score) for item_id, score in enumerate(scores)
               if item_id not in seen_items
           ]
           candidates.sort(key=lambda x: x[1], reverse=True)
           recs = await db.enrich_recommendations(candidates[:n_recs])

       # Cache the result
       await cache.set_recommendations(user_id, [r.dict() for r in recs])
       return recs, False
   ```

5. **PostgreSQL Database Layer** -- `src/serving/db.py`
   - Store user profiles, rating history, and movie metadata
   - Use SQLAlchemy with async support
   - Key queries: get user ratings, get movie details, record new rating
   - Seed the database with MovieLens data on first startup

6. **Build Streamlit Frontend** -- `src/frontend/app.py`
   - User selection dropdown (or ID input)
   - Display personalized recommendation cards with movie posters (use TMDB API
     for poster images)
   - "Similar Movies" feature: click a movie to see similar ones
   - Show which A/B test group the user is in
   - Display recommendation diversity metrics
   - Rating input: let users rate movies and see recommendations update

7. **API Tests** -- `tests/`
   - Unit tests: schema validation, cache hit/miss logic, cold start detection
   - Integration tests: full recommendation flow, cache invalidation on new rating
   - Load tests: verify < 100ms for cached requests with 200 concurrent users

### Skills Learned

- Cache-first serving architecture for ML (Redis)
- Designing recommendation APIs (personalized + similar items)
- PostgreSQL for user data in ML systems
- Cold-start handling strategies
- Building interactive ML frontends with Streamlit

---

## Phase 6: Containerization

**Duration:** 2-3 days
**Objective:** Package the entire recommendation stack into Docker containers.

### Tasks

1. **API Dockerfile** -- `docker/Dockerfile.api`
   - Multi-stage build: install dependencies, then copy only runtime files
   - Pre-download and bake the model into the image
   - Expose port 8000

2. **Frontend Dockerfile** -- `docker/Dockerfile.frontend`
   - Single-stage build for Streamlit
   - Expose port 8501

3. **docker-compose.yaml** -- orchestrate all services
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports: ["8000:8000"]
       environment:
         - MODEL_PATH=/app/models/hybrid_model.pt
         - REDIS_URL=redis://redis:6379
         - DATABASE_URL=postgresql://user:pass@postgres:5432/recommendations
       depends_on:
         redis:
           condition: service_healthy
         postgres:
           condition: service_healthy

     frontend:
       build:
         context: .
         dockerfile: docker/Dockerfile.frontend
       ports: ["8501:8501"]
       environment:
         - API_URL=http://api:8000
       depends_on: [api]

     redis:
       image: redis:7-alpine
       ports: ["6379:6379"]
       volumes: ["redis_data:/data"]
       healthcheck:
         test: ["CMD", "redis-cli", "ping"]
         interval: 10s
         retries: 3

     postgres:
       image: postgres:16-alpine
       ports: ["5432:5432"]
       environment:
         - POSTGRES_USER=user
         - POSTGRES_PASSWORD=pass
         - POSTGRES_DB=recommendations
       volumes: ["postgres_data:/var/lib/postgresql/data"]
       healthcheck:
         test: ["CMD-SHELL", "pg_isready -U user"]
         interval: 10s
         retries: 3

     prometheus:
       image: prom/prometheus
       volumes: ["./prometheus:/etc/prometheus"]
       ports: ["9090:9090"]

     grafana:
       image: grafana/grafana
       volumes: ["./grafana:/etc/grafana/provisioning"]
       ports: ["3000:3000"]
       depends_on: [prometheus]

   volumes:
     redis_data:
     postgres_data:
   ```

4. **Database Seeding Script** -- `scripts/seed_db.sh`
   - Load MovieLens movie metadata into PostgreSQL on first start
   - Create indexes on user_id and movie_id for fast lookups

5. **Cache Warming Script** -- `scripts/warm_cache.sh`
   - Pre-populate Redis with recommendations for the most active users
   - Pre-populate similar-movies for the most popular movies
   - This reduces cold-start latency after deployment

6. **Verify the full stack locally**
   - `docker compose up` -- all services start and connect
   - Frontend can fetch recommendations from API
   - API reads from Redis cache and PostgreSQL
   - Prometheus scrapes metrics, Grafana shows dashboard

### Skills Learned

- Multi-service Docker orchestration (API + DB + Cache + Frontend)
- Service health checks and dependency management
- Database seeding and cache warming strategies
- Container networking for ML systems

---

## Phase 7: Testing & CI/CD

**Duration:** 2-3 days
**Objective:** Automate quality checks for the recommendation pipeline.

### Tasks

1. **Write comprehensive tests**

   **Unit tests:**
   ```
   test_dataset.py
   ├── test_interaction_dataset_negative_sampling
   ├── test_user_item_encoding_is_contiguous
   └── test_temporal_split_no_future_leakage

   test_models.py
   ├── test_ncf_output_shape
   ├── test_ncf_embedding_dimensions
   ├── test_hybrid_handles_content_features
   └── test_svd_predictions_in_valid_range

   test_cache.py
   ├── test_cache_hit_returns_stored_recommendations
   ├── test_cache_miss_returns_none
   ├── test_cache_invalidation_on_new_rating
   └── test_cache_ttl_expiry

   test_ab_router.py
   ├── test_routing_is_deterministic_for_same_user
   ├── test_traffic_split_matches_config
   └── test_new_users_get_assigned_group
   ```

   **Integration tests:**
   ```
   test_api.py
   ├── test_recommend_returns_correct_schema
   ├── test_recommend_excludes_seen_movies
   ├── test_similar_movies_returns_relevant_genres
   ├── test_cold_start_user_gets_popular_movies
   ├── test_rate_invalidates_cache
   └── test_concurrent_recommendation_requests
   ```

   **ML-specific tests:**
   ```
   ├── test_no_future_data_leakage_in_splits
   ├── test_recommendations_change_after_new_rating
   ├── test_model_is_deterministic_with_same_seed
   └── test_similar_movies_are_genre_relevant
   ```

2. **Set up CI pipeline** -- `.github/workflows/ci.yaml`
   - Lint with ruff, type check with mypy
   - Run unit tests (mock Redis and PostgreSQL)
   - Run integration tests with Docker Compose (spin up Redis + PostgreSQL)
   - Build Docker images
   - Smoke test: start stack, request recommendations, verify response

3. **Set up CD pipeline** -- `.github/workflows/cd.yaml`
   - Build and push Docker images on merge to main
   - Deploy to staging, run smoke tests
   - Deploy to production on manual approval

### Skills Learned

- Testing cache-dependent ML systems
- Mocking external services (Redis, PostgreSQL) in tests
- Testing recommendation quality properties
- CI/CD for multi-service ML applications

---

## Phase 8: Deployment

**Duration:** 2-3 days
**Objective:** Deploy the recommendation system to the cloud.

### Tasks

1. **Choose a platform**
   - Railway or GCP Cloud Run for the API
   - Managed Redis (Railway Redis or GCP Memorystore)
   - Managed PostgreSQL (Railway PostgreSQL or Cloud SQL)
   - The key challenge: you now have stateful services (Redis, PostgreSQL), not
     just a stateless API

2. **Deploy the stack**
   - Configure environment variables for each service
   - Set up managed Redis and PostgreSQL instances
   - Deploy API container, verify health endpoint
   - Deploy frontend, point at API URL
   - Run database seeding and cache warming scripts

3. **Load testing**
   - Use `locust` to simulate 200 concurrent users requesting recommendations
   - Target: cached requests < 100ms p95, uncached < 500ms p95
   - Identify bottlenecks: is it Redis, PostgreSQL, model inference, or network?

### Skills Learned

- Deploying stateful ML services (database + cache)
- Managed cloud services for ML infrastructure
- Load testing recommendation systems

---

## Phase 9: Monitoring

**Duration:** 2-3 days
**Objective:** Monitor recommendation quality, system health, and user engagement.

### Tasks

1. **System Metrics** -- `src/monitoring/metrics.py`
   - `recommendation_requests_total` -- counter by endpoint and cache status
   - `recommendation_latency_seconds` -- histogram
   - `cache_hit_rate` -- gauge (should be > 80% after warming)
   - `active_users` -- gauge (unique users in last hour)

2. **Recommendation Quality Metrics** -- `src/monitoring/quality.py`
   - Track the distribution of predicted scores over time
   - Monitor catalog coverage: how many unique movies are being recommended?
   - Alert if coverage drops below threshold (sign of model degradation)

3. **Popularity Bias Monitoring** -- `src/monitoring/bias.py`
   - Track: what percentage of recommendations come from the top 1% most popular movies?
   - Alert if popularity concentration increases (the model is becoming less diverse)
   - Compare bias across A/B test groups

4. **Grafana Dashboard** -- `grafana/dashboards/rec_monitoring.json`
   - Row 1: Request rate, error rate, latency percentiles, cache hit rate
   - Row 2: Recommendation quality (coverage, diversity, average predicted score)
   - Row 3: Popularity bias tracking, genre distribution of recommendations
   - Row 4: A/B test metrics comparison (if active)

5. **Alerting Rules**
   - Cache hit rate < 50% for 10 minutes -> alert (cache may be down)
   - Catalog coverage < 5% -> alert (model recommending same items to everyone)
   - Error rate > 5% for 5 minutes -> alert
   - Latency p95 > 1 second -> alert

### Skills Learned

- Monitoring recommendation-specific metrics
- Popularity bias tracking
- Cache performance monitoring
- Multi-dimensional alerting for ML systems

---

## Phase 10: A/B Testing Framework

**Duration:** 3-4 days
**Objective:** Implement a champion/challenger framework to safely test new recommendation models.

### Tasks

1. **A/B Test Router** -- `src/ab_testing/router.py`
   - Deterministically assign users to test groups based on user_id hash
   - Support configurable traffic splits (e.g., 90% champion, 10% challenger)
   ```python
   class ABTestRouter:
       def __init__(self, config: ABTestConfig):
           self.config = config

       def get_model_for_user(self, user_id: int) -> str:
           """Deterministically route user to a model variant."""
           hash_val = hashlib.md5(str(user_id).encode()).hexdigest()
           bucket = int(hash_val[:8], 16) % 100
           cumulative = 0
           for variant, percentage in self.config.traffic_split.items():
               cumulative += percentage
               if bucket < cumulative:
                   return variant
           return self.config.default_variant
   ```

2. **Engagement Tracker** -- `src/ab_testing/tracker.py`
   - Track per-variant: impressions, clicks (user selected a recommended movie),
     ratings submitted, session duration
   - Store events in PostgreSQL for offline analysis
   - Key business metric: **click-through rate (CTR)** on recommendations

3. **Statistical Significance Analyzer** -- `src/ab_testing/analyzer.py`
   - Implement chi-squared test for CTR comparison
   - Calculate confidence intervals and p-values
   - Report minimum sample size needed for statistical significance
   ```python
   def analyze_ab_test(
       champion_clicks: int, champion_impressions: int,
       challenger_clicks: int, challenger_impressions: int,
       significance_level: float = 0.05,
   ) -> ABTestResult:
       """Analyze A/B test results for statistical significance."""
       champion_ctr = champion_clicks / champion_impressions
       challenger_ctr = challenger_clicks / challenger_impressions
       # Two-proportion z-test
       p_pooled = (champion_clicks + challenger_clicks) / (
           champion_impressions + challenger_impressions
       )
       se = np.sqrt(p_pooled * (1 - p_pooled) * (
           1/champion_impressions + 1/challenger_impressions
       ))
       z_stat = (challenger_ctr - champion_ctr) / se
       p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
       return ABTestResult(
           champion_ctr=champion_ctr,
           challenger_ctr=challenger_ctr,
           lift=(challenger_ctr - champion_ctr) / champion_ctr,
           p_value=p_value,
           is_significant=p_value < significance_level,
       )
   ```

4. **A/B Test Configuration** -- `configs/ab_test_config.yaml`
   ```yaml
   ab_tests:
     ncf_vs_hybrid:
       champion: ncf_v2
       challenger: hybrid_v1
       traffic_split:
         ncf_v2: 80
         hybrid_v1: 20
       start_date: "2026-04-01"
       min_sample_size: 10000
       primary_metric: click_through_rate
       guardrail_metrics:
         - latency_p95_ms: 500
         - error_rate: 0.05
   ```

5. **Dashboard Integration**
   - Add A/B test panel to Grafana: real-time CTR comparison, sample sizes,
     statistical significance indicator
   - Daily email/Slack summary of A/B test progress

### Skills Learned

- A/B testing framework design for ML systems
- Statistical significance testing for business metrics
- Champion/challenger deployment patterns
- The gap between ML metrics and business metrics
- Safe rollout strategies for new models

---

## Timeline Summary

```
Week 1  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 1: Setup & Design Doc    (2 days)
        Phase 2: Data Pipeline         (4 days)

Week 2  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 2: Data Pipeline cont.   (1 day)
        Phase 3: Model Development     (4 days)

Week 3  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 3: Model Development     (3 days)
        Phase 4: Evaluation            (2 days)

Week 4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 4: Evaluation cont.      (2 days)
        Phase 5: API & Serving         (3 days)

Week 5  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 5: API & Serving cont.   (2 days)
        Phase 6: Containerization      (3 days)

Week 6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 7: Testing & CI/CD       (3 days)
        Phase 8: Deployment            (2 days)

Week 7  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 8: Deployment cont.      (1 day)
        Phase 9: Monitoring            (3 days)
        Phase 10: A/B Testing          (1 day)

Week 8  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 10: A/B Testing cont.    (3 days)
        Buffer / catch-up              (2 days)
```

**Total: ~38 days (8 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] Designing recommendation system architectures
- [ ] Working with user-item interaction data (sparse matrices)
- [ ] Exploratory data analysis for recommendation datasets
- [ ] Temporal train/val/test splitting (avoiding data leakage)
- [ ] Matrix factorization (SVD) for collaborative filtering
- [ ] Neural Collaborative Filtering with embedding layers (PyTorch)
- [ ] Hybrid recommendation models (collaborative + content)
- [ ] BPR loss and negative sampling for implicit feedback
- [ ] Ranking evaluation metrics (NDCG@K, Precision@K, Recall@K)
- [ ] Beyond-accuracy metrics (diversity, coverage, novelty)
- [ ] Popularity bias analysis and monitoring
- [ ] Cold start detection and handling strategies
- [ ] Redis caching for ML serving (cache-first architecture)
- [ ] PostgreSQL for user data and rating storage
- [ ] Building recommendation APIs with FastAPI
- [ ] Streamlit frontend for interactive recommendations
- [ ] Multi-service Docker orchestration (API + Redis + PostgreSQL)
- [ ] Cache warming and invalidation strategies
- [ ] A/B testing framework for ML models
- [ ] Statistical significance testing (z-test, confidence intervals)
- [ ] Champion/challenger deployment pattern
- [ ] Business metrics vs ML metrics (CTR vs NDCG)
- [ ] Monitoring recommendation quality in production
- [ ] Experiment tracking with MLflow
- [ ] CI/CD for multi-service ML applications

---

## Key Differences from Classification Projects

This project introduces several concepts not found in standard classification:

| Concept | Classification | Recommendation |
|---------|---------------|----------------|
| Data shape | Features -> Label | User x Item matrix (sparse) |
| Model output | Class probability | Ranked list of items |
| Evaluation | Accuracy, F1 | NDCG@K, Coverage, Diversity |
| Serving | Stateless inference | Cache-first, user state needed |
| Infrastructure | API only | API + Redis + PostgreSQL |
| Testing | A model is good or bad | Champion vs challenger (A/B test) |
| Business metric | Accuracy | Click-through rate, engagement |
| Cold start | N/A | Major design challenge |
| Bias | Class imbalance | Popularity bias |

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Initialize the project
mkdir movie-recommendation-engine && cd movie-recommendation-engine
git init

# 2. Create the folder structure
mkdir -p configs data/{raw,processed,splits} notebooks \
  src/{data,model,training,serving,ab_testing,monitoring,frontend} \
  tests/{unit,integration} docker .github/workflows \
  grafana/dashboards prometheus scripts

# 3. Start writing DESIGN_DOC.md
# 4. Install dependencies: pip install scikit-surprise torch fastapi redis
```
