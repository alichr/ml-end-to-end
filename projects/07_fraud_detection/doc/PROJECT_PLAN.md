# Project 7: Real-Time Fraud Detection (Hard)

## Goal

Build a **production-grade streaming ML pipeline** that detects fraudulent financial
transactions in real-time with sub-100ms latency. The system combines a feature store
(Feast) for consistent feature computation, an XGBoost ensemble for classification,
and Apache Kafka for streaming ingestion. By the end, you will understand how fraud
detection works at banks and fintech companies -- from handling extreme class imbalance
to building streaming pipelines with champion/challenger model deployment.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why fraud detection? | Fraud detection is one of the highest-impact ML applications in industry. Every bank, payment processor, and e-commerce company needs it. It pays extremely well. |
| Why streaming/real-time? | Batch fraud detection is useless -- by the time you detect fraud in a nightly batch job, the money is gone. Real-time scoring (<100ms) is a hard requirement. |
| Why is this hard? | Extreme class imbalance (0.1-1% fraud), adversarial actors who actively evolve, concept drift, latency constraints, and the cost asymmetry between false positives and false negatives. |
| What new skills will I learn? | Streaming architectures (Kafka), feature stores (Feast), handling extreme class imbalance, cost-sensitive evaluation, walk-forward validation, and champion/challenger deployment. |
| How is this different from classification? | In standard classification, you optimize accuracy. In fraud detection, you optimize a cost function where missing fraud costs $1000 and blocking a legitimate transaction costs $10. Accuracy is meaningless when 99% of transactions are legitimate. |

---

## Architecture Overview

```
    Transaction       ┌──────────────┐       ┌──────────────┐
    Stream ──────────▶│ Apache Kafka │──────▶│ Feature      │
    (card swipes,     │ (Ingestion)  │       │ Computation  │
     wire transfers)  └──────────────┘       └──────┬───────┘
                                                    │
                           ┌────────────────────────┤
                           ▼                        ▼
                    ┌──────────────┐        ┌──────────────┐
                    │    Feast     │        │  Real-Time   │
                    │ Feature Store│◀──────▶│  Scorer      │
                    │(Online+Offline)│       │  (<100ms)    │
                    └──────────────┘        └──────┬───────┘
                                                   │
                              ┌─────────────────────┼──────────────┐
                              ▼                     ▼              ▼
                       ┌────────────┐       ┌────────────┐  ┌──────────┐
                       │  FastAPI   │       │ Champion/  │  │  Redis   │
                       │    API     │       │ Challenger │  │  Cache   │
                       └─────┬──────┘       └────────────┘  └──────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
         ┌──────────┐ ┌──────────┐ ┌──────────┐
         │  MLflow  │ │Prometheus│ │ Alert    │
         │ Registry │ │ Metrics  │ │ System   │
         └──────────┘ └────┬─────┘ └──────────┘
                           ▼
                     ┌──────────┐
                     │ Grafana  │
                     │Dashboard │
                     └──────────┘

Everything runs in Docker. Kafka + Feature Store + Redis + API + Monitoring.
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | ML ecosystem standard |
| Streaming | Apache Kafka | Industry standard for event streaming, used at every major fintech |
| Feature Store | Feast | Open-source feature store with online+offline serving, growing adoption |
| ML Framework | XGBoost | Best-in-class for tabular data, fast inference, handles imbalance well |
| Deep Learning | PyTorch | Sequence models for transaction history patterns |
| Cache | Redis | Sub-millisecond feature lookups for real-time scoring |
| API Framework | FastAPI | Async support essential for streaming workloads |
| Experiment Tracking | MLflow | Track experiments, model registry, champion/challenger |
| Containerization | Docker + docker-compose | Orchestrate Kafka, Feast, Redis, API, monitoring |
| CI/CD | GitHub Actions | Automated testing and deployment |
| Monitoring | Prometheus + Grafana | Track model performance, latency, drift |
| Imbalance Handling | imbalanced-learn (SMOTE) | Oversampling for training |
| Testing | pytest | Unit and integration testing |
| Data Processing | pandas, polars | Feature engineering on tabular data |

---

## Project Structure

```
fraud-detection/
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
│   ├── serve_config.yaml            # Serving configuration
│   ├── feature_store/               # Feast feature definitions
│   │   ├── feature_store.yaml       # Feast config
│   │   └── features.py              # Feature view definitions
│   ├── kafka_config.yaml            # Kafka topics, partitions, consumer groups
│   └── champion_challenger.yaml     # Model deployment config
│
├── data/                            # Git-ignored, DVC-tracked
│   ├── raw/                         # Original IEEE-CIS dataset
│   │   ├── train_transaction.csv    # Transaction features
│   │   ├── train_identity.csv       # Identity features
│   │   ├── test_transaction.csv
│   │   └── test_identity.csv
│   ├── processed/                   # Cleaned, engineered features
│   └── splits/                      # Walk-forward validation splits
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Class imbalance, feature distributions
│   ├── 02_feature_engineering.ipynb # Interactive feature development
│   ├── 03_imbalance_analysis.ipynb  # SMOTE, class weights, focal loss comparison
│   ├── 04_model_experiments.ipynb   # Model training and comparison
│   └── 05_cost_analysis.ipynb       # Cost-sensitive evaluation
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py              # Download IEEE-CIS dataset
│   │   ├── preprocess.py            # Clean, handle missing values
│   │   ├── features.py              # Feature engineering pipeline
│   │   ├── velocity.py              # Velocity features (transactions per time window)
│   │   ├── aggregation.py           # Aggregation features (rolling stats)
│   │   ├── graph.py                 # Graph features (shared device/card networks)
│   │   └── split.py                 # Walk-forward temporal validation split
│   │
│   ├── feature_store/
│   │   ├── __init__.py
│   │   ├── definitions.py           # Feast feature view definitions
│   │   ├── materialize.py           # Materialize features to online store
│   │   └── serve.py                 # Online feature retrieval
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py         # XGBoost classifier
│   │   ├── sequence_model.py        # LSTM/Transformer for transaction sequences
│   │   ├── ensemble.py              # Stacking ensemble (XGBoost + sequence)
│   │   ├── imbalance.py             # SMOTE, focal loss, class weight utilities
│   │   └── export.py                # Export models for serving
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                 # Training orchestration
│   │   ├── evaluate.py              # Cost-sensitive evaluation
│   │   ├── threshold.py             # Optimal threshold selection
│   │   ├── walk_forward.py          # Walk-forward validation
│   │   └── callbacks.py             # Custom callbacks
│   │
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── producer.py              # Kafka producer (simulate transactions)
│   │   ├── consumer.py              # Kafka consumer (real-time scoring)
│   │   ├── feature_compute.py       # Real-time feature computation
│   │   └── pipeline.py              # End-to-end streaming pipeline
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application
│   │   ├── schemas.py               # Request/response Pydantic models
│   │   ├── score.py                 # Transaction scoring logic
│   │   ├── champion_challenger.py   # Model routing (champion vs challenger)
│   │   └── investigate.py           # Batch investigation endpoint
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Prometheus metric definitions
│   │   ├── drift.py                 # Feature drift detection
│   │   ├── model_quality.py         # False positive rate tracking
│   │   └── alerts.py                # Alert rules and notification
│   │
│   └── frontend/
│       └── app.py                   # Streamlit dashboard
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py         # Feature computation correctness
│   │   ├── test_velocity.py         # Velocity feature edge cases
│   │   ├── test_model.py            # Model output shape, probability range
│   │   ├── test_threshold.py        # Threshold optimization
│   │   ├── test_schemas.py          # API schema validation
│   │   └── test_champion.py         # Champion/challenger routing
│   ├── integration/
│   │   ├── test_api.py              # Full scoring pipeline
│   │   ├── test_streaming.py        # Kafka producer/consumer
│   │   ├── test_feature_store.py    # Feast online/offline serving
│   │   └── test_training.py         # Training runs without error
│   └── conftest.py                  # Shared fixtures
│
├── docker/
│   ├── Dockerfile.api               # Scoring API container
│   ├── Dockerfile.consumer          # Kafka consumer container
│   ├── Dockerfile.frontend          # Streamlit container
│   └── Dockerfile.training          # Training environment
│
├── docker-compose.yaml              # Kafka + Redis + Feast + API + Monitoring
│
├── .github/
│   └── workflows/
│       ├── ci.yaml                  # Lint -> Test -> Build on PR
│       └── cd.yaml                  # Deploy on merge to main
│
├── grafana/
│   └── dashboards/
│       └── fraud_monitoring.json    # Fraud detection dashboard
│
├── prometheus/
│   └── prometheus.yml               # Scrape config
│
└── scripts/
    ├── setup.sh                     # One-command project setup
    ├── generate_transactions.sh     # Generate synthetic transaction stream
    ├── materialize_features.sh      # Materialize Feast features
    └── run_backtest.sh              # Run walk-forward backtesting
```

---

## Phase 1: Project Setup & Design Doc

**Duration:** 1-2 days
**Objective:** Define the fraud detection problem with its unique constraints.

### Tasks

1. **Write `DESIGN_DOC.md`**
   - **Problem statement:** "Given a financial transaction (card swipe, wire transfer,
     online purchase), predict in real-time (<100ms) whether it is fraudulent, while
     minimizing the cost of both missed fraud and false alarms."
   - **Success criteria:**
     - AUC-PR >= 0.70 (precision-recall, NOT ROC -- see Phase 5 for why)
     - False positive rate < 2% (blocking legitimate transactions costs money)
     - Detection rate >= 80% for high-value fraud (> $500)
     - End-to-end latency < 100ms (from transaction to score)
     - System handles 1000 transactions per second
   - **Cost model:** (this is critical and unique to fraud detection)
     - False negative (missed fraud): average loss = $500
     - False positive (blocked legitimate): average cost = $15 (customer friction,
       support calls, lost revenue)
     - True positive (caught fraud): saves $500
     - True negative (approved legitimate): $0
   - **Out of scope:** real bank integrations, PCI compliance, customer-facing UI
   - **Risks:** extreme class imbalance (0.3% fraud rate), concept drift (fraudsters
     adapt), feature computation latency, seasonal patterns

2. **Initialize the repository**
   - `git init`, create `.gitignore`
   - Create `pyproject.toml`:
     ```toml
     [project]
     name = "fraud-detection"
     version = "0.1.0"
     requires-python = ">=3.11"
     dependencies = [
         "xgboost>=2.0",
         "torch>=2.0",
         "scikit-learn>=1.3",
         "imbalanced-learn>=0.11",
         "feast>=0.34",
         "confluent-kafka>=2.3",
         "fastapi>=0.100",
         "uvicorn>=0.23",
         "redis>=5.0",
         "mlflow>=2.8",
         "streamlit>=1.28",
         "pandas>=2.0",
         "polars>=0.19",
         "prometheus-client>=0.19",
         "pydantic>=2.0",
         "shap>=0.43",
     ]
     ```

3. **Create the folder structure** (as shown above)

4. **Set up development environment**
   - Install and verify Kafka locally (or use Docker from the start)
   - Install Feast and verify with a toy feature store
   - Install Redis and verify connectivity
   - This project has more infrastructure dependencies than any previous one

### Skills Learned

- Designing cost-sensitive ML systems
- Understanding fraud detection constraints (latency, cost asymmetry)
- Setting up streaming infrastructure (Kafka + Feast + Redis)

---

## Phase 2: Data Pipeline

**Duration:** 5-6 days
**Objective:** Download, explore, and engineer features from the IEEE-CIS fraud detection dataset.

### Tasks

1. **Download IEEE-CIS Dataset** -- `src/data/download.py`
   - Download from Kaggle: https://www.kaggle.com/c/ieee-fraud-detection
   - Files: `train_transaction.csv` (590K rows, 394 columns), `train_identity.csv`
     (144K rows, 41 columns)
   - These files are large and have many anonymous features (V1-V339)
   - Script should be idempotent and validate checksums

2. **Exploratory Data Analysis** -- `notebooks/01_eda.ipynb`

   The EDA for fraud detection is fundamentally different from other projects.
   The key insight: **you cannot treat this as a normal classification problem.**

   - **Class imbalance analysis:** this is the defining challenge
     ```python
     fraud_rate = train["isFraud"].mean()
     print(f"Fraud rate: {fraud_rate:.4%}")
     # Expect: ~3.5% in IEEE-CIS (which is actually GENEROUS compared to
     # real-world fraud rates of 0.1-0.5%)

     print(f"Legitimate: {(~train['isFraud'].astype(bool)).sum():,}")
     print(f"Fraudulent: {train['isFraud'].sum():,}")
     # If you train a model that always predicts "not fraud", it gets 96.5% accuracy.
     # This is why accuracy is USELESS for fraud detection.
     ```
   - **Transaction amount distribution:** fraud transactions tend to cluster at
     certain amounts. Plot distributions separately for fraud vs legitimate.
   - **Temporal patterns:** plot fraud rate over time. Is there a trend? Seasonal
     patterns? Specific hours or days with more fraud?
     ```python
     # Convert TransactionDT to datetime (seconds from reference date)
     train["datetime"] = pd.to_datetime(train["TransactionDT"], unit="s",
                                         origin="2017-11-30")
     train["hour"] = train["datetime"].dt.hour
     train["dayofweek"] = train["datetime"].dt.dayofweek

     # Fraud rate by hour of day
     hourly_fraud = train.groupby("hour")["isFraud"].mean()
     # Expect: higher fraud rate at unusual hours (2-5 AM)
     ```
   - **Missing value analysis:** many features have 50%+ missing values.
     Are missing values informative? (Yes -- in fraud data, missing values
     often correlate with fraud because fraudsters provide less information.)
     ```python
     missing_by_fraud = train.groupby("isFraud").apply(
         lambda x: x.isnull().mean()
     ).T
     missing_by_fraud.columns = ["legitimate", "fraud"]
     missing_by_fraud["diff"] = missing_by_fraud["fraud"] - missing_by_fraud["legitimate"]
     # Features where fraud has MORE missing values are informative
     ```
   - **Correlation with target:** which features have the strongest signal?
   - **V-features analysis:** the anonymous V1-V339 features. Use PCA or
     correlation analysis to identify the most useful ones.

3. **Data Preprocessing** -- `src/data/preprocess.py`
   - Handle missing values strategically:
     - Numerical: fill with median, but also create binary "is_missing" indicator
       features (missingness is informative)
     - Categorical: fill with "Unknown" category
   - Encode categorical features:
     - Low cardinality (< 20 categories): one-hot encoding
     - High cardinality (email domain, device type): target encoding or
       frequency encoding
   ```python
   def preprocess_transactions(
       transactions: pd.DataFrame,
       identity: pd.DataFrame,
   ) -> pd.DataFrame:
       """Clean and preprocess fraud detection data."""
       # Merge transaction and identity features
       df = transactions.merge(identity, on="TransactionID", how="left")

       # Create missing value indicators (they are features!)
       for col in df.columns:
           if df[col].isnull().sum() > 0:
               df[f"{col}_missing"] = df[col].isnull().astype(int)

       # Handle numerical missing values
       num_cols = df.select_dtypes(include=["float64", "int64"]).columns
       df[num_cols] = df[num_cols].fillna(df[num_cols].median())

       # Encode categorical features
       cat_cols = df.select_dtypes(include=["object"]).columns
       for col in cat_cols:
           if df[col].nunique() < 20:
               df = pd.get_dummies(df, columns=[col], prefix=col)
           else:
               # Frequency encoding for high cardinality
               freq = df[col].value_counts(normalize=True)
               df[f"{col}_freq"] = df[col].map(freq)
               df.drop(col, axis=1, inplace=True)

       return df
   ```

4. **Feature Engineering: Velocity Features** -- `src/data/velocity.py`
   - Velocity features measure **how fast** a user/card/device is transacting.
     Fraud often involves rapid successive transactions.
   ```python
   def compute_velocity_features(
       df: pd.DataFrame,
       entity_col: str,        # "card1", "addr1", "DeviceInfo", etc.
       time_col: str = "TransactionDT",
       windows: list[int] = [3600, 86400, 604800],  # 1h, 1d, 7d in seconds
   ) -> pd.DataFrame:
       """Compute transaction velocity features for an entity."""
       df = df.sort_values(time_col)
       for window in windows:
           window_name = {3600: "1h", 86400: "1d", 604800: "7d"}[window]
           # Count transactions in window
           df[f"{entity_col}_txn_count_{window_name}"] = df.groupby(entity_col).apply(
               lambda g: g[time_col].apply(
                   lambda t: ((g[time_col] >= t - window) & (g[time_col] < t)).sum()
               )
           ).values
           # Sum of transaction amounts in window
           df[f"{entity_col}_txn_sum_{window_name}"] = df.groupby(entity_col).apply(
               lambda g: g.apply(
                   lambda row: g.loc[
                       (g[time_col] >= row[time_col] - window) &
                       (g[time_col] < row[time_col]),
                       "TransactionAmt"
                   ].sum(),
                   axis=1,
               )
           ).values
       return df
   ```

5. **Feature Engineering: Aggregation Features** -- `src/data/aggregation.py`
   - Rolling statistics per entity (card, user, device):
     - Mean/std/max transaction amount in last N transactions
     - Time since last transaction
     - Number of unique merchants in last N transactions
     - Ratio of current amount to average amount
   ```python
   def compute_aggregation_features(df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
       """Compute rolling aggregation features per entity."""
       df = df.sort_values("TransactionDT")

       # Rolling stats on transaction amount
       grouped = df.groupby(entity_col)["TransactionAmt"]
       df[f"{entity_col}_amt_mean"] = grouped.transform(
           lambda x: x.expanding().mean().shift(1)
       )
       df[f"{entity_col}_amt_std"] = grouped.transform(
           lambda x: x.expanding().std().shift(1)
       )
       df[f"{entity_col}_amt_max"] = grouped.transform(
           lambda x: x.expanding().max().shift(1)
       )

       # Amount ratio: current / historical average
       df[f"{entity_col}_amt_ratio"] = (
           df["TransactionAmt"] / df[f"{entity_col}_amt_mean"].clip(lower=1)
       )

       # Time since last transaction
       df[f"{entity_col}_time_since_last"] = grouped["TransactionDT"].transform(
           lambda x: x.diff()
       )

       return df
   ```

6. **Feature Engineering: Graph Features** -- `src/data/graph.py`
   - Fraud often involves shared infrastructure (same device, same IP, same email
     domain across multiple fraudulent accounts)
   - Build a graph where nodes are entities (cards, emails, devices) and edges
     connect entities that appear in the same transaction
   ```python
   def compute_graph_features(df: pd.DataFrame) -> pd.DataFrame:
       """Compute graph-based features from entity relationships."""
       # How many unique cards share this email domain?
       email_card_count = df.groupby("P_emaildomain")["card1"].nunique()
       df["email_card_count"] = df["P_emaildomain"].map(email_card_count)

       # How many unique emails use this card?
       card_email_count = df.groupby("card1")["P_emaildomain"].nunique()
       df["card_email_count"] = df["card1"].map(card_email_count)

       # Device sharing: how many cards have used this device?
       if "DeviceInfo" in df.columns:
           device_card_count = df.groupby("DeviceInfo")["card1"].nunique()
           df["device_card_count"] = df["DeviceInfo"].map(device_card_count)

       return df
   ```

7. **Walk-Forward Temporal Split** -- `src/data/split.py`
   - **Critical:** you MUST use temporal splitting, never random splitting.
     In production, you only have past data to predict future transactions.
   - Walk-forward validation: train on months 1-3, validate on month 4,
     then train on months 1-4, validate on month 5, etc.
   ```python
   def walk_forward_split(
       df: pd.DataFrame,
       time_col: str = "TransactionDT",
       n_splits: int = 5,
       test_size_days: int = 14,
   ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
       """Create walk-forward temporal splits for fraud detection.

       This simulates real-world model deployment where you train on historical
       data and evaluate on future data.
       """
       df = df.sort_values(time_col)
       min_time = df[time_col].min()
       max_time = df[time_col].max()
       total_range = max_time - min_time
       test_range = test_size_days * 86400  # Convert days to seconds

       splits = []
       for i in range(n_splits):
           test_end = max_time - i * test_range
           test_start = test_end - test_range
           train = df[df[time_col] < test_start]
           test = df[(df[time_col] >= test_start) & (df[time_col] < test_end)]
           if len(train) > 0 and len(test) > 0:
               splits.append((train, test))

       return list(reversed(splits))  # Chronological order
   ```

8. **Define DVC pipeline** -- `dvc.yaml`
   - Stage 1: download -> Stage 2: preprocess -> Stage 3: velocity features ->
     Stage 4: aggregation features -> Stage 5: graph features -> Stage 6: split

### Skills Learned

- EDA for fraud detection (class imbalance as the central challenge)
- Understanding why accuracy is useless for imbalanced problems
- Velocity feature engineering (transactions per time window)
- Aggregation features (rolling statistics per entity)
- Graph-based features for fraud networks
- Walk-forward temporal validation (essential for fraud detection)
- Missing values as informative features

---

## Phase 3: Feature Store

**Duration:** 3-4 days
**Objective:** Set up Feast for consistent feature serving across training and inference.

### Tasks

1. **Why a Feature Store?**
   - Without a feature store, you compute features differently in training (batch,
     on historical data) and serving (real-time, on live data). This "training-serving
     skew" is the #1 cause of model degradation in production.
   - Feast provides: one feature definition, two serving modes (offline for training,
     online for real-time inference).

2. **Set Up Feast** -- `configs/feature_store/`
   ```yaml
   # feature_store.yaml
   project: fraud_detection
   registry: data/registry.db
   provider: local
   online_store:
     type: redis
     connection_string: "redis://localhost:6379"
   offline_store:
     type: file
   entity_key_serialization_version: 2
   ```

3. **Define Feature Views** -- `src/feature_store/definitions.py`
   ```python
   from feast import Entity, FeatureView, Field, FileSource
   from feast.types import Float32, Int64, String

   # Entities
   card_entity = Entity(
       name="card_id",
       join_keys=["card1"],
       description="Credit card identifier",
   )

   transaction_entity = Entity(
       name="transaction_id",
       join_keys=["TransactionID"],
       description="Transaction identifier",
   )

   # Feature sources
   transaction_features_source = FileSource(
       path="data/processed/transaction_features.parquet",
       timestamp_field="event_timestamp",
   )

   # Feature views
   card_velocity_features = FeatureView(
       name="card_velocity_features",
       entities=[card_entity],
       schema=[
           Field(name="txn_count_1h", dtype=Int64),
           Field(name="txn_count_1d", dtype=Int64),
           Field(name="txn_count_7d", dtype=Int64),
           Field(name="txn_sum_1h", dtype=Float32),
           Field(name="txn_sum_1d", dtype=Float32),
           Field(name="txn_sum_7d", dtype=Float32),
           Field(name="amt_mean", dtype=Float32),
           Field(name="amt_std", dtype=Float32),
           Field(name="amt_ratio", dtype=Float32),
           Field(name="time_since_last", dtype=Float32),
       ],
       source=transaction_features_source,
       online=True,
       ttl=timedelta(days=7),
   )

   card_graph_features = FeatureView(
       name="card_graph_features",
       entities=[card_entity],
       schema=[
           Field(name="email_card_count", dtype=Int64),
           Field(name="card_email_count", dtype=Int64),
           Field(name="device_card_count", dtype=Int64),
       ],
       source=transaction_features_source,
       online=True,
       ttl=timedelta(days=1),
   )
   ```

4. **Materialize Features** -- `src/feature_store/materialize.py`
   - Materialize offline features to the online store (Redis) for real-time serving
   ```python
   from feast import FeatureStore

   def materialize_features(start_date: datetime, end_date: datetime) -> None:
       """Materialize features from offline store to online store (Redis)."""
       store = FeatureStore(repo_path="configs/feature_store")
       store.materialize(start_date=start_date, end_date=end_date)
       logger.info(f"Materialized features from {start_date} to {end_date}")
   ```

5. **Online Feature Retrieval** -- `src/feature_store/serve.py`
   - Retrieve features in real-time (<10ms) for scoring
   ```python
   def get_online_features(card_id: str) -> dict:
       """Retrieve features from Feast online store for real-time scoring."""
       store = FeatureStore(repo_path="configs/feature_store")
       features = store.get_online_features(
           features=[
               "card_velocity_features:txn_count_1h",
               "card_velocity_features:txn_count_1d",
               "card_velocity_features:txn_count_7d",
               "card_velocity_features:txn_sum_1h",
               "card_velocity_features:amt_mean",
               "card_velocity_features:amt_ratio",
               "card_velocity_features:time_since_last",
               "card_graph_features:email_card_count",
               "card_graph_features:device_card_count",
           ],
           entity_rows=[{"card1": card_id}],
       )
       return features.to_dict()
   ```

6. **Offline Feature Retrieval for Training**
   - Use Feast's point-in-time join to get historical features for training
   - This ensures features are computed at the correct timestamp (no future leakage)
   ```python
   def get_training_features(entity_df: pd.DataFrame) -> pd.DataFrame:
       """Get historical features for training with point-in-time correctness."""
       store = FeatureStore(repo_path="configs/feature_store")
       training_df = store.get_historical_features(
           entity_df=entity_df,
           features=[
               "card_velocity_features:txn_count_1h",
               "card_velocity_features:txn_count_1d",
               # ... all features
           ],
       ).to_df()
       return training_df
   ```

### Skills Learned

- Feature store concepts (online vs offline serving)
- Feast setup and configuration
- Feature view definitions and entity management
- Feature materialization (offline to online)
- Point-in-time joins for training data (avoiding future leakage)
- Training-serving consistency (the #1 cause of ML bugs in production)

---

## Phase 4: Model Development

**Duration:** 5-7 days
**Objective:** Build models that handle extreme class imbalance and provide reliable fraud scores.

### Tasks

1. **Handle Class Imbalance** -- `src/model/imbalance.py`
   - This is the central challenge. Three approaches, each with tradeoffs:

   **Approach 1: Class Weights**
   ```python
   def compute_class_weights(y: np.ndarray) -> dict:
       """Compute class weights inversely proportional to frequency."""
       n_samples = len(y)
       n_fraud = y.sum()
       n_legit = n_samples - n_fraud
       weight_fraud = n_samples / (2 * n_fraud)
       weight_legit = n_samples / (2 * n_legit)
       return {0: weight_legit, 1: weight_fraud}
       # For 3.5% fraud rate: weight_fraud ~ 14.3, weight_legit ~ 0.52
   ```

   **Approach 2: SMOTE (Synthetic Minority Oversampling)**
   ```python
   from imblearn.over_sampling import SMOTE
   from imblearn.pipeline import Pipeline as ImbPipeline

   def create_smote_pipeline(model, sampling_strategy: float = 0.3):
       """Create pipeline with SMOTE oversampling."""
       return ImbPipeline([
           ("smote", SMOTE(sampling_strategy=sampling_strategy, random_state=42)),
           ("model", model),
       ])
       # sampling_strategy=0.3 means oversample fraud to 30% of legitimate count
       # Do NOT oversample to 50/50 -- it creates too many synthetic samples
   ```

   **Approach 3: Focal Loss** (for neural models)
   ```python
   class FocalLoss(nn.Module):
       """Focal loss for extreme class imbalance.
       Down-weights well-classified examples, focuses on hard cases."""
       def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
           super().__init__()
           self.alpha = alpha
           self.gamma = gamma

       def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
           bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
           pt = torch.exp(-bce_loss)
           alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
           focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
           return focal_loss.mean()
   ```

2. **XGBoost Baseline** -- `src/model/xgboost_model.py`
   - XGBoost is the go-to for tabular fraud detection (fast, handles missing values
     natively, excellent feature importance)
   ```python
   import xgboost as xgb

   def train_xgboost(
       X_train: np.ndarray,
       y_train: np.ndarray,
       X_val: np.ndarray,
       y_val: np.ndarray,
       scale_pos_weight: float | None = None,
   ) -> xgb.Booster:
       """Train XGBoost for fraud detection."""
       if scale_pos_weight is None:
           # Automatically compute from class ratio
           scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

       dtrain = xgb.DMatrix(X_train, label=y_train)
       dval = xgb.DMatrix(X_val, label=y_val)

       params = {
           "objective": "binary:logistic",
           "eval_metric": ["aucpr", "logloss"],  # AUC-PR, not AUC-ROC!
           "scale_pos_weight": scale_pos_weight,
           "max_depth": 6,
           "learning_rate": 0.05,
           "subsample": 0.8,
           "colsample_bytree": 0.8,
           "min_child_weight": 5,
           "reg_alpha": 0.1,
           "reg_lambda": 1.0,
           "tree_method": "hist",  # Fast histogram-based training
           "random_state": 42,
       }

       model = xgb.train(
           params,
           dtrain,
           num_boost_round=1000,
           evals=[(dtrain, "train"), (dval, "val")],
           early_stopping_rounds=50,
           verbose_eval=100,
       )
       return model
   ```

3. **Sequence Model** -- `src/model/sequence_model.py`
   - Fraudsters have patterns in their transaction sequences (rapid small purchases
     followed by a large purchase, or geographic jumps)
   - Use an LSTM or Transformer to model transaction history per card
   ```python
   class TransactionSequenceModel(nn.Module):
       """LSTM model that processes a card's transaction history."""
       def __init__(
           self,
           n_features: int,
           hidden_dim: int = 64,
           n_layers: int = 2,
           dropout: float = 0.3,
       ):
           super().__init__()
           self.lstm = nn.LSTM(
               input_size=n_features,
               hidden_size=hidden_dim,
               num_layers=n_layers,
               batch_first=True,
               dropout=dropout,
           )
           self.classifier = nn.Sequential(
               nn.Linear(hidden_dim, 32),
               nn.ReLU(),
               nn.Dropout(dropout),
               nn.Linear(32, 1),
           )

       def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
           """
           x: (batch, max_seq_len, n_features) - padded transaction sequences
           lengths: (batch,) - actual sequence lengths
           """
           packed = nn.utils.rnn.pack_padded_sequence(
               x, lengths.cpu(), batch_first=True, enforce_sorted=False
           )
           _, (hidden, _) = self.lstm(packed)
           # Use last hidden state
           out = self.classifier(hidden[-1])
           return out.squeeze(-1)
   ```

4. **Ensemble Model** -- `src/model/ensemble.py`
   - Combine XGBoost (good at tabular features) with sequence model (good at
     temporal patterns) via stacking
   ```python
   class FraudEnsemble:
       """Stacking ensemble: XGBoost + Sequence Model -> Logistic Regression."""
       def __init__(self, xgb_model, seq_model, meta_model=None):
           self.xgb_model = xgb_model
           self.seq_model = seq_model
           self.meta_model = meta_model or LogisticRegression()

       def predict_proba(self, X_tabular, X_sequence, seq_lengths):
           xgb_scores = self.xgb_model.predict(xgb.DMatrix(X_tabular))
           seq_scores = torch.sigmoid(
               self.seq_model(X_sequence, seq_lengths)
           ).detach().numpy()
           meta_features = np.column_stack([xgb_scores, seq_scores])
           return self.meta_model.predict_proba(meta_features)[:, 1]
   ```

5. **Walk-Forward Validation** -- `src/training/walk_forward.py`
   - Standard cross-validation is WRONG for time-series fraud data.
     Walk-forward validation simulates real deployment.
   ```python
   def walk_forward_evaluate(
       df: pd.DataFrame,
       model_fn: Callable,
       n_splits: int = 5,
   ) -> list[dict]:
       """Evaluate model using walk-forward validation."""
       splits = walk_forward_split(df, n_splits=n_splits)
       results = []
       for i, (train_df, test_df) in enumerate(splits):
           X_train, y_train = train_df.drop("isFraud", axis=1), train_df["isFraud"]
           X_test, y_test = test_df.drop("isFraud", axis=1), test_df["isFraud"]

           model = model_fn()
           model.fit(X_train, y_train)
           y_pred_proba = model.predict_proba(X_test)[:, 1]

           metrics = evaluate_fraud_model(y_test, y_pred_proba)
           metrics["split"] = i
           metrics["train_size"] = len(train_df)
           metrics["test_size"] = len(test_df)
           metrics["test_fraud_rate"] = y_test.mean()
           results.append(metrics)

           logger.info(f"Split {i}: AUC-PR={metrics['auc_pr']:.4f}, "
                       f"fraud_rate={metrics['test_fraud_rate']:.4%}")
       return results
   ```

6. **Run Experiments** (track all in MLflow)

   | Experiment | Model | Imbalance Strategy | Expected Result |
   |-----------|-------|-------------------|----------------|
   | Baseline | XGBoost | None (raw data) | AUC-PR ~0.55, many false negatives |
   | Weighted | XGBoost | scale_pos_weight | AUC-PR ~0.65 |
   | SMOTE | XGBoost + SMOTE(0.3) | Oversampling | AUC-PR ~0.67 |
   | Tuned | XGBoost | Tuned weights + hyperparams | AUC-PR ~0.70 |
   | Sequence | LSTM | Focal loss | AUC-PR ~0.63 (less tabular data) |
   | Ensemble | XGBoost + LSTM stacking | Combined | AUC-PR ~0.72 |

7. **Feature Importance with SHAP** -- critical for fraud detection
   - Regulators require explainability: WHY was this transaction flagged?
   ```python
   import shap

   def explain_prediction(model, X_sample: np.ndarray, feature_names: list[str]):
       """Generate SHAP explanations for fraud predictions."""
       explainer = shap.TreeExplainer(model)
       shap_values = explainer.shap_values(X_sample)
       # For a single transaction:
       # "This transaction was flagged because: amount_ratio=15.2 (15x normal),
       #  txn_count_1h=8 (8 transactions in last hour), device_card_count=5
       #  (5 different cards from same device)"
       return shap_values
   ```

### Skills Learned

- Handling extreme class imbalance (class weights, SMOTE, focal loss)
- XGBoost for tabular fraud detection
- Sequence modeling for transaction history (LSTM)
- Stacking ensembles for combining model types
- Walk-forward validation for time-series ML
- SHAP explanations for model interpretability

---

## Phase 5: Evaluation

**Duration:** 3-4 days
**Objective:** Evaluate models using cost-sensitive metrics, NOT accuracy.

### Tasks

1. **Why NOT Accuracy?**
   - If 99.7% of transactions are legitimate, a model that always says "not fraud"
     gets 99.7% accuracy. This is useless.
   - If 99.7% of transactions are legitimate, AUC-ROC can look great (0.95+) even
     when the model is terrible. ROC is dominated by true negatives.
   - **Use AUC-PR (Precision-Recall) instead.** It focuses on the minority class.

2. **Precision-Recall Evaluation** -- `src/training/evaluate.py`
   ```python
   from sklearn.metrics import (
       precision_recall_curve, average_precision_score,
       confusion_matrix, classification_report
   )

   def evaluate_fraud_model(
       y_true: np.ndarray,
       y_pred_proba: np.ndarray,
       cost_fp: float = 15.0,    # Cost of blocking legitimate transaction
       cost_fn: float = 500.0,   # Cost of missing fraud
   ) -> dict:
       """Comprehensive fraud model evaluation.

       Key insight: we optimize for COST, not accuracy.
       """
       # AUC-PR (primary metric)
       auc_pr = average_precision_score(y_true, y_pred_proba)

       # Precision-Recall curve
       precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

       # Find optimal threshold based on cost
       optimal_threshold = find_cost_optimal_threshold(
           y_true, y_pred_proba, cost_fp, cost_fn
       )
       y_pred = (y_pred_proba >= optimal_threshold).astype(int)

       # Confusion matrix at optimal threshold
       tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

       # Cost analysis
       total_cost = fp * cost_fp + fn * cost_fn
       cost_per_transaction = total_cost / len(y_true)

       # Dollar amounts saved
       dollars_saved = tp * cost_fn  # Fraud we caught
       dollars_lost_fraud = fn * cost_fn  # Fraud we missed
       dollars_lost_friction = fp * cost_fp  # Legitimate we blocked

       return {
           "auc_pr": auc_pr,
           "optimal_threshold": optimal_threshold,
           "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
           "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
           "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
           "false_positive_rate": fp / (fp + tn),
           "total_cost": total_cost,
           "cost_per_transaction": cost_per_transaction,
           "dollars_saved": dollars_saved,
           "dollars_lost_fraud": dollars_lost_fraud,
           "dollars_lost_friction": dollars_lost_friction,
           "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
       }
   ```

3. **Cost-Optimal Threshold Selection** -- `src/training/threshold.py`
   ```python
   def find_cost_optimal_threshold(
       y_true: np.ndarray,
       y_pred_proba: np.ndarray,
       cost_fp: float = 15.0,
       cost_fn: float = 500.0,
       n_thresholds: int = 1000,
   ) -> float:
       """Find the threshold that minimizes total cost.

       This is NOT the same as maximizing F1 or accuracy.
       With cost_fn >> cost_fp, the optimal threshold is lower (more aggressive
       fraud detection), because missing fraud is much more expensive.
       """
       thresholds = np.linspace(0, 1, n_thresholds)
       costs = []
       for t in thresholds:
           y_pred = (y_pred_proba >= t).astype(int)
           tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
           total_cost = fp * cost_fp + fn * cost_fn
           costs.append(total_cost)

       optimal_idx = np.argmin(costs)
       optimal_threshold = thresholds[optimal_idx]
       logger.info(f"Optimal threshold: {optimal_threshold:.4f} "
                   f"(min cost: ${costs[optimal_idx]:,.0f})")
       return optimal_threshold
   ```

4. **Cost-Sensitivity Analysis** -- `notebooks/05_cost_analysis.ipynb`
   - Plot: total cost vs threshold for different cost ratios
   - Show: how does the optimal threshold change if fraud costs $1000 vs $100?
   - Generate business-friendly report: "At the optimal threshold, the model saves
     $X per 1000 transactions while incorrectly blocking Y legitimate transactions."

5. **Segment Analysis**
   - Evaluate separately by: transaction amount (low/medium/high), card type,
     time of day, geographic region
   - Critical: is the model fair across segments? Does it disproportionately
     flag certain card types?
   - High-value fraud detection rate must be >= 80% (this is a hard requirement)

6. **Generate Evaluation Report**
   - AUC-PR curves for all models
   - Cost analysis table
   - Per-segment performance breakdown
   - SHAP importance plots
   - Log all artifacts to MLflow

### Skills Learned

- Cost-sensitive evaluation (why accuracy and AUC-ROC fail for imbalanced data)
- Precision-Recall curves and AUC-PR
- Cost-optimal threshold selection
- Business-impact analysis (translating ML metrics to dollars)
- Fairness evaluation across segments

---

## Phase 6: Streaming Pipeline

**Duration:** 5-6 days
**Objective:** Build a real-time scoring pipeline with Kafka for sub-100ms transaction scoring.

### Tasks

1. **Kafka Setup** -- `configs/kafka_config.yaml`
   ```yaml
   kafka:
     bootstrap_servers: "kafka:9092"
     topics:
       transactions: "fraud.transactions.raw"
       scores: "fraud.transactions.scored"
       alerts: "fraud.alerts"
     consumer_group: "fraud-scoring-group"
     partitions: 4
     replication_factor: 1
   ```

2. **Kafka Producer (Transaction Simulator)** -- `src/streaming/producer.py`
   - Simulate a stream of transactions for development and testing
   ```python
   from confluent_kafka import Producer

   class TransactionProducer:
       def __init__(self, bootstrap_servers: str, topic: str):
           self.producer = Producer({"bootstrap.servers": bootstrap_servers})
           self.topic = topic

       def send_transaction(self, transaction: dict) -> None:
           """Send a transaction to the Kafka topic."""
           self.producer.produce(
               self.topic,
               key=str(transaction["TransactionID"]).encode(),
               value=json.dumps(transaction).encode(),
               callback=self._delivery_report,
           )
           self.producer.poll(0)

       def simulate_stream(
           self, df: pd.DataFrame, rate: float = 100.0
       ) -> None:
           """Simulate a transaction stream from historical data.
           rate: transactions per second."""
           interval = 1.0 / rate
           for _, row in df.iterrows():
               self.send_transaction(row.to_dict())
               time.sleep(interval)

       def _delivery_report(self, err, msg):
           if err:
               logger.error(f"Delivery failed: {err}")
   ```

3. **Kafka Consumer (Real-Time Scorer)** -- `src/streaming/consumer.py`
   - Consume transactions, compute features, score, produce results
   ```python
   from confluent_kafka import Consumer

   class FraudScoringConsumer:
       def __init__(
           self,
           bootstrap_servers: str,
           input_topic: str,
           output_topic: str,
           model,
           feature_store: FeatureStore,
       ):
           self.consumer = Consumer({
               "bootstrap.servers": bootstrap_servers,
               "group.id": "fraud-scoring-group",
               "auto.offset.reset": "latest",
           })
           self.consumer.subscribe([input_topic])
           self.producer = Producer({"bootstrap.servers": bootstrap_servers})
           self.output_topic = output_topic
           self.model = model
           self.feature_store = feature_store

       def process_messages(self) -> None:
           """Main processing loop: consume, score, produce."""
           while True:
               msg = self.consumer.poll(timeout=1.0)
               if msg is None:
                   continue
               if msg.error():
                   logger.error(f"Consumer error: {msg.error()}")
                   continue

               start = time.perf_counter()
               transaction = json.loads(msg.value())

               # Get features from feature store
               features = self.feature_store.get_online_features(
                   transaction["card1"]
               )

               # Combine transaction features with historical features
               input_features = self._prepare_features(transaction, features)

               # Score transaction
               fraud_score = self.model.predict_proba(input_features)
               latency_ms = (time.perf_counter() - start) * 1000

               # Produce scored result
               result = {
                   "TransactionID": transaction["TransactionID"],
                   "fraud_score": float(fraud_score),
                   "is_fraud": fraud_score >= self.threshold,
                   "latency_ms": latency_ms,
                   "model_version": self.model_version,
               }
               self.producer.produce(
                   self.output_topic,
                   value=json.dumps(result).encode(),
               )

               # Track latency
               SCORING_LATENCY.observe(latency_ms / 1000)
               if latency_ms > 100:
                   logger.warning(f"Slow scoring: {latency_ms:.0f}ms")
   ```

4. **Real-Time Feature Computation** -- `src/streaming/feature_compute.py`
   - Some features must be computed in real-time (e.g., "transactions in last hour"
     changes with every new transaction)
   - Use Redis for real-time counters and rolling windows
   ```python
   class RealTimeFeatureComputer:
       def __init__(self, redis_client: redis.Redis):
           self.redis = redis_client

       def update_and_get_features(self, transaction: dict) -> dict:
           """Update real-time counters and return current features."""
           card_id = transaction["card1"]
           amount = transaction["TransactionAmt"]
           timestamp = transaction["TransactionDT"]

           pipe = self.redis.pipeline()

           # Increment transaction count (with TTL for windowed counts)
           pipe.incr(f"card:{card_id}:txn_count_1h")
           pipe.expire(f"card:{card_id}:txn_count_1h", 3600)
           pipe.incr(f"card:{card_id}:txn_count_1d")
           pipe.expire(f"card:{card_id}:txn_count_1d", 86400)

           # Update running sum
           pipe.incrbyfloat(f"card:{card_id}:txn_sum_1h", amount)
           pipe.expire(f"card:{card_id}:txn_sum_1h", 3600)

           # Store last transaction time
           pipe.set(f"card:{card_id}:last_txn_time", timestamp)

           results = pipe.execute()

           return {
               "txn_count_1h": int(results[0]),
               "txn_count_1d": int(results[2]),
               "txn_sum_1h": float(results[4]),
               "time_since_last": timestamp - float(
                   self.redis.get(f"card:{card_id}:last_txn_time") or timestamp
               ),
           }
   ```

5. **End-to-End Streaming Pipeline** -- `src/streaming/pipeline.py`
   - Wire everything together: Kafka consumer -> feature compute -> feature store
     lookup -> model scoring -> Kafka producer
   - Measure end-to-end latency (target: < 100ms)
   - Handle failures gracefully (Kafka retries, feature store timeouts)

6. **Latency Optimization**
   - Profile the pipeline: where is time spent?
     - Feature store lookup: ~5-10ms (Redis)
     - Real-time feature computation: ~5-10ms (Redis)
     - Model inference: ~1-5ms (XGBoost is fast)
     - Kafka overhead: ~5-10ms
     - Total target: < 100ms
   - Optimization strategies:
     - Batch feature lookups when possible
     - Keep model in memory (never reload per request)
     - Use XGBoost's predict method on DMatrix (fastest path)
     - Minimize serialization/deserialization

### Skills Learned

- Apache Kafka producer/consumer patterns
- Real-time feature computation with Redis
- Streaming ML pipeline architecture
- Latency profiling and optimization
- Combining feature store with streaming features

---

## Phase 7: API & Serving

**Duration:** 3-4 days
**Objective:** Build a REST API for on-demand scoring, batch investigation, and a monitoring dashboard.

### Tasks

1. **Define API Schemas** -- `src/serving/schemas.py`
   ```python
   class TransactionScoreRequest(BaseModel):
       transaction_id: str
       card_id: str
       amount: float
       merchant: str | None = None
       timestamp: datetime | None = None
       # Additional transaction fields...

   class FraudScoreResponse(BaseModel):
       transaction_id: str
       fraud_score: float          # 0.0 to 1.0
       is_fraud: bool              # Based on threshold
       risk_level: str             # "low", "medium", "high", "critical"
       top_risk_factors: list[str] # SHAP-based explanations
       model_version: str
       latency_ms: float
       champion_or_challenger: str

   class BatchInvestigationRequest(BaseModel):
       card_ids: list[str] | None = None
       date_range: tuple[datetime, datetime] | None = None
       min_fraud_score: float = 0.5

   class BatchInvestigationResponse(BaseModel):
       flagged_transactions: list[FraudScoreResponse]
       total_flagged: int
       total_amount_at_risk: float
       risk_distribution: dict[str, int]  # {"low": 10, "medium": 5, "high": 3}
   ```

2. **Build FastAPI Application** -- `src/serving/app.py`
   - `POST /score` -- score a single transaction in real-time
   - `POST /investigate` -- batch investigate flagged transactions
   - `GET /card/{card_id}/history` -- get fraud score history for a card
   - `GET /health` -- health check (model loaded, Kafka connected, Redis connected, Feast available)
   - `GET /metrics` -- Prometheus metrics endpoint
   - `GET /model/info` -- current model version, threshold, champion/challenger status

3. **Champion/Challenger Routing** -- `src/serving/champion_challenger.py`
   - Route a percentage of traffic to the challenger model
   - Track performance of both models in parallel
   ```python
   class ChampionChallengerRouter:
       def __init__(self, config: dict):
           self.champion_model = load_model(config["champion"]["model_path"])
           self.challenger_model = load_model(config["challenger"]["model_path"])
           self.challenger_percentage = config["challenger"]["traffic_percentage"]

       def score(self, features: np.ndarray, transaction_id: str) -> tuple[float, str]:
           """Score transaction with champion or challenger model."""
           # Deterministic routing based on transaction ID
           use_challenger = (
               int(hashlib.md5(transaction_id.encode()).hexdigest()[:8], 16) % 100
               < self.challenger_percentage
           )
           if use_challenger:
               score = self.challenger_model.predict_proba(features)
               return float(score), "challenger"
           else:
               score = self.champion_model.predict_proba(features)
               return float(score), "champion"
   ```

4. **Build Streamlit Dashboard** -- `src/frontend/app.py`
   - Real-time fraud score distribution
   - Transaction investigation interface (search by card, date range, risk level)
   - Model comparison: champion vs challenger performance
   - Alert feed: recent high-risk transactions
   - Feature importance visualization (SHAP)

5. **API Tests**
   - Unit tests: schema validation, threshold logic, champion/challenger routing
   - Integration tests: full scoring pipeline with mock Kafka and Redis
   - Latency tests: verify < 100ms for real-time scoring endpoint

### Skills Learned

- Building real-time scoring APIs
- Champion/challenger model serving
- Batch investigation endpoints for fraud analysts
- SHAP-based explanation APIs
- Building fraud investigation dashboards

---

## Phase 8: Containerization

**Duration:** 2-3 days
**Objective:** Package the entire fraud detection stack into Docker containers.

### Tasks

1. **docker-compose.yaml** -- orchestrate all services
   ```yaml
   services:
     zookeeper:
       image: confluentinc/cp-zookeeper:7.5.0
       environment:
         ZOOKEEPER_CLIENT_PORT: 2181
       ports: ["2181:2181"]

     kafka:
       image: confluentinc/cp-kafka:7.5.0
       depends_on: [zookeeper]
       ports: ["9092:9092"]
       environment:
         KAFKA_BROKER_ID: 1
         KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
         KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
         KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

     redis:
       image: redis:7-alpine
       ports: ["6379:6379"]
       healthcheck:
         test: ["CMD", "redis-cli", "ping"]

     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports: ["8000:8000"]
       environment:
         - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
         - REDIS_URL=redis://redis:6379
         - MODEL_PATH=/app/models/xgboost_fraud.json
         - FEAST_REPO_PATH=/app/configs/feature_store
       depends_on:
         kafka:
           condition: service_started
         redis:
           condition: service_healthy

     consumer:
       build:
         context: .
         dockerfile: docker/Dockerfile.consumer
       environment:
         - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
         - REDIS_URL=redis://redis:6379
         - MODEL_PATH=/app/models/xgboost_fraud.json
       depends_on: [kafka, redis]

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

2. **Service Dockerfiles**
   - `Dockerfile.api` -- FastAPI scoring service
   - `Dockerfile.consumer` -- Kafka consumer for streaming scoring
   - `Dockerfile.frontend` -- Streamlit dashboard

3. **Verify the full stack**
   - `docker compose up` -- all services start and connect
   - Produce test transactions via Kafka
   - Verify they are scored and results appear on dashboard
   - Check Prometheus metrics and Grafana dashboard

### Skills Learned

- Orchestrating streaming infrastructure with Docker
- Kafka + Zookeeper containerization
- Multi-container ML pipeline deployment
- Service dependency management for complex stacks

---

## Phase 9: Testing & CI/CD

**Duration:** 2-3 days
**Objective:** Automate quality checks for the fraud detection pipeline.

### Tasks

1. **Write comprehensive tests**

   **Unit tests:**
   ```
   test_features.py
   ├── test_velocity_features_correct_counts
   ├── test_aggregation_features_no_future_leakage
   ├── test_graph_features_entity_counts
   └── test_missing_value_indicators_created

   test_velocity.py
   ├── test_velocity_with_single_transaction
   ├── test_velocity_window_boundaries
   └── test_velocity_handles_time_gaps

   test_model.py
   ├── test_xgboost_output_probability_range
   ├── test_xgboost_handles_missing_values
   ├── test_ensemble_combines_scores
   └── test_focal_loss_gradient

   test_threshold.py
   ├── test_cost_optimal_threshold_minimizes_cost
   ├── test_threshold_changes_with_cost_ratio
   └── test_threshold_is_between_0_and_1

   test_champion.py
   ├── test_routing_is_deterministic
   ├── test_traffic_split_matches_config
   └── test_both_models_produce_valid_scores
   ```

   **Integration tests:**
   ```
   test_api.py
   ├── test_score_endpoint_returns_valid_response
   ├── test_score_latency_under_100ms
   ├── test_batch_investigation
   ├── test_health_endpoint_checks_all_services
   └── test_concurrent_scoring_requests

   test_streaming.py
   ├── test_producer_sends_to_kafka
   ├── test_consumer_scores_transactions
   ├── test_end_to_end_latency
   └── test_consumer_handles_malformed_messages

   test_feature_store.py
   ├── test_online_feature_retrieval
   ├── test_offline_point_in_time_join
   └── test_feature_materialization
   ```

   **ML-specific tests:**
   ```
   ├── test_no_future_leakage_in_features
   ├── test_walk_forward_splits_are_chronological
   ├── test_model_auc_pr_above_threshold
   ├── test_false_positive_rate_below_threshold
   └── test_feature_computation_matches_training_and_serving
   ```

2. **CI Pipeline** -- `.github/workflows/ci.yaml`
   - Lint, type check, run unit tests
   - Spin up Kafka + Redis via Docker Compose for integration tests
   - Run streaming integration tests
   - Build all Docker images
   - Smoke test: start full stack, score a transaction, verify response

3. **CD Pipeline** -- `.github/workflows/cd.yaml`
   - Build and push Docker images on merge to main
   - Deploy to staging, run integration tests
   - Deploy to production with champion/challenger routing
   - Automated rollback if error rate spikes

### Skills Learned

- Testing streaming ML pipelines
- Testing with Kafka and Redis in CI
- ML model regression testing (AUC-PR thresholds)
- Automated rollback strategies

---

## Phase 10: Monitoring

**Duration:** 3-4 days
**Objective:** Monitor model quality, feature drift, false positive rates, and champion/challenger performance.

### Tasks

1. **System Metrics** -- `src/monitoring/metrics.py`
   - `scoring_requests_total` -- counter by model version and outcome
   - `scoring_latency_seconds` -- histogram (alert if > 100ms)
   - `kafka_consumer_lag` -- gauge (how far behind is the consumer?)
   - `feature_store_latency_seconds` -- histogram
   - `active_consumers` -- gauge

2. **Model Quality Monitoring** -- `src/monitoring/model_quality.py`
   - Track false positive rate over time (most important operational metric)
   - Track fraud detection rate over time (requires delayed labels)
   - Compare champion vs challenger performance
   ```python
   class ModelQualityTracker:
       def __init__(self):
           self.fp_rate = Gauge(
               "fraud_false_positive_rate",
               "Rolling false positive rate",
               ["model_version"],
           )
           self.detection_rate = Gauge(
               "fraud_detection_rate",
               "Rolling fraud detection rate",
               ["model_version"],
           )
           self.score_distribution = Histogram(
               "fraud_score_distribution",
               "Distribution of fraud scores",
               ["model_version"],
               buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
           )

       def update_from_labels(
           self,
           model_version: str,
           predictions: list[bool],
           actuals: list[bool],
       ) -> None:
           """Update quality metrics when ground truth labels arrive."""
           fp = sum(1 for p, a in zip(predictions, actuals) if p and not a)
           tp = sum(1 for p, a in zip(predictions, actuals) if p and a)
           fn = sum(1 for p, a in zip(predictions, actuals) if not p and a)
           tn = sum(1 for p, a in zip(predictions, actuals) if not p and not a)

           if fp + tn > 0:
               self.fp_rate.labels(model_version=model_version).set(fp / (fp + tn))
           if tp + fn > 0:
               self.detection_rate.labels(model_version=model_version).set(tp / (tp + fn))
   ```

3. **Feature Drift Detection** -- `src/monitoring/drift.py`
   - Monitor feature distributions over time
   - If feature distributions shift, the model may degrade
   - Use Population Stability Index (PSI) or Kolmogorov-Smirnov test
   ```python
   def compute_psi(
       expected: np.ndarray,
       actual: np.ndarray,
       n_bins: int = 10,
   ) -> float:
       """Compute Population Stability Index between two distributions.
       PSI < 0.1: no significant change
       PSI 0.1-0.25: moderate change (investigate)
       PSI > 0.25: significant change (retrain)
       """
       expected_hist, bin_edges = np.histogram(expected, bins=n_bins)
       actual_hist, _ = np.histogram(actual, bins=bin_edges)
       # Avoid division by zero
       expected_pct = np.clip(expected_hist / expected_hist.sum(), 1e-6, None)
       actual_pct = np.clip(actual_hist / actual_hist.sum(), 1e-6, None)
       psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
       return psi
   ```

4. **Champion/Challenger Monitoring**
   - Side-by-side comparison dashboard
   - Track: fraud scores, false positive rates, detection rates, latency
   - Statistical significance test: when can we confidently say the challenger
     is better or worse?
   - Auto-promote challenger to champion if it statistically outperforms

5. **Grafana Dashboard** -- `grafana/dashboards/fraud_monitoring.json`
   - Row 1: Transaction rate, scoring latency, Kafka consumer lag, error rate
   - Row 2: False positive rate, fraud detection rate (when labels available),
     fraud score distribution
   - Row 3: Feature drift (PSI per feature), top drifting features
   - Row 4: Champion vs challenger comparison (scores, latency, quality)
   - Row 5: Cost analysis (estimated dollars saved, dollars lost to friction)

6. **Alerting Rules**
   - Scoring latency > 100ms for 1 minute -> alert (SLA breach)
   - False positive rate > 2% for 1 hour -> alert (model degradation)
   - Kafka consumer lag > 10,000 messages -> alert (scoring cannot keep up)
   - Feature PSI > 0.25 for any key feature -> alert (distribution shift)
   - Challenger model worse than champion by > 5% for 24 hours -> auto-rollback
   - Health check failing -> alert immediately

### Skills Learned

- Monitoring fraud-specific metrics (false positive rate, detection rate)
- Feature drift detection with PSI
- Champion/challenger performance monitoring
- Cost-based monitoring (translating ML metrics to business impact)
- Streaming infrastructure monitoring (Kafka consumer lag)
- Automated model rollback strategies

---

## Timeline Summary

```
Week 1   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 1: Setup & Design Doc    (2 days)
         Phase 2: Data Pipeline         (3 days)

Week 2   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 2: Data Pipeline cont.   (3 days)
         Phase 3: Feature Store         (2 days)

Week 3   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 3: Feature Store cont.   (2 days)
         Phase 4: Model Development     (3 days)

Week 4   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 4: Model Development     (4 days)
         Phase 5: Evaluation            (1 day)

Week 5   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 5: Evaluation cont.      (3 days)
         Phase 6: Streaming Pipeline    (2 days)

Week 6   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 6: Streaming Pipeline    (4 days)
         Phase 7: API & Serving         (1 day)

Week 7   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 7: API & Serving cont.   (3 days)
         Phase 8: Containerization      (2 days)

Week 8   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 8: Containerization      (1 day)
         Phase 9: Testing & CI/CD       (3 days)
         Phase 10: Monitoring           (1 day)

Week 9   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
         Phase 10: Monitoring cont.     (3 days)
         Buffer / catch-up              (2 days)
```

**Total: ~45 days (9 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] Designing cost-sensitive ML systems
- [ ] Understanding fraud detection constraints (latency, cost asymmetry, adversaries)
- [ ] EDA for extremely imbalanced datasets
- [ ] Why accuracy and AUC-ROC are useless for fraud detection
- [ ] Handling extreme class imbalance (class weights, SMOTE, focal loss)
- [ ] Velocity feature engineering (transaction frequency analysis)
- [ ] Aggregation feature engineering (rolling statistics)
- [ ] Graph-based feature engineering (entity relationship networks)
- [ ] Walk-forward temporal validation (not random cross-validation)
- [ ] XGBoost for tabular fraud detection
- [ ] Sequence modeling for transaction history (LSTM)
- [ ] Stacking ensemble (combining model types)
- [ ] Feature store (Feast) for online + offline feature serving
- [ ] Training-serving consistency (eliminating skew)
- [ ] Point-in-time feature joins (avoiding future leakage)
- [ ] Precision-Recall evaluation (AUC-PR as primary metric)
- [ ] Cost-optimal threshold selection
- [ ] Business impact analysis (translating metrics to dollars)
- [ ] SHAP explanations for model interpretability
- [ ] Apache Kafka producer/consumer patterns
- [ ] Real-time feature computation with Redis
- [ ] Streaming ML pipeline architecture (<100ms latency)
- [ ] Champion/challenger model deployment
- [ ] Feature drift detection (PSI, KS test)
- [ ] False positive rate monitoring
- [ ] Automated model rollback
- [ ] Multi-container Docker orchestration (Kafka + Redis + Feast + API)
- [ ] CI/CD for streaming ML applications
- [ ] Fraud investigation dashboards

---

## Key Differences from Standard Classification Projects

| Concept | Standard Classification | Fraud Detection |
|---------|------------------------|----------------|
| Class balance | Roughly balanced | 0.1-3% positive rate |
| Primary metric | Accuracy, F1 | AUC-PR, cost function |
| Evaluation | Random cross-validation | Walk-forward temporal validation |
| Threshold | 0.5 by default | Cost-optimized (often 0.1-0.3) |
| Serving | Batch or request/response | Real-time streaming (<100ms) |
| Features | Static | Real-time + historical (feature store) |
| Infrastructure | API only | Kafka + Redis + Feast + API |
| Model deployment | Replace old model | Champion/challenger |
| Monitoring | Accuracy drift | False positive rate, feature drift, cost |
| Adversary | None | Fraudsters actively adapt |
| Explainability | Nice to have | Required (regulatory) |
| Feature engineering | Standard transforms | Velocity, aggregation, graph features |
| Cost of errors | Symmetric | Highly asymmetric ($500 FN vs $15 FP) |

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Initialize the project
mkdir fraud-detection && cd fraud-detection
git init

# 2. Create the folder structure
mkdir -p configs/feature_store data/{raw,processed,splits} notebooks \
  src/{data,feature_store,model,training,streaming,serving,monitoring,frontend} \
  tests/{unit,integration} docker .github/workflows \
  grafana/dashboards prometheus scripts

# 3. Start Docker services (Kafka, Redis)
docker compose up -d kafka redis

# 4. Install dependencies
pip install xgboost feast confluent-kafka fastapi redis mlflow

# 5. Start writing DESIGN_DOC.md
```
