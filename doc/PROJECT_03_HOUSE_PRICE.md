# End-to-End ML Project Plan: House Price Prediction

## Goal

Build a **production-grade** regression API that predicts house prices from structured
tabular features, with **model interpretability** via SHAP explanations. This project
introduces tabular ML, feature engineering, multi-model comparison, hyperparameter tuning
with Optuna, and data drift detection with Evidently -- skills that cover the majority
of real-world ML applications in industry.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why house price prediction? | Tabular data + regression is the most common ML task in industry. Finance, insurance, real estate, e-commerce -- most business ML is structured data, not images or text. |
| Why multiple models? | Real projects compare models. You need to know when XGBoost beats a neural network and why. |
| Why SHAP? | In regulated industries (finance, healthcare), you must explain predictions. SHAP is the gold standard for model interpretability. |
| How is this different from Projects 1-2? | Tabular data, heavy feature engineering, regression (not classification), model comparison, interpretability, and data drift detection. |

---

## Architecture Overview

```
                    ┌──────────────────┐
                    │    Streamlit     │
                    │    Frontend     │
                    │(interactive inputs)│
                    └──────┬───────────┘
                           │ HTTP
                           ▼
                    ┌──────────────┐      ┌──────────────────┐
  Feature JSON ────▶   FastAPI    │──────▶│  ML Model        │
                    │   Server    │       │  (XGBoost/LGBM)  │
                    └──────┬───────┘      └────────┬─────────┘
                           │                       │
                           │               ┌───────▼────────┐
                           │               │  SHAP Explainer│
                           │               │  (per-predict) │
                           │               └────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Logging │ │Prometheus│ │  MLflow  │
        │(Structured)│ │ Metrics │ │ Registry │
        └──────────┘ └────┬─────┘ └──────────┘
                          │
              ┌───────────┼───────────┐
              ▼                       ▼
        ┌──────────┐          ┌──────────┐
        │ Grafana  │          │ Evidently│
        │Dashboard │          │Drift Rpt │
        └──────────┘          └──────────┘

Everything runs in Docker. Deployed to Cloud Run / Railway.
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | ML ecosystem standard |
| ML Frameworks | scikit-learn, XGBoost, LightGBM | Industry standards for tabular data |
| Hyperparameter Tuning | Optuna | Modern, efficient, Bayesian optimization |
| Interpretability | SHAP | Gold standard for model explanations |
| Data Analysis | Pandas, NumPy | Tabular data essentials |
| Visualization | Matplotlib, Seaborn, Plotly | EDA and SHAP visualizations |
| Experiment Tracking | MLflow | Tracks params, metrics, artifacts, model registry |
| Data Versioning | DVC | Git for data -- essential for reproducibility |
| Drift Detection | Evidently | Open-source ML monitoring and drift detection |
| API Framework | FastAPI | Fast, modern, auto-generates OpenAPI docs |
| Frontend | Streamlit | Simplest way to build ML UIs with interactive inputs |
| Containerization | Docker + docker-compose | Deployment standard everywhere |
| CI/CD | GitHub Actions | Free, integrated with GitHub |
| Monitoring | Prometheus + Grafana | Industry standard observability |
| Testing | pytest | Python standard |
| Linting | Ruff + mypy | Fast linting + type checking |

---

## Project Structure

```
house-price-prediction/
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
│   ├── feature_config.yaml          # Feature engineering configuration
│   ├── serve_config.yaml            # Serving configuration
│   └── data_config.yaml             # Data paths, splits
│
├── data/                            # Git-ignored, DVC-tracked
│   ├── raw/                         # Original downloaded data
│   ├── processed/                   # Cleaned, feature-engineered data
│   ├── splits/                      # train/ val/ test/ CSVs
│   └── reference/                   # Reference data for drift detection
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Feature engineering experiments
│   ├── 03_model_comparison.ipynb    # Model comparison & selection
│   ├── 04_hyperparameter_tuning.ipynb  # Optuna tuning
│   └── 05_shap_analysis.ipynb       # SHAP interpretability analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py              # Download dataset script
│   │   ├── validate.py              # Data integrity checks
│   │   ├── preprocess.py            # Cleaning, missing value handling
│   │   └── features.py              # Feature engineering pipeline
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── models.py                # Model definitions (XGBoost, LGBM, Ridge)
│   │   ├── tuning.py                # Optuna hyperparameter tuning
│   │   └── export.py                # Export models for serving
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                 # Training pipeline
│   │   ├── evaluate.py              # Evaluation metrics & residual analysis
│   │   └── compare.py               # Multi-model comparison
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application
│   │   ├── schemas.py               # Request/response Pydantic models
│   │   ├── middleware.py            # Rate limiting, auth, CORS
│   │   ├── predict.py               # Inference logic
│   │   └── explain.py               # SHAP explanation generation
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Prometheus metric definitions
│   │   ├── drift.py                 # Feature drift detection (Evidently)
│   │   └── reports.py               # Drift report generation
│   │
│   └── frontend/
│       └── app.py                   # Streamlit UI with interactive inputs
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py         # Feature engineering correctness
│   │   ├── test_preprocessing.py    # Data cleaning tests
│   │   ├── test_model.py            # Model output validation
│   │   ├── test_schemas.py          # API schema validation
│   │   └── test_explain.py          # SHAP explanation tests
│   ├── integration/
│   │   ├── test_api.py              # Full API predict pipeline
│   │   ├── test_training.py         # Training runs without error
│   │   └── test_drift.py            # Drift detection pipeline
│   └── conftest.py                  # Shared fixtures (sample data)
│
├── docker/
│   ├── Dockerfile.api               # Multi-stage build for API
│   ├── Dockerfile.frontend          # Streamlit container
│   └── Dockerfile.training          # Training environment
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
│       └── regression_monitoring.json  # Pre-configured dashboard
│
├── prometheus/
│   └── prometheus.yml               # Scrape config
│
└── scripts/
    ├── setup.sh                     # One-command project setup
    ├── train.sh                     # Run training with default config
    ├── tune.sh                      # Run Optuna hyperparameter search
    └── deploy.sh                    # Deploy to cloud
```

---

## Phase 1: Project Setup & Design Doc

**Duration:** 1-2 days
**Objective:** Define what you are building before writing any code.

### Tasks

1. **Write `DESIGN_DOC.md`**
   - **Problem statement:** "Given a set of house features (square footage, bedrooms, neighborhood, etc.), predict the sale price"
   - **Success criteria:**
     - RMSE < $25,000 on test set (Ames Housing)
     - R-squared >= 0.88 on test set
     - MAE < $15,000
     - Inference latency < 50ms per prediction (including SHAP)
     - API handles 100 concurrent requests
     - Every prediction includes a SHAP explanation
   - **Out of scope:** real-time market data integration, image-based features (house photos), geographic visualization, time-series price trends
   - **Risks:** multicollinearity between features, outlier properties (mansions), missing data patterns, data leakage from time-dependent features

2. **Initialize the repository**
   - `git init`, create `.gitignore`
   - Create `pyproject.toml` with all dependencies:
     ```toml
     [project]
     name = "house-price-prediction"
     version = "0.1.0"
     requires-python = ">=3.11"
     dependencies = [
         "scikit-learn>=1.3.0",
         "xgboost>=2.0.0",
         "lightgbm>=4.2.0",
         "optuna>=3.4.0",
         "shap>=0.44.0",
         "pandas>=2.1.0",
         "numpy>=1.26.0",
         "matplotlib>=3.8.0",
         "seaborn>=0.13.0",
         "plotly>=5.18.0",
         "mlflow>=2.9.0",
         "evidently>=0.4.10",
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
   - Python virtual environment
   - Pre-commit hooks: ruff, mypy, trailing whitespace

### Skills Learned

- Writing ML design documents for regression tasks
- Understanding success criteria for regression (RMSE, R-squared, MAE)
- Planning for model interpretability from the start

---

## Phase 2: Data Pipeline

**Duration:** 4-5 days
**Objective:** Get clean, feature-engineered, versioned data ready for training.

### Tasks

1. **Download the dataset** -- `src/data/download.py`
   - Option A: **Ames Housing** (1,460 houses, 79 features) -- recommended, richer features
   - Option B: **California Housing** (20,640 houses, 8 features) -- simpler, larger
   ```python
   # Option A: Ames Housing from OpenML
   from sklearn.datasets import fetch_openml
   housing = fetch_openml(name="house_prices", as_frame=True)

   # Option B: California Housing
   from sklearn.datasets import fetch_california_housing
   housing = fetch_california_housing(as_frame=True)
   ```
   - Save raw data to `data/raw/` and track with DVC

2. **Set up DVC**
   - `dvc init`, configure remote storage
   - Track `data/raw/` with DVC
   - **Important for tabular data:** also version your feature engineering configuration

3. **Exploratory Data Analysis (EDA)** -- `notebooks/01_eda.ipynb`
   - **Target variable analysis:**
     - Sale price distribution (likely right-skewed -- log transform needed?)
     - Outlier identification (houses > $500K)
     ```python
     import seaborn as sns

     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
     # Raw distribution
     sns.histplot(df["SalePrice"], bins=50, ax=axes[0])
     axes[0].set_title("Sale Price Distribution")
     # Log-transformed
     sns.histplot(np.log1p(df["SalePrice"]), bins=50, ax=axes[1])
     axes[1].set_title("Log(Sale Price) Distribution")
     ```

   - **Feature analysis:**
     - Correlation heatmap (top 15 features correlated with SalePrice)
     - Scatter plots: SalePrice vs GrLivArea, TotalBsmtSF, GarageArea
     - Box plots: SalePrice by Neighborhood, OverallQual, GarageCars
     ```python
     # Correlation with target
     correlations = df.corr()["SalePrice"].sort_values(ascending=False)
     print(correlations.head(15))

     # Correlation heatmap for top features
     top_features = correlations.index[1:16]  # Exclude SalePrice itself
     sns.heatmap(df[top_features].corr(), annot=True, cmap="coolwarm")
     ```

   - **Missing data analysis:**
     - Missing value counts per feature (bar chart)
     - Missing value patterns (are they random or systematic?)
     - Features with > 50% missing -- consider dropping
     ```python
     missing = df.isnull().sum()
     missing = missing[missing > 0].sort_values(ascending=False)
     print(f"Features with missing values: {len(missing)}")
     missing.plot(kind="barh", figsize=(8, 10))
     ```

   - **Feature type analysis:**
     - Numerical features: distributions, outliers (box plots)
     - Categorical features: cardinality, value counts
     - Ordinal features: proper ordering (ExterQual: Po < Fa < TA < Gd < Ex)

   - **Multicollinearity check:**
     - Variance Inflation Factor (VIF) for numerical features
     - Identify and document highly correlated feature pairs

4. **Data preprocessing** -- `src/data/preprocess.py`
   - **Missing value handling** (strategy depends on the feature):
     ```python
     class MissingValueHandler:
         def __init__(self):
             self.strategies = {
                 # Numeric: median imputation
                 "LotFrontage": "median",
                 "MasVnrArea": "zero",
                 "GarageYrBlt": "median",
                 # Categorical: "None" means feature absent
                 "Alley": "None",         # No alley access
                 "PoolQC": "None",        # No pool
                 "Fence": "None",         # No fence
                 "FireplaceQu": "None",   # No fireplace
                 "GarageType": "None",    # No garage
                 "BsmtQual": "None",      # No basement
             }

         def transform(self, df):
             for col, strategy in self.strategies.items():
                 if strategy == "median":
                     df[col].fillna(df[col].median(), inplace=True)
                 elif strategy == "zero":
                     df[col].fillna(0, inplace=True)
                 elif strategy == "None":
                     df[col].fillna("None", inplace=True)
             return df
     ```

   - **Outlier handling:**
     - Remove extreme outliers (e.g., GrLivArea > 4000 with low price -- likely data errors)
     - Document every removal decision

   - **Encoding categorical features:**
     - **Ordinal encoding** for ordered categories (ExterQual, BsmtQual, etc.):
       ```python
       quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
       ```
     - **Target encoding** for high-cardinality categoricals (Neighborhood):
       ```python
       from sklearn.preprocessing import TargetEncoder
       encoder = TargetEncoder(smooth="auto")
       df["Neighborhood_encoded"] = encoder.fit_transform(
           df[["Neighborhood"]], df["SalePrice"]
       )
       ```
     - **One-hot encoding** for low-cardinality categoricals (HouseStyle, RoofStyle)
     - **Why not one-hot everything?** Neighborhood has 25 values -- one-hot creates 25 sparse columns. Target encoding creates 1 informative column.

5. **Feature engineering** -- `src/data/features.py`
   - This is where tabular ML shines. Create new features from domain knowledge:
     ```python
     class FeatureEngineer:
         """Create new features from existing ones."""

         def transform(self, df):
             # Total square footage
             df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

             # Total bathrooms
             df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                                + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])

             # Total porch area
             df["TotalPorchSF"] = (df["OpenPorchSF"] + df["EnclosedPorch"]
                                    + df["3SsnPorch"] + df["ScreenPorch"])

             # House age and remodel age
             df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
             df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

             # Was the house remodeled?
             df["HasRemod"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

             # Quality x Area interactions
             df["OverallQual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
             df["OverallQual_x_TotalSF"] = df["OverallQual"] * df["TotalSF"]

             # Has feature flags
             df["HasPool"] = (df["PoolArea"] > 0).astype(int)
             df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
             df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
             df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

             # Log transform skewed numerical features
             skewed_features = ["LotArea", "GrLivArea", "TotalBsmtSF"]
             for feat in skewed_features:
                 df[f"{feat}_log"] = np.log1p(df[feat])

             return df
     ```
   - **Why feature engineering matters for tabular data:** unlike deep learning on images/text, tree-based models do not automatically learn feature interactions. You must create them.

6. **Define data splits and DVC pipeline**
   - Split into train (70%), validation (15%), test (15%)
   - **Critical:** split BEFORE feature engineering that uses target (target encoding). Otherwise you have data leakage.
   - Save reference data for drift detection:
     ```python
     # Save training data statistics as reference for drift detection
     reference_data = X_train.describe().to_dict()
     with open("data/reference/feature_stats.json", "w") as f:
         json.dump(reference_data, f)
     ```
   - DVC pipeline:
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
       features:
         cmd: python -m src.data.features
         deps:
           - data/processed/
           - src/data/features.py
           - configs/feature_config.yaml
         outs:
           - data/splits/
           - data/reference/
     ```

### Skills Learned

- EDA for tabular/structured data (correlations, distributions, missing patterns)
- Missing value imputation strategies (not just "fill with mean")
- Feature engineering for tabular ML (the highest-leverage skill for structured data)
- Encoding categorical variables (ordinal, target, one-hot -- when to use each)
- Preventing data leakage in feature engineering
- Building reproducible feature pipelines

---

## Phase 3: Model Development & Experiment Tracking

**Duration:** 5-6 days
**Objective:** Train multiple models, tune hyperparameters with Optuna, track experiments, pick the best.

### Tasks

1. **Define the models** -- `src/model/models.py`
   - Train three different model families and compare:
     ```python
     from sklearn.linear_model import Ridge
     from xgboost import XGBRegressor
     from lightgbm import LGBMRegressor

     def get_models():
         return {
             "ridge": Ridge(alpha=1.0),
             "xgboost": XGBRegressor(
                 n_estimators=500,
                 max_depth=6,
                 learning_rate=0.05,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 random_state=42,
             ),
             "lightgbm": LGBMRegressor(
                 n_estimators=500,
                 max_depth=6,
                 learning_rate=0.05,
                 subsample=0.8,
                 colsample_bytree=0.8,
                 random_state=42,
                 verbose=-1,
             ),
         }
     ```
   - **Why these three:**
     - **Ridge Regression:** simple baseline. If it performs well, your features are good. If it performs poorly, the relationship is nonlinear.
     - **XGBoost:** industry workhorse. Excellent with tabular data, handles missing values natively, built-in regularization.
     - **LightGBM:** faster than XGBoost on large datasets, histogram-based splitting, handles categoricals natively.

2. **Write training config** -- `configs/train_config.yaml`
   ```yaml
   target:
     column: SalePrice
     log_transform: true   # Predict log(price), convert back

   models:
     ridge:
       alpha: 1.0
       fit_intercept: true

     xgboost:
       n_estimators: 500
       max_depth: 6
       learning_rate: 0.05
       subsample: 0.8
       colsample_bytree: 0.8
       early_stopping_rounds: 50
       eval_metric: rmse

     lightgbm:
       n_estimators: 500
       max_depth: 6
       learning_rate: 0.05
       subsample: 0.8
       colsample_bytree: 0.8
       early_stopping_rounds: 50
       metric: rmse

   cross_validation:
     n_splits: 5
     strategy: kfold
     shuffle: true
     random_state: 42

   tuning:
     n_trials: 100
     timeout: 3600        # 1 hour max
     metric: rmse
   ```

3. **Write the training pipeline** -- `src/training/train.py`
   - Train all models with cross-validation:
     ```python
     from sklearn.model_selection import cross_val_score
     import mlflow

     def train_and_evaluate(models, X_train, y_train, X_val, y_val):
         results = {}

         for name, model in models.items():
             with mlflow.start_run(run_name=name):
                 # Cross-validation on training set
                 cv_scores = cross_val_score(
                     model, X_train, y_train,
                     cv=5, scoring="neg_root_mean_squared_error"
                 )
                 cv_rmse = -cv_scores.mean()

                 # Fit on full training set
                 model.fit(X_train, y_train)

                 # Evaluate on validation set
                 y_pred = model.predict(X_val)
                 val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                 val_mae = mean_absolute_error(y_val, y_pred)
                 val_r2 = r2_score(y_val, y_pred)

                 # Log to MLflow
                 mlflow.log_params(model.get_params())
                 mlflow.log_metric("cv_rmse", cv_rmse)
                 mlflow.log_metric("val_rmse", val_rmse)
                 mlflow.log_metric("val_mae", val_mae)
                 mlflow.log_metric("val_r2", val_r2)

                 results[name] = {
                     "model": model,
                     "cv_rmse": cv_rmse,
                     "val_rmse": val_rmse,
                     "val_r2": val_r2,
                 }

         return results
     ```

4. **Model comparison** -- `src/training/compare.py`
   - Compare models on key metrics:
     ```python
     def compare_models(results):
         comparison = pd.DataFrame({
             name: {
                 "CV RMSE": r["cv_rmse"],
                 "Val RMSE": r["val_rmse"],
                 "Val MAE": r["val_mae"],
                 "Val R2": r["val_r2"],
             }
             for name, r in results.items()
         }).T

         print(comparison.to_string())
         return comparison
     ```
   - Visualize: bar chart of RMSE per model, scatter of predicted vs actual per model

5. **Hyperparameter tuning with Optuna** -- `src/model/tuning.py`
   - Tune the best-performing model (likely XGBoost or LightGBM):
     ```python
     import optuna

     def objective(trial):
         params = {
             "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
             "max_depth": trial.suggest_int("max_depth", 3, 10),
             "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
             "subsample": trial.suggest_float("subsample", 0.6, 1.0),
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
             "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
             "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
             "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
         }

         model = XGBRegressor(**params, random_state=42)
         cv_scores = cross_val_score(
             model, X_train, y_train,
             cv=5, scoring="neg_root_mean_squared_error"
         )
         return -cv_scores.mean()

     study = optuna.create_study(direction="minimize")
     study.optimize(objective, n_trials=100, timeout=3600)

     print(f"Best RMSE: {study.best_value:.2f}")
     print(f"Best params: {study.best_params}")
     ```
   - Visualize Optuna results:
     ```python
     from optuna.visualization import plot_optimization_history
     from optuna.visualization import plot_param_importances

     plot_optimization_history(study)
     plot_param_importances(study)
     ```
   - Log best parameters and Optuna plots to MLflow

6. **Set up MLflow experiment tracking**
   - Log per-experiment: model type, hyperparameters, CV and validation metrics
   - Log artifacts: trained model, feature importance plot, comparison table
   - **Register the best model** in MLflow Model Registry
   - Also log the feature engineering pipeline (scaler, encoder objects) -- they are part of the model

7. **Run experiments** (track all in MLflow)

   | Experiment | What Changes | Expected Result |
   |-----------|-------------|----------------|
   | Ridge baseline | Default Ridge, scaled features | R2 ~0.85, establishes linear baseline |
   | XGBoost default | Default XGBoost params | R2 ~0.88, beats linear |
   | LightGBM default | Default LightGBM params | R2 ~0.88, comparable to XGBoost |
   | XGBoost tuned | Optuna-tuned XGBoost | R2 ~0.91, best single model |
   | LightGBM tuned | Optuna-tuned LightGBM | R2 ~0.90, close to XGBoost |
   | Log target | Predict log(price) | Lower RMSE on expensive houses |
   | Fewer features | Top 20 features by importance | Simpler model, slight accuracy drop |
   | Ensemble | Average of XGBoost + LightGBM | R2 ~0.92, best overall |

8. **Pick the best model** using MLflow UI
   - Compare all runs on val_rmse and val_r2
   - Consider: accuracy vs complexity vs inference speed
   - Promote best model to "Production" in MLflow registry

### Skills Learned

- Training and comparing multiple model families
- Cross-validation for regression
- Hyperparameter tuning with Optuna (Bayesian optimization)
- Understanding why tree-based models dominate tabular data
- Feature importance analysis
- Model selection methodology (not just picking the lowest error)

---

## Phase 4: Evaluation & Interpretability

**Duration:** 3-4 days
**Objective:** Thoroughly evaluate the model, analyze residuals, and generate SHAP explanations.

### Tasks

1. **Comprehensive evaluation** -- `src/training/evaluate.py`
   - Metrics on the **held-out test set**:
     - RMSE (Root Mean Squared Error) -- penalizes large errors
     - MAE (Mean Absolute Error) -- average error in dollars
     - MAPE (Mean Absolute Percentage Error) -- relative error
     - R-squared -- proportion of variance explained
   ```python
   from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

   def evaluate_regression(y_true, y_pred):
       rmse = np.sqrt(mean_squared_error(y_true, y_pred))
       mae = mean_absolute_error(y_true, y_pred)
       mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
       r2 = r2_score(y_true, y_pred)

       return {
           "rmse": rmse,
           "mae": mae,
           "mape": mape,
           "r2": r2,
       }
   ```

2. **Residual analysis** -- `notebooks/03_model_comparison.ipynb`
   - **Predicted vs Actual scatter plot** (should be tight around y=x line):
     ```python
     fig, ax = plt.subplots(figsize=(8, 8))
     ax.scatter(y_test, y_pred, alpha=0.5, s=10)
     ax.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], "r--", lw=2)
     ax.set_xlabel("Actual Price")
     ax.set_ylabel("Predicted Price")
     ax.set_title("Predicted vs Actual House Prices")
     ```

   - **Residual distribution** (should be approximately normal, centered at 0):
     ```python
     residuals = y_test - y_pred
     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
     # Histogram
     axes[0].hist(residuals, bins=50)
     axes[0].set_title("Residual Distribution")
     # Residuals vs predicted
     axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
     axes[1].axhline(y=0, color="r", linestyle="--")
     axes[1].set_title("Residuals vs Predicted")
     # QQ plot
     from scipy import stats
     stats.probplot(residuals, plot=axes[2])
     ```

   - **Error by price range** (are expensive houses harder to predict?):
     ```python
     bins = [0, 100000, 200000, 300000, 500000, float("inf")]
     labels = ["<100K", "100-200K", "200-300K", "300-500K", "500K+"]
     df_eval["price_bin"] = pd.cut(y_test, bins=bins, labels=labels)
     df_eval.groupby("price_bin")["abs_error"].mean().plot(kind="bar")
     ```

   - **Error by feature** (which neighborhoods have the highest error?):
     - Group errors by categorical features to find systematic biases

3. **SHAP analysis** -- `notebooks/05_shap_analysis.ipynb`
   - **Global feature importance** (which features matter most across all predictions):
     ```python
     import shap

     explainer = shap.TreeExplainer(model)
     shap_values = explainer.shap_values(X_test)

     # Summary plot: feature importance + direction of effect
     shap.summary_plot(shap_values, X_test)

     # Bar plot: mean absolute SHAP values
     shap.summary_plot(shap_values, X_test, plot_type="bar")
     ```

   - **Individual prediction explanations** (why was THIS house priced at $250K?):
     ```python
     # Waterfall plot for a single prediction
     idx = 0  # First test example
     shap.waterfall_plot(shap.Explanation(
         values=shap_values[idx],
         base_values=explainer.expected_value,
         data=X_test.iloc[idx],
         feature_names=X_test.columns.tolist(),
     ))
     ```

   - **Dependence plots** (how does each feature affect price?):
     ```python
     # How does overall quality affect price?
     shap.dependence_plot("OverallQual", shap_values, X_test)

     # How does living area affect price, colored by quality?
     shap.dependence_plot("GrLivArea", shap_values, X_test,
                          interaction_index="OverallQual")
     ```

   - **Force plot** for a batch of predictions:
     ```python
     shap.force_plot(explainer.expected_value, shap_values[:100], X_test.iloc[:100])
     ```

4. **Performance benchmarking**
   - Measure inference latency per model:
     - Ridge: expected < 1ms
     - XGBoost: expected < 5ms
     - LightGBM: expected < 5ms
     - SHAP explanation: expected < 50ms (TreeExplainer is fast)
   - Memory footprint: model size on disk and in memory
   - Feature engineering pipeline time

5. **Export model for serving** -- `src/model/export.py`
   - Save the full prediction pipeline:
     ```python
     import joblib

     # Save the model
     joblib.dump(model, "models/xgboost_model.joblib")

     # Save the preprocessing pipeline
     joblib.dump(preprocessor, "models/preprocessor.joblib")

     # Save the feature engineer
     joblib.dump(feature_engineer, "models/feature_engineer.joblib")

     # Save the SHAP explainer (pre-computed background)
     joblib.dump(explainer, "models/shap_explainer.joblib")
     ```

6. **Write `MODEL_CARD.md`**
   - Model type and training data description
   - Feature list and engineering details
   - Evaluation metrics with confidence intervals
   - SHAP-based feature importance ranking
   - Known limitations:
     - Trained on Ames, Iowa data (may not generalize to other markets)
     - Outlier predictions for luxury properties
     - Missing data handling assumptions
   - Intended use: price estimation for standard residential properties

### Skills Learned

- Regression evaluation metrics and when to use each
- Residual analysis (the most important diagnostic tool for regression)
- SHAP for model interpretability (global and local explanations)
- Dependence plots for understanding feature effects
- Exporting full prediction pipelines (not just the model)

---

## Phase 5: API & Serving Layer

**Duration:** 3-4 days
**Objective:** Wrap the model in a production API with SHAP explanations per prediction.

### Tasks

1. **Define API schemas** -- `src/serving/schemas.py`
   ```python
   from pydantic import BaseModel, Field

   class HouseFeatures(BaseModel):
       """Input features for house price prediction."""
       overall_qual: int = Field(..., ge=1, le=10, description="Overall quality (1-10)")
       gr_liv_area: float = Field(..., gt=0, description="Above ground living area (sq ft)")
       garage_cars: int = Field(..., ge=0, le=5, description="Garage car capacity")
       total_bsmt_sf: float = Field(..., ge=0, description="Total basement sq ft")
       full_bath: int = Field(..., ge=0, description="Number of full bathrooms")
       year_built: int = Field(..., ge=1800, le=2030, description="Year built")
       year_remod: int = Field(..., ge=1800, le=2030, description="Year remodeled")
       lot_area: float = Field(..., gt=0, description="Lot size (sq ft)")
       neighborhood: str = Field(..., description="Neighborhood name")
       # ... additional features

   class ShapExplanation(BaseModel):
       """SHAP explanation for a single prediction."""
       base_value: float              # Average predicted price
       feature_contributions: dict[str, float]  # Feature -> SHAP value
       top_positive_features: list[dict]  # Top features pushing price UP
       top_negative_features: list[dict]  # Top features pushing price DOWN

   class PricePrediction(BaseModel):
       predicted_price: float         # Predicted sale price in dollars
       confidence_interval: dict[str, float]  # {"lower": 180000, "upper": 220000}
       model_version: str
       latency_ms: float
       explanation: ShapExplanation    # SHAP explanation

   class HealthResponse(BaseModel):
       status: str
       model_loaded: bool
       model_version: str
       uptime_seconds: float
       feature_count: int
   ```

2. **Build FastAPI application** -- `src/serving/app.py`
   - `POST /predict` -- single house prediction with SHAP explanation
   - `POST /predict/batch` -- batch predictions (up to 20 houses)
   - `GET /health` -- health check
   - `GET /metrics` -- Prometheus metrics endpoint
   - `GET /features` -- return expected feature names and ranges
   ```python
   from fastapi import FastAPI
   from contextlib import asynccontextmanager

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       app.state.model = joblib.load("models/xgboost_model.joblib")
       app.state.preprocessor = joblib.load("models/preprocessor.joblib")
       app.state.feature_engineer = joblib.load("models/feature_engineer.joblib")
       app.state.explainer = joblib.load("models/shap_explainer.joblib")
       yield

   app = FastAPI(title="House Price Prediction API", lifespan=lifespan)

   @app.post("/predict", response_model=PricePrediction)
   async def predict(features: HouseFeatures):
       start = time.perf_counter()

       # Feature engineering
       df = pd.DataFrame([features.model_dump()])
       df_processed = app.state.feature_engineer.transform(df)
       X = app.state.preprocessor.transform(df_processed)

       # Prediction
       log_price = app.state.model.predict(X)[0]
       price = np.expm1(log_price)  # Reverse log transform

       # SHAP explanation
       shap_values = app.state.explainer.shap_values(X)[0]
       explanation = build_explanation(shap_values, X, app.state.explainer)

       latency = (time.perf_counter() - start) * 1000

       return PricePrediction(
           predicted_price=round(float(price), 2),
           confidence_interval={"lower": price * 0.9, "upper": price * 1.1},
           model_version=MODEL_VERSION,
           latency_ms=latency,
           explanation=explanation,
       )
   ```

3. **SHAP explanation generation** -- `src/serving/explain.py`
   ```python
   def build_explanation(shap_values, X, explainer):
       feature_names = X.columns.tolist() if hasattr(X, "columns") else [f"f{i}" for i in range(len(shap_values))]
       contributions = dict(zip(feature_names, shap_values.tolist()))

       # Sort by absolute SHAP value
       sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

       top_positive = [
           {"feature": name, "contribution": val}
           for name, val in sorted_features if val > 0
       ][:5]

       top_negative = [
           {"feature": name, "contribution": val}
           for name, val in sorted_features if val < 0
       ][:5]

       return ShapExplanation(
           base_value=float(explainer.expected_value),
           feature_contributions=contributions,
           top_positive_features=top_positive,
           top_negative_features=top_negative,
       )
   ```

4. **API middleware** -- `src/serving/middleware.py`
   - **Input validation:** feature ranges (square footage > 0, bedrooms >= 0, etc.)
   - **Rate limiting:** max 200 requests/minute per IP
   - **CORS:** allow frontend to call the API
   - **Request ID:** unique ID per request for tracing
   - **Error handling:** return clean JSON errors

5. **Build Streamlit frontend** -- `src/frontend/app.py`
   - Interactive form with sliders and dropdowns for each feature:
     ```python
     import streamlit as st

     st.title("House Price Predictor")

     col1, col2 = st.columns(2)
     with col1:
         overall_qual = st.slider("Overall Quality", 1, 10, 5)
         gr_liv_area = st.number_input("Living Area (sq ft)", 300, 6000, 1500)
         total_bsmt_sf = st.number_input("Basement (sq ft)", 0, 3000, 800)
         lot_area = st.number_input("Lot Area (sq ft)", 1000, 100000, 10000)

     with col2:
         year_built = st.slider("Year Built", 1900, 2025, 1990)
         garage_cars = st.slider("Garage Cars", 0, 4, 2)
         full_bath = st.slider("Full Bathrooms", 0, 4, 2)
         neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS)

     if st.button("Predict Price"):
         result = requests.post(f"{API_URL}/predict", json={...})
         prediction = result.json()

         st.metric("Predicted Price", f"${prediction['predicted_price']:,.0f}")

         # SHAP waterfall chart
         st.subheader("Why this price?")
         explanation = prediction["explanation"]
         # Render top contributing features as horizontal bar chart
         ...
     ```
   - Display SHAP explanation as a horizontal bar chart
   - Show confidence interval
   - "Compare" mode: predict two houses side by side

6. **API tests** -- `tests/`
   - Unit tests: feature validation, preprocessing, SHAP formatting
   - Integration tests: full request -> response with sample house features
   - Edge case tests: minimum/maximum feature values, missing optional features
   - SHAP consistency tests: explanations sum to prediction - base_value

### Skills Learned

- Serving tabular ML models (feature engineering at inference time)
- SHAP explanations in production APIs
- Building interactive ML frontends with Streamlit
- Input validation for structured features (ranges, types)
- Packaging full prediction pipelines (preprocessor + model + explainer)

---

## Phase 6: Containerization

**Duration:** 2 days
**Objective:** Package everything into Docker containers.

### Tasks

1. **API Dockerfile** -- `docker/Dockerfile.api`
   ```dockerfile
   FROM python:3.11-slim AS builder
   WORKDIR /app
   COPY pyproject.toml .
   RUN pip install --no-cache-dir .

   FROM python:3.11-slim
   COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
   COPY src/ /app/src/
   COPY models/ /app/models/
   EXPOSE 8000
   CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
   ```
   - Tabular models are small (~10MB for XGBoost), so Docker images are lightweight
   - Include preprocessor, feature engineer, and SHAP explainer in models/

2. **Frontend Dockerfile** -- `docker/Dockerfile.frontend`

3. **docker-compose.yaml** -- orchestrate everything
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports: ["8000:8000"]
       environment:
         - MODEL_PATH=/app/models/xgboost_model.joblib
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

4. **Verify the full stack locally**

### Skills Learned

- Docker for tabular ML (much smaller images than NLP/CV)
- Packaging preprocessing pipelines in containers
- Multi-service orchestration

---

## Phase 7: Testing & CI/CD

**Duration:** 2-3 days
**Objective:** Automate quality checks so nothing broken reaches production.

### Tasks

1. **Write comprehensive tests**

   **Unit tests**:
   ```
   test_features.py
   ├── test_total_sf_is_sum_of_components
   ├── test_house_age_is_positive
   ├── test_log_transform_handles_zero
   ├── test_feature_engineer_output_columns
   └── test_feature_engineer_is_deterministic

   test_preprocessing.py
   ├── test_missing_value_handler_fills_none
   ├── test_ordinal_encoding_preserves_order
   ├── test_target_encoding_handles_unseen_category
   └── test_scaler_normalizes_correctly

   test_model.py
   ├── test_model_output_is_single_value
   ├── test_model_predictions_are_positive
   ├── test_model_loads_from_joblib
   └── test_larger_house_costs_more  # Sanity check

   test_explain.py
   ├── test_shap_values_sum_to_prediction_minus_base
   ├── test_explanation_has_all_features
   └── test_top_features_are_sorted_by_importance
   ```

   **Integration tests**:
   ```
   test_api.py
   ├── test_predict_returns_200_with_valid_features
   ├── test_predict_returns_422_on_invalid_features
   ├── test_predict_includes_shap_explanation
   ├── test_batch_predict_multiple_houses
   ├── test_features_endpoint_returns_schema
   └── test_health_endpoint

   test_training.py
   ├── test_full_pipeline_runs_on_sample_data
   └── test_model_beats_baseline

   test_drift.py
   ├── test_drift_detection_on_identical_data_returns_no_drift
   └── test_drift_detection_on_shifted_data_returns_drift
   ```

   **ML-specific tests**:
   ```
   ├── test_feature_pipeline_matches_training_and_serving
   ├── test_model_is_deterministic
   ├── test_expensive_house_predicted_higher_than_cheap
   ├── test_shap_explanations_are_consistent_across_runs
   └── test_model_handles_edge_case_inputs
   ```

2. **Set up CI/CD pipelines** -- `.github/workflows/`

3. **Pre-commit hooks**

### Skills Learned

- Testing feature engineering pipelines
- Testing model interpretability (SHAP consistency)
- Behavioral tests for regression models
- Testing data drift detection

---

## Phase 8: Deployment

**Duration:** 2-3 days
**Objective:** Get the application running on the internet.

### Tasks

1. **Choose a platform** (same considerations as previous projects)

   | Platform | Pros | Cons | Cost |
   |----------|------|------|------|
   | **Railway** | Simplest deployment | Less control | Free tier available |
   | **GCP Cloud Run** | Auto-scaling, pay-per-use | More setup | Free tier generous |
   | **AWS ECS/Fargate** | Enterprise standard | Most complex | Free tier limited |

   **Tabular model advantage:** small model files (~10MB), fast inference, low memory -- cheap to deploy.

2. **Prepare for deployment**
   - All config from environment variables
   - Health check endpoint works
   - Models baked into Docker image
   - Feature validation prevents garbage predictions

3. **Deploy API and frontend**

4. **Load testing**
   - Target: 200 concurrent users, p95 latency < 100ms (tabular models are fast)
   - Test SHAP explanation generation under load

5. **Set up infrastructure**

### Skills Learned

- Deploying lightweight tabular ML models
- Load testing with SHAP explanation overhead
- Cost optimization for simple models

---

## Phase 9: Monitoring & Observability

**Duration:** 3-4 days
**Objective:** Monitor predictions, features, and detect data drift with Evidently.

### Tasks

1. **Structured logging**
   - Every prediction logged as structured JSON:
     ```json
     {
       "timestamp": "2026-03-22T10:30:00Z",
       "request_id": "abc-123",
       "predicted_price": 215000.0,
       "model_version": "v1.0",
       "latency_ms": 12,
       "top_feature": "OverallQual",
       "input_feature_count": 15,
       "status": "success"
     }
     ```

2. **Prometheus metrics** -- `src/monitoring/metrics.py`
   - `prediction_requests_total` -- counter by status
   - `prediction_latency_seconds` -- histogram
   - `predicted_price` -- histogram (track price distribution)
   - `shap_computation_seconds` -- histogram
   - `feature_values` -- histograms per key feature (detect input drift)
   - `model_info` -- gauge with version label

3. **Grafana dashboard** -- `grafana/dashboards/regression_monitoring.json`
   - Row 1: Request rate, error rate, latency percentiles
   - Row 2: Predicted price distribution over time, residual distribution (if ground truth available)
   - Row 3: Key feature distributions (living area, quality, year built)
   - Row 4: SHAP computation time, system metrics

4. **Data drift detection with Evidently** -- `src/monitoring/drift.py`
   - Evidently compares current production data against training reference data:
     ```python
     from evidently.report import Report
     from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

     class DriftDetector:
         def __init__(self, reference_data: pd.DataFrame):
             self.reference = reference_data

         def check_drift(self, current_data: pd.DataFrame) -> dict:
             report = Report(metrics=[DataDriftPreset()])
             report.run(
                 reference_data=self.reference,
                 current_data=current_data,
             )

             result = report.as_dict()
             drift_detected = result["metrics"][0]["result"]["dataset_drift"]
             drifted_features = [
                 col for col, info in result["metrics"][0]["result"]["drift_by_columns"].items()
                 if info["drift_detected"]
             ]

             return {
                 "drift_detected": drift_detected,
                 "drifted_features": drifted_features,
                 "drift_share": result["metrics"][0]["result"]["share_of_drifted_columns"],
             }
     ```

   - **Feature drift monitoring:**
     - Statistical tests per feature (KS test for numerical, chi-squared for categorical)
     - Detect when input distributions shift from training data
     - Example: if users start querying larger houses than the model was trained on

   - **Prediction drift monitoring:**
     - Track predicted price distribution over time
     - Alert if mean predicted price shifts more than 10% from baseline

   - **Schedule drift reports:**
     - Generate Evidently HTML reports daily/weekly
     - Store in MLflow as artifacts for review

5. **Alerting rules**
   - Error rate > 5% for 5 minutes -> alert
   - p95 latency > 200ms for 5 minutes -> alert
   - Data drift detected (> 30% of features drifted) -> alert
   - Predicted price distribution shift > 2 standard deviations -> alert
   - Feature out of training range (e.g., lot area > max seen in training) -> warning

### Skills Learned

- Data drift detection with Evidently (the most practical ML monitoring skill)
- Feature distribution monitoring for tabular data
- Statistical tests for drift (KS test, chi-squared)
- Generating and storing drift reports
- Regression-specific monitoring (prediction distribution, not class distribution)

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
        Phase 3: Model (cont.)      (3 days)
        Phase 4: Evaluation         (2 days)

Week 4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 4: Evaluation (cont.) (2 days)
        Phase 5: API & Serving      (3 days)

Week 5  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 5: API (cont.)        (1 day)
        Phase 6: Docker             (2 days)
        Phase 7: CI/CD              (2 days)

Week 6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 7: CI/CD (cont.)      (1 day)
        Phase 8: Deployment         (2 days)
        Phase 9: Monitoring         (2 days)

Week 7  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 9: Monitoring (cont.) (2 days)
        Buffer / catch-up           (3 days)
```

**Total: ~35 days (7 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] EDA for tabular/structured data (correlations, distributions, missing patterns)
- [ ] Missing value imputation strategies
- [ ] Feature engineering for tabular ML (interactions, aggregations, transformations)
- [ ] Encoding categorical variables (ordinal, target, one-hot)
- [ ] Preventing data leakage in feature pipelines
- [ ] Training and comparing multiple model families (Ridge, XGBoost, LightGBM)
- [ ] Cross-validation for regression
- [ ] Hyperparameter tuning with Optuna (Bayesian optimization)
- [ ] Regression evaluation metrics (RMSE, MAE, MAPE, R-squared)
- [ ] Residual analysis and diagnostics
- [ ] SHAP for model interpretability (global and local explanations)
- [ ] Dependence plots and feature interaction analysis
- [ ] Exporting full prediction pipelines (preprocessor + model + explainer)
- [ ] Building REST APIs with feature-based input validation
- [ ] SHAP explanations in production APIs
- [ ] Interactive Streamlit frontends with sliders and dropdowns
- [ ] Docker containerization for lightweight tabular models
- [ ] Testing feature engineering pipelines
- [ ] Testing model interpretability (SHAP consistency)
- [ ] CI/CD with GitHub Actions
- [ ] Cloud deployment for lightweight models
- [ ] Data drift detection with Evidently
- [ ] Feature distribution monitoring
- [ ] Prometheus metrics for regression models
- [ ] Grafana dashboards for regression monitoring
- [ ] Structured logging for tabular predictions

---

## Key Differences from Projects 1-2

| Aspect | Project 1 (CV) | Project 2 (NLP) | Project 3 (Tabular) |
|--------|----------------|-----------------|---------------------|
| Data type | Images | Text | Structured/tabular |
| Preprocessing | Resize, augment | Tokenize | Impute, encode, engineer |
| Key skill | Transfer learning | Fine-tuning | Feature engineering |
| Models | 1 (MobileNetV2) | 1 (DistilBERT) | 3+ compared |
| Task | Binary classification | Multi-class classification | Regression |
| Tuning | Manual | Manual | Optuna (automated) |
| Interpretability | N/A | Attention | SHAP (gold standard) |
| Model size | ~15MB | ~260MB | ~10MB |
| Inference speed | ~50ms | ~70ms | ~5ms |
| Monitoring | Prediction drift | Text drift | Feature drift (Evidently) |
| Framework | PyTorch | Hugging Face | scikit-learn ecosystem |

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Create the project directory
mkdir house-price-prediction && cd house-price-prediction
git init

# 2. Create the folder structure
mkdir -p configs data/{raw,processed,splits,reference} notebooks \
  src/{data,model,training,serving,monitoring,frontend} \
  tests/{unit,integration} docker .github/workflows \
  grafana/dashboards prometheus scripts

# 3. Verify key libraries are available
python -c "import xgboost; print(f'XGBoost {xgboost.__version__}')"
python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__}')"
python -c "import shap; print(f'SHAP {shap.__version__}')"
python -c "import optuna; print(f'Optuna {optuna.__version__}')"

# 4. Start writing DESIGN_DOC.md
```
