# End-to-End ML Project Plan: Energy Demand Forecasting

## Goal

Build a **production-grade** energy demand forecasting system that predicts electricity
consumption for the **next 24 hours** using LSTM, Prophet, and ensemble methods. This
project introduces time series forecasting, temporal cross-validation, scheduled
retraining with Airflow, and the unique challenges of deploying models that must be
continuously updated as new data arrives.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why energy forecasting? | Time series forecasting is critical in energy, finance, logistics, and operations. Energy is a clean domain: clear seasonality, measurable accuracy, real business impact. |
| Why multiple models? | Prophet captures trend + seasonality with minimal tuning. LSTM captures complex nonlinear patterns. Ensembling combines their strengths. |
| Why Airflow? | Forecasting models go stale fast. In production, retraining must be automated and scheduled. Airflow is the industry standard for ML orchestration. |
| How is this different from Projects 1-3? | Time series domain introduces temporal dependencies, walk-forward validation (no random splits!), scheduled retraining, and Airflow for orchestration. |

---

## Architecture Overview

```
                                        ┌───────────────┐
                                        │   Streamlit   │
                                        │  Dashboard    │
                                        │(time series   │
                                        │  plots)       │
                                        └──────┬────────┘
                                               │ HTTP
                                               ▼
                    ┌───────────────────────────────────────────┐
                    │              FastAPI Server               │
                    │                                          │
                    │  POST /forecast   GET /historical        │
                    │  GET /health      GET /accuracy          │
                    └──────────┬───────────────┬───────────────┘
                               │               │
                    ┌──────────▼──────┐  ┌─────▼──────────────┐
                    │  Ensemble Model │  │  Historical Data   │
                    │  ┌───────────┐  │  │  Store (SQLite/    │
                    │  │  Prophet  │  │  │  PostgreSQL)       │
                    │  ├───────────┤  │  └────────────────────┘
                    │  │   LSTM    │  │
                    │  └───────────┘  │
                    └─────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌──────────┐    ┌──────────┐
        │  Airflow │   │Prometheus│    │  MLflow  │
        │(scheduled│   │ Metrics  │    │ Registry │
        │retrain)  │   └────┬─────┘    └──────────┘
        └──────────┘        ▼
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
| Deep Learning | PyTorch (LSTM) | Industry standard, flexible RNN implementation |
| Statistical Model | Prophet | Excellent for time series with seasonality + trend |
| Classical ML | scikit-learn | Preprocessing, metrics, pipelines |
| Orchestration | Apache Airflow | Industry standard for scheduling ML pipelines |
| Data Analysis | Pandas, NumPy | Time series manipulation essentials |
| Visualization | Matplotlib, Plotly | Interactive time series plots |
| Experiment Tracking | MLflow | Tracks params, metrics, artifacts, model registry |
| Data Versioning | DVC | Git for data -- essential for reproducibility |
| API Framework | FastAPI | Fast, modern, auto-generates OpenAPI docs |
| Frontend | Streamlit | Interactive time series dashboards |
| Containerization | Docker + docker-compose | Deployment standard everywhere |
| CI/CD | GitHub Actions | Free, integrated with GitHub |
| Monitoring | Prometheus + Grafana | Industry standard observability |
| Testing | pytest | Python standard |
| Linting | Ruff + mypy | Fast linting + type checking |

---

## Project Structure

```
energy-forecasting/
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
│   ├── data_config.yaml             # Data paths, time ranges
│   └── airflow_config.yaml          # Retraining schedule configuration
│
├── data/                            # Git-ignored, DVC-tracked
│   ├── raw/                         # Original downloaded time series
│   ├── processed/                   # Cleaned, feature-engineered data
│   └── splits/                      # Walk-forward validation splits
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis (seasonality, trends)
│   ├── 02_feature_engineering.ipynb # Lag features, rolling windows, time features
│   ├── 03_prophet_baseline.ipynb    # Prophet model development
│   ├── 04_lstm_model.ipynb          # LSTM model development
│   ├── 05_ensemble.ipynb            # Ensemble strategy
│   └── 06_evaluation.ipynb          # Walk-forward evaluation & error analysis
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py              # Download dataset script
│   │   ├── validate.py              # Data integrity checks (gaps, outliers)
│   │   ├── preprocess.py            # Cleaning, resampling, imputation
│   │   └── features.py              # Lag features, rolling stats, time features
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── prophet_model.py         # Prophet wrapper
│   │   ├── lstm_model.py            # LSTM architecture and training
│   │   ├── ensemble.py              # Ensemble strategy (weighted average)
│   │   └── export.py                # Export models for serving
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                 # Training pipeline (all models)
│   │   ├── evaluate.py              # Evaluation metrics & visualization
│   │   ├── walk_forward.py          # Walk-forward cross-validation
│   │   └── retrain.py               # Retraining logic for Airflow
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py                   # FastAPI application
│   │   ├── schemas.py               # Request/response Pydantic models
│   │   ├── middleware.py            # Rate limiting, auth, CORS
│   │   └── predict.py               # Forecast generation logic
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Prometheus metric definitions
│   │   ├── accuracy.py              # Forecast accuracy tracking
│   │   └── drift.py                 # Temporal pattern drift detection
│   │
│   ├── orchestration/
│   │   ├── __init__.py
│   │   └── dags/
│   │       ├── retrain_dag.py       # Airflow DAG for scheduled retraining
│   │       └── data_ingest_dag.py   # Airflow DAG for data ingestion
│   │
│   └── frontend/
│       └── app.py                   # Streamlit dashboard
│
├── tests/
│   ├── unit/
│   │   ├── test_features.py         # Time feature engineering tests
│   │   ├── test_preprocessing.py    # Data cleaning tests
│   │   ├── test_prophet.py          # Prophet model tests
│   │   ├── test_lstm.py             # LSTM model tests
│   │   ├── test_ensemble.py         # Ensemble logic tests
│   │   ├── test_schemas.py          # API schema validation
│   │   └── test_walk_forward.py     # Walk-forward split tests
│   ├── integration/
│   │   ├── test_api.py              # Full API forecast pipeline
│   │   ├── test_training.py         # Training runs without error
│   │   └── test_retrain.py          # Retraining pipeline test
│   └── conftest.py                  # Shared fixtures (sample time series)
│
├── docker/
│   ├── Dockerfile.api               # Multi-stage build for API
│   ├── Dockerfile.frontend          # Streamlit container
│   ├── Dockerfile.training          # Training environment
│   └── Dockerfile.airflow           # Airflow scheduler
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
│       └── forecasting_monitoring.json  # Pre-configured dashboard
│
├── prometheus/
│   └── prometheus.yml               # Scrape config
│
└── scripts/
    ├── setup.sh                     # One-command project setup
    ├── train.sh                     # Run training with default config
    ├── backtest.sh                  # Run walk-forward backtesting
    └── deploy.sh                    # Deploy to cloud
```

---

## Phase 1: Project Setup & Design Doc

**Duration:** 1-2 days
**Objective:** Define what you are building before writing any code.

### Tasks

1. **Write `DESIGN_DOC.md`**
   - **Problem statement:** "Given historical energy consumption data, forecast electricity demand for the next 24 hours at hourly granularity"
   - **Success criteria:**
     - MAPE (Mean Absolute Percentage Error) < 5% on test period
     - MAE < 50 kWh on hourly predictions
     - Forecast generation latency < 2 seconds for 24-hour horizon
     - API handles 50 concurrent requests
     - Model automatically retrains weekly
     - Forecast accuracy does not degrade more than 20% between retraining cycles
   - **Out of scope:** real-time streaming predictions (sub-hourly), multi-building forecasting, weather data integration (Phase 2 stretch goal), anomaly detection
   - **Risks:**
     - Missing data gaps in time series (sensor failures)
     - Concept drift (seasonal changes, building renovations)
     - Holiday effects (irregular spikes/drops)
     - Model staleness between retraining cycles

2. **Initialize the repository**
   - `git init`, create `.gitignore`
   - Create `pyproject.toml` with all dependencies:
     ```toml
     [project]
     name = "energy-forecasting"
     version = "0.1.0"
     requires-python = ">=3.11"
     dependencies = [
         "torch>=2.1.0",
         "prophet>=1.1.5",
         "scikit-learn>=1.3.0",
         "pandas>=2.1.0",
         "numpy>=1.26.0",
         "matplotlib>=3.8.0",
         "plotly>=5.18.0",
         "mlflow>=2.9.0",
         "apache-airflow>=2.8.0",
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
   - Pre-commit hooks
   - Verify Airflow can be installed locally (or plan for Docker-only Airflow)

### Skills Learned

- Writing ML design documents for time series forecasting
- Understanding success criteria for forecasting (MAPE, MAE, forecast horizon)
- Planning for scheduled retraining from the start

---

## Phase 2: Data Pipeline

**Duration:** 4-5 days
**Objective:** Get clean, feature-engineered time series data ready for modeling.

### Tasks

1. **Download the dataset** -- `src/data/download.py`
   - **UCI Household Electric Power Consumption** dataset:
     - 2,075,259 measurements over ~4 years (Dec 2006 - Nov 2010)
     - 1-minute sampling interval
     - Features: global active power, reactive power, voltage, intensity, sub-metering
   ```python
   import pandas as pd
   from urllib.request import urlretrieve

   URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"

   def download_dataset(output_dir="data/raw/"):
       urlretrieve(URL, f"{output_dir}/household_power_consumption.zip")
       # Extract and parse
       df = pd.read_csv(
           f"{output_dir}/household_power_consumption.txt",
           sep=";",
           parse_dates={"datetime": ["Date", "Time"]},
           dayfirst=True,
           low_memory=False,
           na_values=["?"],
       )
       df.set_index("datetime", inplace=True)
       df.to_parquet(f"{output_dir}/energy_raw.parquet")
   ```

2. **Set up DVC**
   - Track `data/raw/` with DVC
   - **Time series specific:** also version your data by time range (e.g., "data up to 2009-12-31" for reproducible backtesting)

3. **Exploratory Data Analysis (EDA)** -- `notebooks/01_eda.ipynb`

   - **Time series overview:**
     ```python
     # Resample to hourly for cleaner visualization
     hourly = df["Global_active_power"].resample("h").mean()

     # Full time series plot
     fig, ax = plt.subplots(figsize=(16, 4))
     hourly.plot(ax=ax)
     ax.set_title("Hourly Energy Consumption (2006-2010)")
     ax.set_ylabel("kW")
     ```

   - **Seasonality analysis** (the most important EDA step for time series):
     ```python
     # Daily pattern: average by hour of day
     hourly_pattern = df.groupby(df.index.hour)["Global_active_power"].mean()
     hourly_pattern.plot(kind="bar", title="Average Energy by Hour of Day")

     # Weekly pattern: average by day of week
     daily = df["Global_active_power"].resample("D").mean()
     weekly_pattern = daily.groupby(daily.index.dayofweek).mean()
     weekly_pattern.plot(kind="bar", title="Average Energy by Day of Week")

     # Yearly pattern: average by month
     monthly_pattern = daily.groupby(daily.index.month).mean()
     monthly_pattern.plot(kind="bar", title="Average Energy by Month")
     ```

   - **Decomposition** (separate trend, seasonality, and residuals):
     ```python
     from statsmodels.tsa.seasonal import seasonal_decompose

     # Decompose the daily time series
     result = seasonal_decompose(daily, model="additive", period=365)
     result.plot()
     plt.tight_layout()
     ```

   - **Stationarity test:**
     ```python
     from statsmodels.tsa.stattools import adfuller

     adf_result = adfuller(hourly.dropna())
     print(f"ADF Statistic: {adf_result[0]:.4f}")
     print(f"p-value: {adf_result[1]:.4f}")
     # If p < 0.05, the series is stationary (good for modeling)
     ```

   - **Autocorrelation analysis:**
     ```python
     from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
     plot_acf(hourly.dropna(), lags=72, ax=axes[0])   # 3 days
     plot_pacf(hourly.dropna(), lags=72, ax=axes[1])
     ```
     - Expect strong autocorrelation at lags 24 (daily cycle) and 168 (weekly cycle)

   - **Missing data analysis:**
     - Count missing values and their patterns
     - Identify gaps (consecutive missing hours/days)
     - Visualize missing data pattern:
       ```python
       missing = hourly.isnull()
       fig, ax = plt.subplots(figsize=(16, 2))
       ax.fill_between(missing.index, 0, missing.astype(int), color="red", alpha=0.5)
       ax.set_title("Missing Data Gaps")
       ```

   - **Outlier detection:**
     - Z-score method or IQR method for hourly values
     - Check for negative power values (sensor errors)
     - Check for unrealistically high values

4. **Data preprocessing** -- `src/data/preprocess.py`
   - **Resample to hourly** (from 1-minute to 1-hour granularity):
     ```python
     def resample_to_hourly(df):
         """Resample minute-level data to hourly, handling missing values."""
         hourly = df.resample("h").agg({
             "Global_active_power": "mean",
             "Global_reactive_power": "mean",
             "Voltage": "mean",
             "Global_intensity": "mean",
             "Sub_metering_1": "sum",
             "Sub_metering_2": "sum",
             "Sub_metering_3": "sum",
         })
         return hourly
     ```

   - **Handle missing data** (critical for time series -- you cannot just drop rows):
     ```python
     def impute_missing(series, method="interpolate"):
         """Impute missing values in time series."""
         if method == "interpolate":
             # Linear interpolation for short gaps (< 6 hours)
             return series.interpolate(method="linear", limit=6)
         elif method == "seasonal":
             # For longer gaps, use same hour from previous week
             filled = series.copy()
             for idx in filled[filled.isnull()].index:
                 same_hour_prev_week = idx - pd.Timedelta(weeks=1)
                 if same_hour_prev_week in filled.index:
                     filled[idx] = filled[same_hour_prev_week]
             return filled
     ```
   - **Why not drop missing rows?** In time series, every row represents a specific time. Dropping rows creates temporal gaps that break lag features and LSTM sequences.

   - **Handle outliers:**
     - Cap extreme values at the 99.5th percentile
     - Replace negative power values with interpolation
     - Document every decision

5. **Feature engineering** -- `src/data/features.py`
   - This is where time series ML differs fundamentally from tabular ML:
     ```python
     class TimeSeriesFeatureEngineer:
         """Create features from time series data."""

         def transform(self, df):
             target_col = "Global_active_power"

             # === Time-based features ===
             df["hour"] = df.index.hour
             df["day_of_week"] = df.index.dayofweek
             df["day_of_month"] = df.index.day
             df["month"] = df.index.month
             df["week_of_year"] = df.index.isocalendar().week.astype(int)
             df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
             df["quarter"] = df.index.quarter

             # Cyclical encoding (so hour 23 is close to hour 0)
             df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
             df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
             df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
             df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
             df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
             df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

             # === Lag features ===
             # Same hour yesterday
             df["lag_24h"] = df[target_col].shift(24)
             # Same hour last week
             df["lag_168h"] = df[target_col].shift(168)
             # 1 hour ago, 2 hours ago, 3 hours ago
             for lag in [1, 2, 3, 6, 12]:
                 df[f"lag_{lag}h"] = df[target_col].shift(lag)

             # === Rolling window features ===
             # Rolling statistics over the past 24 hours
             df["rolling_24h_mean"] = df[target_col].rolling(24).mean()
             df["rolling_24h_std"] = df[target_col].rolling(24).std()
             df["rolling_24h_min"] = df[target_col].rolling(24).min()
             df["rolling_24h_max"] = df[target_col].rolling(24).max()

             # Rolling statistics over the past 7 days
             df["rolling_7d_mean"] = df[target_col].rolling(168).mean()
             df["rolling_7d_std"] = df[target_col].rolling(168).std()

             # === Difference features ===
             # Hour-over-hour change
             df["diff_1h"] = df[target_col].diff(1)
             # Day-over-day change (same hour)
             df["diff_24h"] = df[target_col].diff(24)

             # === Holiday/special event flags ===
             # Simple approach: mark known holidays
             # (In production, use a holiday calendar library)
             df["is_holiday"] = 0  # Placeholder

             return df
     ```

   - **Critical: lag features create data leakage if not handled correctly.** When creating features for the forecast horizon, you can only use data that would be available at prediction time. Lag_1h is NOT available when predicting 24 hours ahead -- only lag_24h and beyond are valid.

   - **Feature matrix for multi-step forecasting:**
     ```python
     def create_forecast_features(df, horizon=24):
         """Create features that are valid for the full forecast horizon.

         When predicting 24 hours ahead:
         - lag_24h: available (it is the current observation)
         - lag_1h through lag_23h: NOT available at all forecast steps
         - Time features (hour, day_of_week): always available (deterministic)
         - Rolling features: only up to the forecast origin point
         """
         valid_features = [
             # Time features (always known for future)
             "hour_sin", "hour_cos", "dow_sin", "dow_cos",
             "month_sin", "month_cos", "is_weekend",
             # Lag features (available at forecast origin)
             "lag_24h", "lag_168h",
             # Rolling features (computed at forecast origin)
             "rolling_24h_mean", "rolling_24h_std",
             "rolling_7d_mean", "rolling_7d_std",
         ]
         return valid_features
     ```

6. **Define data splits for time series** -- **NO RANDOM SPLITTING**
   - **Why random splitting is wrong for time series:** if you randomly assign January 15 to training and January 14 to test, your model sees the future during training. This is data leakage and inflates metrics.
   - **Correct approach: temporal split:**
     ```python
     def temporal_split(df, train_end, val_end):
         """Split time series data chronologically.

         Example:
           2007-2009 = training
           2009-01 to 2009-06 = validation
           2009-07 to 2010-11 = test
         """
         train = df[:train_end]
         val = df[train_end:val_end]
         test = df[val_end:]
         return train, val, test

     train, val, test = temporal_split(
         df,
         train_end="2009-01-01",
         val_end="2009-07-01",
     )
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
     ```

### Skills Learned

- EDA for time series (seasonality, trends, stationarity, autocorrelation)
- Time series decomposition (trend + seasonality + residual)
- Imputation strategies for time series (interpolation, seasonal filling -- never drop!)
- Feature engineering for time series (lags, rolling windows, cyclical encoding)
- Understanding data leakage in time series (no random splits, careful with lag features)
- Temporal train/val/test splitting

---

## Phase 3: Model Development & Experiment Tracking

**Duration:** 5-7 days
**Objective:** Build Prophet baseline, LSTM model, ensemble, and validate with walk-forward cross-validation.

### Tasks

1. **Prophet baseline** -- `src/model/prophet_model.py`
   - Prophet is excellent for time series with strong seasonality and trend:
     ```python
     from prophet import Prophet

     class ProphetForecaster:
         def __init__(self, config):
             self.config = config
             self.model = None

         def fit(self, df):
             """Fit Prophet model.

             Prophet expects columns: 'ds' (datetime) and 'y' (target).
             """
             prophet_df = df.reset_index().rename(
                 columns={"datetime": "ds", "Global_active_power": "y"}
             )

             self.model = Prophet(
                 yearly_seasonality=True,
                 weekly_seasonality=True,
                 daily_seasonality=True,
                 changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10.0,
             )

             # Add custom seasonality for working hours
             self.model.add_seasonality(
                 name="working_hours",
                 period=1,          # Daily
                 fourier_order=8,   # Capture complex daily pattern
             )

             self.model.fit(prophet_df)

         def predict(self, periods=24, freq="h"):
             """Generate forecast for the next `periods` hours."""
             future = self.model.make_future_dataframe(
                 periods=periods, freq=freq
             )
             forecast = self.model.predict(future)
             return forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
     ```

   - **Why start with Prophet:**
     - No feature engineering needed (it learns seasonality automatically)
     - Produces uncertainty intervals out of the box
     - Fast to fit (seconds, not minutes)
     - Sets a strong baseline to beat

2. **LSTM model** -- `src/model/lstm_model.py`
   - Build an LSTM for sequence-to-sequence forecasting:
     ```python
     import torch
     import torch.nn as nn

     class LSTMForecaster(nn.Module):
         def __init__(
             self,
             input_size: int,      # Number of input features
             hidden_size: int = 64,
             num_layers: int = 2,
             dropout: float = 0.2,
             forecast_horizon: int = 24,
         ):
             super().__init__()
             self.hidden_size = hidden_size
             self.num_layers = num_layers
             self.forecast_horizon = forecast_horizon

             self.lstm = nn.LSTM(
                 input_size=input_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 dropout=dropout,
                 batch_first=True,
             )

             self.fc = nn.Sequential(
                 nn.Linear(hidden_size, hidden_size // 2),
                 nn.ReLU(),
                 nn.Dropout(dropout),
                 nn.Linear(hidden_size // 2, forecast_horizon),
             )

         def forward(self, x):
             # x shape: (batch, sequence_length, input_size)
             lstm_out, (h_n, c_n) = self.lstm(x)
             # Use the last hidden state
             last_hidden = lstm_out[:, -1, :]
             # Predict all 24 hours at once
             forecast = self.fc(last_hidden)
             return forecast
     ```

   - **Data preparation for LSTM:**
     ```python
     class TimeSeriesDataset(torch.utils.data.Dataset):
         def __init__(self, data, features, target, lookback=168, horizon=24):
             """
             Args:
                 data: DataFrame with features and target
                 features: list of feature column names
                 target: target column name
                 lookback: number of past hours to use as input (7 days)
                 horizon: number of future hours to predict (24 hours)
             """
             self.lookback = lookback
             self.horizon = horizon
             self.X = data[features].values
             self.y = data[target].values

         def __len__(self):
             return len(self.X) - self.lookback - self.horizon + 1

         def __getitem__(self, idx):
             X = self.X[idx : idx + self.lookback]
             y = self.y[idx + self.lookback : idx + self.lookback + self.horizon]
             return torch.FloatTensor(X), torch.FloatTensor(y)
     ```

   - **Training the LSTM:**
     ```python
     def train_lstm(model, train_loader, val_loader, config):
         optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
             optimizer, patience=5, factor=0.5
         )
         criterion = nn.MSELoss()

         best_val_loss = float("inf")

         for epoch in range(config["epochs"]):
             model.train()
             train_loss = 0
             for X_batch, y_batch in train_loader:
                 optimizer.zero_grad()
                 y_pred = model(X_batch)
                 loss = criterion(y_pred, y_batch)
                 loss.backward()
                 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                 optimizer.step()
                 train_loss += loss.item()

             # Validation
             model.eval()
             val_loss = 0
             with torch.no_grad():
                 for X_batch, y_batch in val_loader:
                     y_pred = model(X_batch)
                     val_loss += criterion(y_pred, y_batch).item()

             val_loss /= len(val_loader)
             scheduler.step(val_loss)

             if val_loss < best_val_loss:
                 best_val_loss = val_loss
                 torch.save(model.state_dict(), "models/best_lstm.pt")

             mlflow.log_metrics({
                 "train_loss": train_loss / len(train_loader),
                 "val_loss": val_loss,
             }, step=epoch)
     ```

3. **Write training config** -- `configs/train_config.yaml`
   ```yaml
   target:
     column: Global_active_power
     resample_freq: "h"

   prophet:
     yearly_seasonality: true
     weekly_seasonality: true
     daily_seasonality: true
     changepoint_prior_scale: 0.05
     seasonality_prior_scale: 10.0

   lstm:
     lookback_hours: 168       # 7 days of history
     forecast_horizon: 24      # Predict next 24 hours
     hidden_size: 64
     num_layers: 2
     dropout: 0.2
     learning_rate: 0.001
     batch_size: 64
     epochs: 50
     early_stopping_patience: 10

   ensemble:
     method: weighted_average
     prophet_weight: 0.4
     lstm_weight: 0.6

   walk_forward:
     n_splits: 5
     train_size_days: 365     # 1 year minimum training
     test_size_days: 30       # Evaluate on 30-day windows
   ```

4. **Walk-forward cross-validation** -- `src/training/walk_forward.py`
   - **This is the correct way to validate time series models:**
     ```python
     def walk_forward_validation(df, model_fn, n_splits=5,
                                  train_size_days=365, test_size_days=30):
         """Walk-forward validation: train on expanding window, test on next period.

         Split 1: Train [day 0 → day 365]     Test [day 365 → day 395]
         Split 2: Train [day 0 → day 395]     Test [day 395 → day 425]
         Split 3: Train [day 0 → day 425]     Test [day 425 → day 455]
         ...

         The training window EXPANDS. The test window SLIDES forward.
         The model NEVER sees future data.
         """
         results = []
         min_train_end = df.index[0] + pd.Timedelta(days=train_size_days)
         test_duration = pd.Timedelta(days=test_size_days)

         for split_idx in range(n_splits):
             train_end = min_train_end + split_idx * test_duration
             test_end = train_end + test_duration

             if test_end > df.index[-1]:
                 break

             train_data = df[:train_end]
             test_data = df[train_end:test_end]

             # Train model on training window
             model = model_fn()
             model.fit(train_data)

             # Generate forecasts for test period
             forecasts = []
             for forecast_origin in pd.date_range(
                 start=test_data.index[0],
                 end=test_data.index[-1] - pd.Timedelta(hours=24),
                 freq="24h",
             ):
                 forecast = model.predict(periods=24)
                 actual = test_data.loc[
                     forecast_origin:forecast_origin + pd.Timedelta(hours=23)
                 ]
                 forecasts.append({"forecast": forecast, "actual": actual})

             # Calculate metrics for this split
             split_metrics = evaluate_forecasts(forecasts)
             results.append(split_metrics)

             print(f"Split {split_idx + 1}: "
                   f"Train [{df.index[0].date()} → {train_end.date()}] "
                   f"Test [{train_end.date()} → {test_end.date()}] "
                   f"MAPE: {split_metrics['mape']:.2f}%")

         return results
     ```

   - **Why walk-forward validation is essential:**
     - Standard k-fold cross-validation violates temporal ordering
     - Walk-forward simulates real production usage: train on past, predict future
     - Reveals if model performance degrades over time (concept drift)

5. **Ensemble model** -- `src/model/ensemble.py`
   ```python
   class EnsembleForecaster:
       def __init__(self, prophet_model, lstm_model, weights=None):
           self.prophet = prophet_model
           self.lstm = lstm_model
           self.weights = weights or {"prophet": 0.4, "lstm": 0.6}

       def predict(self, periods=24):
           prophet_forecast = self.prophet.predict(periods=periods)
           lstm_forecast = self.lstm.predict(periods=periods)

           ensemble_forecast = (
               self.weights["prophet"] * prophet_forecast["yhat"].values
               + self.weights["lstm"] * lstm_forecast
           )

           return {
               "ensemble": ensemble_forecast,
               "prophet": prophet_forecast["yhat"].values,
               "lstm": lstm_forecast,
               "lower": prophet_forecast["yhat_lower"].values,  # Use Prophet uncertainty
               "upper": prophet_forecast["yhat_upper"].values,
           }

       def optimize_weights(self, val_data):
           """Find optimal weights using validation data."""
           from scipy.optimize import minimize_scalar

           def objective(prophet_weight):
               lstm_weight = 1 - prophet_weight
               blended = (prophet_weight * self.prophet_preds
                         + lstm_weight * self.lstm_preds)
               return mean_absolute_error(val_data, blended)

           result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
           self.weights = {
               "prophet": result.x,
               "lstm": 1 - result.x,
           }
           print(f"Optimal weights: Prophet={result.x:.2f}, LSTM={1-result.x:.2f}")
   ```

6. **Set up MLflow experiment tracking**
   - Log per-experiment: model type, hyperparameters, walk-forward CV metrics
   - Log artifacts: trained models, forecast plots, walk-forward results
   - Log time-specific metadata: training data range, test data range
   - **Register the best model** in MLflow Model Registry

7. **Run experiments** (track all in MLflow)

   | Experiment | What Changes | Expected Result |
   |-----------|-------------|----------------|
   | Prophet baseline | Default Prophet | MAPE ~8-10% |
   | Prophet tuned | Tuned seasonality priors | MAPE ~6-8% |
   | LSTM baseline | 168h lookback, 64 hidden | MAPE ~7-9% |
   | LSTM deeper | 128 hidden, 3 layers | MAPE ~6-8%, slower |
   | LSTM more features | Add sub-metering features | MAPE ~5-7% |
   | Ensemble equal | 50/50 Prophet + LSTM | MAPE ~5-6% |
   | Ensemble optimized | Optimized weights | MAPE ~4-5% |
   | Shorter lookback | LSTM with 72h lookback | Slightly worse, faster |

8. **Pick the best model** using MLflow UI
   - Compare walk-forward CV results across all models
   - Consider: accuracy vs training time vs inference speed vs complexity
   - Promote ensemble model to "Production" in MLflow registry

### Skills Learned

- Prophet for time series forecasting (seasonality, trend, uncertainty)
- LSTM architecture for sequence-to-sequence prediction
- Walk-forward cross-validation (the correct way to validate time series)
- Ensemble methods for combining forecasts
- Understanding lookback windows and forecast horizons
- Experiment tracking for time series models

---

## Phase 4: Evaluation & Error Analysis

**Duration:** 2-3 days
**Objective:** Thoroughly evaluate forecast accuracy and understand failure modes.

### Tasks

1. **Comprehensive evaluation** -- `src/training/evaluate.py`
   - Metrics on the **held-out test period**:
     - MAE (Mean Absolute Error) -- average error in kWh
     - MAPE (Mean Absolute Percentage Error) -- relative error
     - RMSE -- penalizes large forecast errors
     - Directional accuracy -- did we predict up/down correctly?
   ```python
   def evaluate_forecast(actual, predicted):
       mae = mean_absolute_error(actual, predicted)
       mape = np.mean(np.abs((actual - predicted) / actual)) * 100
       rmse = np.sqrt(mean_squared_error(actual, predicted))

       # Directional accuracy
       actual_direction = np.diff(actual) > 0
       pred_direction = np.diff(predicted) > 0
       dir_accuracy = np.mean(actual_direction == pred_direction) * 100

       return {
           "mae": mae,
           "mape": mape,
           "rmse": rmse,
           "directional_accuracy": dir_accuracy,
       }
   ```

2. **Forecast visualization** -- `notebooks/06_evaluation.ipynb`
   - **Actual vs Predicted time series plot** (the most important visualization):
     ```python
     import plotly.graph_objects as go

     fig = go.Figure()
     fig.add_trace(go.Scatter(
         x=test_dates, y=actual,
         name="Actual", line=dict(color="blue")
     ))
     fig.add_trace(go.Scatter(
         x=test_dates, y=predicted,
         name="Forecast", line=dict(color="red", dash="dash")
     ))
     fig.add_trace(go.Scatter(
         x=test_dates, y=upper_bound,
         fill=None, mode="lines", line=dict(width=0),
         showlegend=False
     ))
     fig.add_trace(go.Scatter(
         x=test_dates, y=lower_bound,
         fill="tonexty", mode="lines", line=dict(width=0),
         name="95% Confidence", fillcolor="rgba(255,0,0,0.1)"
     ))
     fig.update_layout(title="24-Hour Energy Demand Forecast vs Actual")
     ```

   - **Error by hour of day** (is the model worse at certain times?):
     ```python
     errors_by_hour = pd.DataFrame({
         "hour": test_data.index.hour,
         "abs_error": np.abs(actual - predicted),
     }).groupby("hour").mean()

     errors_by_hour.plot(kind="bar", title="Mean Absolute Error by Hour of Day")
     ```
     - Expect higher errors during morning ramp-up (7-9 AM) and evening peak (6-8 PM)

   - **Error by day of week** (are weekends harder to predict?):
     ```python
     errors_by_dow = pd.DataFrame({
         "day": test_data.index.day_name(),
         "abs_error": np.abs(actual - predicted),
     }).groupby("day").mean()
     ```

   - **Error by forecast horizon** (does accuracy degrade from hour 1 to hour 24?):
     ```python
     # For multi-step forecasts, plot MAPE for each hour ahead
     horizon_errors = []
     for h in range(24):
         mape_h = np.mean(np.abs((actual_h[h] - pred_h[h]) / actual_h[h])) * 100
         horizon_errors.append(mape_h)

     plt.bar(range(24), horizon_errors)
     plt.xlabel("Hours Ahead")
     plt.ylabel("MAPE (%)")
     plt.title("Forecast Error by Horizon (hours ahead)")
     ```
     - Expect error to increase with forecast horizon (hour 1 is easier than hour 24)

   - **Error on special days:**
     - Performance on holidays vs normal days
     - Performance during extreme weather (if weather data available)
     - Performance during data gaps (post-imputation)

3. **Model comparison visualization**
   - Side-by-side: Prophet vs LSTM vs Ensemble for the same test period
   - Component contribution: when does Prophet help? when does LSTM help?
   - Identify cases where the ensemble fails but one model succeeds

4. **Performance benchmarking**
   - Prophet forecast generation: expected < 1 second for 24 hours
   - LSTM forecast generation: expected < 100ms for 24 hours
   - Ensemble: sum of both, but models can run in parallel
   - Memory footprint: Prophet (~50MB), LSTM (~5MB)

5. **Write `MODEL_CARD.md`**
   - Model type: ensemble of Prophet + LSTM
   - Training data: 3 years of hourly energy consumption
   - Walk-forward CV results with confidence intervals
   - Known limitations:
     - Accuracy degrades for forecast hours 18-24
     - Poor performance on holidays (not enough holiday data)
     - Assumes no sudden changes in building usage patterns
     - Trained on single household -- may not generalize to commercial buildings
   - Retraining strategy: weekly with expanding window

### Skills Learned

- Time series evaluation metrics (MAPE, directional accuracy)
- Forecast visualization with confidence intervals
- Error analysis by time components (hour, day, season)
- Evaluating multi-horizon forecasts (hour 1 vs hour 24 accuracy)
- Comparing ensemble components

---

## Phase 5: API & Serving Layer

**Duration:** 3-4 days
**Objective:** Build a forecast API with both prediction and historical data endpoints.

### Tasks

1. **Define API schemas** -- `src/serving/schemas.py`
   ```python
   from pydantic import BaseModel, Field
   from datetime import datetime

   class ForecastRequest(BaseModel):
       forecast_origin: datetime | None = Field(
           None, description="Start point for forecast (default: now)"
       )
       horizon_hours: int = Field(
           24, ge=1, le=72, description="Number of hours to forecast"
       )

   class HourlyForecast(BaseModel):
       timestamp: datetime
       predicted_kw: float
       lower_bound: float           # 95% confidence lower
       upper_bound: float           # 95% confidence upper
       prophet_kw: float            # Prophet component
       lstm_kw: float               # LSTM component

   class ForecastResponse(BaseModel):
       forecast_origin: datetime
       horizon_hours: int
       forecasts: list[HourlyForecast]
       model_version: str
       generation_latency_ms: float
       ensemble_weights: dict[str, float]

   class HistoricalDataResponse(BaseModel):
       start: datetime
       end: datetime
       data: list[dict]             # [{"timestamp": ..., "actual_kw": ..., "predicted_kw": ...}]

   class AccuracyResponse(BaseModel):
       period: str                  # "last_24h", "last_7d", "last_30d"
       mae: float
       mape: float
       rmse: float
       directional_accuracy: float

   class HealthResponse(BaseModel):
       status: str
       model_loaded: bool
       model_version: str
       last_retrain: datetime
       uptime_seconds: float
       data_freshness: str          # How old is the latest data point
   ```

2. **Build FastAPI application** -- `src/serving/app.py`
   - `POST /forecast` -- generate energy demand forecast
   - `GET /historical` -- retrieve historical data with forecasts overlaid
   - `GET /accuracy` -- show recent forecast accuracy (backtested)
   - `GET /health` -- health check with data freshness info
   - `GET /metrics` -- Prometheus metrics endpoint
   ```python
   from fastapi import FastAPI
   from contextlib import asynccontextmanager

   @asynccontextmanager
   async def lifespan(app: FastAPI):
       # Load models at startup
       app.state.prophet = load_prophet_model("models/prophet/")
       app.state.lstm = load_lstm_model("models/lstm/best_lstm.pt")
       app.state.ensemble_weights = load_weights("models/ensemble_weights.json")
       app.state.scaler = joblib.load("models/scaler.joblib")
       app.state.last_retrain = load_retrain_timestamp()
       yield

   app = FastAPI(title="Energy Demand Forecasting API", lifespan=lifespan)

   @app.post("/forecast", response_model=ForecastResponse)
   async def forecast(request: ForecastRequest):
       start = time.perf_counter()
       origin = request.forecast_origin or datetime.utcnow()

       # Fetch recent historical data for LSTM input
       recent_data = fetch_recent_data(hours=168)  # 7 days

       # Generate forecasts from both models
       prophet_forecast = app.state.prophet.predict(
           periods=request.horizon_hours
       )
       lstm_forecast = predict_lstm(
           app.state.lstm, recent_data, app.state.scaler,
           horizon=request.horizon_hours
       )

       # Ensemble
       w = app.state.ensemble_weights
       ensemble = (w["prophet"] * prophet_forecast
                   + w["lstm"] * lstm_forecast)

       latency = (time.perf_counter() - start) * 1000

       forecasts = [
           HourlyForecast(
               timestamp=origin + timedelta(hours=h+1),
               predicted_kw=float(ensemble[h]),
               lower_bound=float(prophet_lower[h]),
               upper_bound=float(prophet_upper[h]),
               prophet_kw=float(prophet_forecast[h]),
               lstm_kw=float(lstm_forecast[h]),
           )
           for h in range(request.horizon_hours)
       ]

       return ForecastResponse(
           forecast_origin=origin,
           horizon_hours=request.horizon_hours,
           forecasts=forecasts,
           model_version=MODEL_VERSION,
           generation_latency_ms=latency,
           ensemble_weights=w,
       )

   @app.get("/accuracy", response_model=AccuracyResponse)
   async def accuracy(period: str = "last_7d"):
       """Show how accurate recent forecasts have been."""
       metrics = compute_recent_accuracy(period)
       return AccuracyResponse(**metrics)
   ```

3. **Build Streamlit dashboard** -- `src/frontend/app.py`
   - Interactive time series dashboard:
     ```python
     import streamlit as st
     import plotly.graph_objects as go

     st.title("Energy Demand Forecast Dashboard")

     # Sidebar controls
     horizon = st.sidebar.slider("Forecast Horizon (hours)", 1, 72, 24)
     show_components = st.sidebar.checkbox("Show Model Components", False)

     # Fetch forecast
     response = requests.post(f"{API_URL}/forecast",
                              json={"horizon_hours": horizon})
     forecast = response.json()

     # Main chart: historical + forecast
     fig = go.Figure()

     # Historical data (last 7 days)
     historical = requests.get(f"{API_URL}/historical?days=7").json()
     fig.add_trace(go.Scatter(
         x=[d["timestamp"] for d in historical["data"]],
         y=[d["actual_kw"] for d in historical["data"]],
         name="Actual", line=dict(color="blue")
     ))

     # Forecast
     fig.add_trace(go.Scatter(
         x=[f["timestamp"] for f in forecast["forecasts"]],
         y=[f["predicted_kw"] for f in forecast["forecasts"]],
         name="Forecast", line=dict(color="red", dash="dash")
     ))

     # Confidence interval
     fig.add_trace(go.Scatter(
         x=[f["timestamp"] for f in forecast["forecasts"]],
         y=[f["upper_bound"] for f in forecast["forecasts"]],
         fill=None, mode="lines", line=dict(width=0),
         showlegend=False
     ))
     fig.add_trace(go.Scatter(
         x=[f["timestamp"] for f in forecast["forecasts"]],
         y=[f["lower_bound"] for f in forecast["forecasts"]],
         fill="tonexty", mode="lines", line=dict(width=0),
         name="95% Confidence", fillcolor="rgba(255,0,0,0.1)"
     ))

     if show_components:
         fig.add_trace(go.Scatter(
             x=[f["timestamp"] for f in forecast["forecasts"]],
             y=[f["prophet_kw"] for f in forecast["forecasts"]],
             name="Prophet", line=dict(color="green", dash="dot")
         ))
         fig.add_trace(go.Scatter(
             x=[f["timestamp"] for f in forecast["forecasts"]],
             y=[f["lstm_kw"] for f in forecast["forecasts"]],
             name="LSTM", line=dict(color="orange", dash="dot")
         ))

     fig.update_layout(
         title=f"{horizon}-Hour Energy Demand Forecast",
         xaxis_title="Time",
         yaxis_title="Energy (kW)",
     )
     st.plotly_chart(fig, use_container_width=True)

     # Accuracy metrics
     st.subheader("Recent Forecast Accuracy")
     col1, col2, col3, col4 = st.columns(4)
     accuracy = requests.get(f"{API_URL}/accuracy?period=last_7d").json()
     col1.metric("MAE", f"{accuracy['mae']:.2f} kW")
     col2.metric("MAPE", f"{accuracy['mape']:.1f}%")
     col3.metric("RMSE", f"{accuracy['rmse']:.2f} kW")
     col4.metric("Direction Acc.", f"{accuracy['directional_accuracy']:.0f}%")
     ```

4. **API middleware** -- `src/serving/middleware.py`
   - Input validation: forecast horizon limits, valid datetime ranges
   - Rate limiting, CORS, request ID, error handling
   - Cache recent forecasts (same origin -> same result within a time window)

5. **API tests**
   - Integration tests: full forecast request -> response
   - Validate forecast timestamps are in the future
   - Validate forecast values are physically reasonable (no negative energy)
   - Test historical data endpoint with various date ranges
   - Test accuracy endpoint

### Skills Learned

- Building forecast APIs (multi-step output, confidence intervals)
- Interactive time series dashboards with Plotly + Streamlit
- Caching strategies for forecast APIs
- Serving multiple models simultaneously (ensemble at inference time)
- Historical data endpoints for visualization

---

## Phase 6: Containerization

**Duration:** 2-3 days
**Objective:** Package everything into Docker containers, including Airflow.

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
   - Include both Prophet and LSTM model files
   - Prophet models can be large (~50MB serialized)

2. **Frontend Dockerfile** -- `docker/Dockerfile.frontend`

3. **Airflow Dockerfile** -- `docker/Dockerfile.airflow`
   - Based on `apache/airflow:2.8.0-python3.11`
   - Include your training and data scripts
   ```dockerfile
   FROM apache/airflow:2.8.0-python3.11
   COPY requirements-airflow.txt /opt/airflow/
   RUN pip install -r /opt/airflow/requirements-airflow.txt
   COPY src/ /opt/airflow/src/
   COPY configs/ /opt/airflow/configs/
   COPY src/orchestration/dags/ /opt/airflow/dags/
   ```

4. **docker-compose.yaml** -- orchestrate everything
   ```yaml
   services:
     api:
       build:
         context: .
         dockerfile: docker/Dockerfile.api
       ports: ["8000:8000"]
       environment:
         - PROPHET_MODEL_PATH=/app/models/prophet/
         - LSTM_MODEL_PATH=/app/models/lstm/best_lstm.pt
         - LOG_LEVEL=info
       volumes:
         - model-store:/app/models  # Shared with Airflow for model updates

     frontend:
       build:
         context: .
         dockerfile: docker/Dockerfile.frontend
       ports: ["8501:8501"]
       depends_on: [api]

     airflow-webserver:
       build:
         context: .
         dockerfile: docker/Dockerfile.airflow
       ports: ["8080:8080"]
       environment:
         - AIRFLOW__CORE__EXECUTOR=LocalExecutor
         - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db
       volumes:
         - model-store:/opt/airflow/models
       command: webserver

     airflow-scheduler:
       build:
         context: .
         dockerfile: docker/Dockerfile.airflow
       environment:
         - AIRFLOW__CORE__EXECUTOR=LocalExecutor
         - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////opt/airflow/airflow.db
       volumes:
         - model-store:/opt/airflow/models
       command: scheduler

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
     model-store:
   ```

5. **Verify the full stack locally**
   - API serves forecasts, dashboard displays them
   - Airflow scheduler runs, webserver accessible at :8080
   - Prometheus and Grafana operational

### Skills Learned

- Docker for multi-model serving (Prophet + LSTM)
- Containerizing Airflow for scheduled tasks
- Shared volumes for model updates between training and serving
- Multi-service orchestration with 5+ containers

---

## Phase 7: Testing & CI/CD

**Duration:** 2-3 days
**Objective:** Automate quality checks with time-series-specific tests.

### Tasks

1. **Write comprehensive tests**

   **Unit tests**:
   ```
   test_features.py
   ├── test_lag_features_have_correct_offset
   ├── test_rolling_features_window_size
   ├── test_cyclical_encoding_range  # sin/cos in [-1, 1]
   ├── test_time_features_match_datetime
   └── test_feature_engineer_handles_missing_values

   test_preprocessing.py
   ├── test_resample_to_hourly
   ├── test_imputation_fills_short_gaps
   ├── test_imputation_uses_seasonal_for_long_gaps
   └── test_outlier_capping

   test_prophet.py
   ├── test_prophet_output_length_matches_horizon
   ├── test_prophet_includes_uncertainty_intervals
   └── test_prophet_handles_missing_data

   test_lstm.py
   ├── test_lstm_output_shape_is_batch_x_horizon
   ├── test_lstm_predictions_are_positive
   ├── test_lstm_handles_variable_lookback
   └── test_lstm_loads_from_checkpoint

   test_ensemble.py
   ├── test_weights_sum_to_one
   ├── test_ensemble_output_is_weighted_average
   └── test_weight_optimization_improves_accuracy

   test_walk_forward.py
   ├── test_splits_are_chronological
   ├── test_no_future_data_in_training
   ├── test_correct_number_of_splits
   └── test_expanding_window_grows
   ```

   **Integration tests**:
   ```
   test_api.py
   ├── test_forecast_returns_correct_number_of_hours
   ├── test_forecast_timestamps_are_in_future
   ├── test_forecast_values_are_physically_reasonable
   ├── test_historical_endpoint_returns_data
   ├── test_accuracy_endpoint
   └── test_health_endpoint_shows_data_freshness

   test_training.py
   ├── test_prophet_trains_on_sample_data
   ├── test_lstm_trains_one_epoch
   └── test_ensemble_combines_forecasts

   test_retrain.py
   ├── test_retrain_pipeline_runs_end_to_end
   └── test_new_model_is_compared_to_production
   ```

   **Time series-specific tests**:
   ```
   ├── test_no_future_leakage_in_features
   ├── test_walk_forward_splits_are_temporal
   ├── test_forecast_degrades_gracefully_with_missing_recent_data
   ├── test_model_produces_different_forecasts_for_weekday_vs_weekend
   └── test_forecast_confidence_interval_widens_with_horizon
   ```

2. **Set up CI/CD pipelines**

3. **Pre-commit hooks**

### Skills Learned

- Testing time series pipelines (temporal correctness, no future leakage)
- Testing multi-model ensembles
- Testing scheduled retraining pipelines
- Behavioral tests for forecasting (weekday vs weekend patterns)

---

## Phase 8: Deployment

**Duration:** 2-3 days
**Objective:** Get the forecasting system running in production.

### Tasks

1. **Choose a platform**

   | Platform | Pros | Cons | Cost |
   |----------|------|------|------|
   | **Railway** | Simple | No Airflow support | Free tier |
   | **GCP Cloud Run + Cloud Scheduler** | Serverless, scheduled jobs | More setup | Free tier generous |
   | **AWS ECS + EventBridge** | Enterprise standard | Most complex | Free tier limited |

   **Forecasting-specific consideration:** you need BOTH a serving platform (API) and a scheduling platform (retraining). Cloud Scheduler or EventBridge replaces Airflow in cloud deployments.

2. **Prepare for deployment**
   - All config from environment variables
   - Health check with data freshness indicator
   - Models loaded at startup
   - Graceful handling of stale data (warn if data is > 24 hours old)

3. **Deploy API, frontend, and scheduler**
   - Option A: deploy Airflow separately (more infrastructure)
   - Option B: use cloud-native scheduler (Cloud Scheduler triggers a Cloud Run job)

4. **Load testing**
   - Target: 50 concurrent users requesting forecasts
   - Test with different horizons (1h, 24h, 72h)
   - Measure: API latency, model inference time, data fetch time

5. **Set up infrastructure**

### Skills Learned

- Deploying multi-component ML systems (API + scheduler)
- Cloud-native scheduling as an alternative to Airflow
- Deploying systems that require periodic updates

---

## Phase 9: Monitoring, Retraining & Observability

**Duration:** 4-5 days
**Objective:** Monitor forecast accuracy, detect degradation, and automate retraining.

### Tasks

1. **Structured logging**
   - Every forecast logged as structured JSON:
     ```json
     {
       "timestamp": "2026-03-22T10:30:00Z",
       "request_id": "abc-123",
       "forecast_origin": "2026-03-22T10:00:00Z",
       "horizon_hours": 24,
       "model_version": "v1.2",
       "generation_latency_ms": 450,
       "prophet_weight": 0.4,
       "lstm_weight": 0.6,
       "mean_forecast_kw": 1.85,
       "status": "success"
     }
     ```

2. **Prometheus metrics** -- `src/monitoring/metrics.py`
   - `forecast_requests_total` -- counter by horizon and status
   - `forecast_latency_seconds` -- histogram
   - `forecast_values_kw` -- histogram (predicted energy distribution)
   - `model_info` -- gauge with version, last_retrain timestamp
   - `data_freshness_seconds` -- gauge (time since latest data point)
   - `forecast_accuracy_mape` -- gauge (rolling accuracy of past forecasts)

3. **Forecast accuracy monitoring** -- `src/monitoring/accuracy.py`
   - **This is unique to forecasting:** you can measure accuracy after the fact
   ```python
   class ForecastAccuracyTracker:
       """Track forecast accuracy by comparing predictions to actuals
       once actual data becomes available."""

       def __init__(self, db):
           self.db = db

       def store_forecast(self, forecast_origin, forecasts):
           """Store forecast for later comparison."""
           self.db.save_forecasts(forecast_origin, forecasts)

       def update_accuracy(self):
           """Compare stored forecasts against now-available actuals."""
           # Find forecasts whose target period has passed
           pending = self.db.get_pending_forecasts()

           for forecast in pending:
               actual = self.db.get_actual_data(
                   start=forecast["target_start"],
                   end=forecast["target_end"],
               )

               if actual is not None:
                   mape = compute_mape(actual, forecast["predicted"])
                   mae = compute_mae(actual, forecast["predicted"])

                   # Update Prometheus metrics
                   FORECAST_MAPE.set(mape)
                   FORECAST_MAE.set(mae)

                   # Log accuracy
                   self.db.mark_forecast_evaluated(forecast["id"], {
                       "mape": mape, "mae": mae,
                   })

                   # Alert if accuracy degrades
                   if mape > MAPE_THRESHOLD:
                       trigger_alert(f"Forecast MAPE {mape:.1f}% exceeds threshold")
   ```

4. **Grafana dashboard** -- `grafana/dashboards/forecasting_monitoring.json`
   - Row 1: Request rate, error rate, generation latency
   - Row 2: **Rolling forecast accuracy** (MAPE over last 7 days, 30 days) -- the most important panel
   - Row 3: Forecast value distribution, confidence interval width over time
   - Row 4: Data freshness, model age (time since last retrain)
   - Row 5: System metrics (CPU, memory)
   - Row 6: Retraining history (last retrain time, accuracy before/after)

5. **Scheduled retraining with Airflow** -- `src/orchestration/dags/retrain_dag.py`
   ```python
   from airflow import DAG
   from airflow.operators.python import PythonOperator
   from datetime import datetime, timedelta

   default_args = {
       "owner": "ml-team",
       "depends_on_past": False,
       "email_on_failure": True,
       "retries": 1,
       "retry_delay": timedelta(minutes=5),
   }

   dag = DAG(
       "energy_model_retrain",
       default_args=default_args,
       description="Weekly retraining of energy forecasting models",
       schedule_interval="0 2 * * 0",  # Every Sunday at 2 AM
       start_date=datetime(2026, 1, 1),
       catchup=False,
   )

   def fetch_latest_data(**kwargs):
       """Pull the latest energy data into the training pipeline."""
       from src.data.download import fetch_recent_data
       data = fetch_recent_data(days=30)
       # Append to existing training data
       existing = pd.read_parquet("data/processed/energy_hourly.parquet")
       updated = pd.concat([existing, data]).drop_duplicates()
       updated.to_parquet("data/processed/energy_hourly.parquet")
       return len(data)

   def retrain_models(**kwargs):
       """Retrain Prophet and LSTM on updated data."""
       from src.training.retrain import retrain_pipeline
       metrics = retrain_pipeline(
           data_path="data/processed/energy_hourly.parquet",
           config_path="configs/train_config.yaml",
       )
       return metrics

   def evaluate_and_promote(**kwargs):
       """Compare new model to production, promote if better."""
       ti = kwargs["ti"]
       new_metrics = ti.xcom_pull(task_ids="retrain_models")

       production_metrics = get_production_model_metrics()

       if new_metrics["mape"] < production_metrics["mape"] * 1.05:
           # New model is at least as good (5% tolerance)
           promote_model(new_metrics["run_id"])
           reload_serving_model()
           return "promoted"
       else:
           log_warning(f"New model MAPE {new_metrics['mape']:.2f}% "
                       f"worse than production {production_metrics['mape']:.2f}%")
           return "rejected"

   fetch_task = PythonOperator(
       task_id="fetch_latest_data",
       python_callable=fetch_latest_data,
       dag=dag,
   )

   retrain_task = PythonOperator(
       task_id="retrain_models",
       python_callable=retrain_models,
       dag=dag,
   )

   promote_task = PythonOperator(
       task_id="evaluate_and_promote",
       python_callable=evaluate_and_promote,
       dag=dag,
   )

   fetch_task >> retrain_task >> promote_task
   ```

   - **Key principle:** never auto-promote a worse model. Compare against production and only promote if the new model is at least as good.

6. **Temporal drift detection** -- `src/monitoring/drift.py`
   - Detect when energy consumption patterns change:
     ```python
     class TemporalDriftDetector:
         def __init__(self, baseline_period_days=90):
             self.baseline = None

         def set_baseline(self, data):
             """Compute baseline statistics from a reference period."""
             self.baseline = {
                 "hourly_pattern": data.groupby(data.index.hour).mean(),
                 "weekly_pattern": data.groupby(data.index.dayofweek).mean(),
                 "overall_mean": data.mean(),
                 "overall_std": data.std(),
             }

         def check_drift(self, recent_data):
             """Compare recent data patterns to baseline."""
             recent_hourly = recent_data.groupby(recent_data.index.hour).mean()
             recent_weekly = recent_data.groupby(recent_data.index.dayofweek).mean()

             # Compare hourly patterns using cosine similarity
             hourly_sim = cosine_similarity(
                 self.baseline["hourly_pattern"].values.reshape(1, -1),
                 recent_hourly.values.reshape(1, -1),
             )[0][0]

             # Compare mean consumption
             mean_shift = abs(
                 recent_data.mean() - self.baseline["overall_mean"]
             ) / self.baseline["overall_std"]

             drift_detected = hourly_sim < 0.95 or mean_shift > 2.0

             return {
                 "drift_detected": drift_detected,
                 "hourly_pattern_similarity": hourly_sim,
                 "mean_consumption_shift_sigma": mean_shift,
                 "recommendation": "retrain" if drift_detected else "no action",
             }
     ```

7. **Alerting rules**
   - Forecast MAPE > 10% for 3 consecutive days -> alert + trigger retraining
   - Data freshness > 6 hours -> alert (data pipeline may be broken)
   - Model age > 14 days without retraining -> warning
   - Temporal pattern drift detected -> trigger retraining
   - Retraining failed -> alert immediately
   - New model rejected (worse than production) -> notify team

### Skills Learned

- Forecast accuracy monitoring (retrospective evaluation)
- Scheduled retraining with Airflow (DAGs, tasks, XCom)
- Model promotion gates (only promote if better)
- Temporal drift detection for time series
- Building retraining pipelines with automated quality checks
- Alerting on model staleness and accuracy degradation

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
        Phase 3: Model (cont.)      (4 days)
        Phase 4: Evaluation         (1 day)

Week 4  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 4: Evaluation (cont.) (2 days)
        Phase 5: API & Serving      (3 days)

Week 5  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 5: API (cont.)        (1 day)
        Phase 6: Docker             (2 days)
        Phase 7: CI/CD              (2 days)

Week 6  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 7: CI/CD (cont.)      (1 day)
        Phase 8: Deployment         (3 days)
        Phase 9: Monitoring         (1 day)

Week 7  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Phase 9: Monitoring (cont.) (4 days)
        Buffer / catch-up           (1 day)

Week 8  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        Buffer / polish / stretch   (5 days)
```

**Total: ~40 days (8 weeks at a comfortable pace)**

---

## Skills Checklist

When you complete this project, you will have hands-on experience with:

- [ ] EDA for time series (seasonality, trends, stationarity, autocorrelation)
- [ ] Time series decomposition (trend + seasonality + residual)
- [ ] Imputation strategies for time series (interpolation, seasonal filling)
- [ ] Feature engineering for time series (lags, rolling windows, cyclical encoding)
- [ ] Understanding data leakage in time series contexts
- [ ] Temporal train/val/test splitting (no random splits)
- [ ] Prophet for time series forecasting
- [ ] LSTM architecture for sequence-to-sequence prediction
- [ ] Walk-forward cross-validation
- [ ] Ensemble methods for combining forecasts
- [ ] Time series evaluation metrics (MAPE, MAE, directional accuracy)
- [ ] Forecast visualization with confidence intervals
- [ ] Error analysis by time components (hour, day, season, horizon)
- [ ] Building forecast APIs (multi-step output, confidence intervals)
- [ ] Interactive time series dashboards with Plotly + Streamlit
- [ ] Serving multiple models simultaneously (ensemble at inference time)
- [ ] Docker containerization for multi-model systems
- [ ] Containerizing Airflow for scheduled tasks
- [ ] Testing time series pipelines (temporal correctness, no leakage)
- [ ] CI/CD with GitHub Actions
- [ ] Cloud deployment for periodic-update systems
- [ ] Forecast accuracy monitoring (retrospective evaluation)
- [ ] Scheduled retraining with Airflow (DAGs, tasks, XCom)
- [ ] Model promotion gates (only promote if better)
- [ ] Temporal drift detection for time series
- [ ] Prometheus metrics for forecasting systems
- [ ] Grafana dashboards for forecast monitoring
- [ ] Alerting on model staleness and accuracy degradation

---

## Key Differences from Projects 1-3

| Aspect | Project 1 (CV) | Project 2 (NLP) | Project 3 (Tabular) | Project 4 (Time Series) |
|--------|----------------|-----------------|---------------------|------------------------|
| Data type | Images | Text | Tabular | Time series |
| Key challenge | Transfer learning | Tokenization | Feature engineering | Temporal dependencies |
| Validation | Random split | Random split | Random split | Walk-forward (temporal) |
| Models | 1 | 1 | 3+ compared | 2 + ensemble |
| Retraining | One-off | One-off | One-off | Scheduled (Airflow) |
| Monitoring | Prediction drift | Text drift | Feature drift | Forecast accuracy drift |
| Orchestration | None | None | None | Airflow |
| Unique skill | ONNX export | Hugging Face | SHAP | Walk-forward CV |
| Complexity | Beginner | Beginner+ | Intermediate | Intermediate |
| Timeline | 6 weeks | 6-7 weeks | 7 weeks | 8 weeks |

---

## Getting Started

Ready to begin? Start with Phase 1:

```bash
# 1. Create the project directory
mkdir energy-forecasting && cd energy-forecasting
git init

# 2. Create the folder structure
mkdir -p configs data/{raw,processed,splits} notebooks \
  src/{data,model,training,serving,monitoring,orchestration/dags,frontend} \
  tests/{unit,integration} docker .github/workflows \
  grafana/dashboards prometheus scripts

# 3. Verify key libraries are available
python -c "from prophet import Prophet; print('Prophet OK')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import airflow; print(f'Airflow {airflow.__version__}')"

# 4. Start writing DESIGN_DOC.md
```
