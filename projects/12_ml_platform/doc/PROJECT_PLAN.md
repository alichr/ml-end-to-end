# Project 12: Internal ML Platform (Expert+)

## Goal

Build a **mini internal ML platform** that ML engineers can use to register models,
manage features, deploy to production, and monitor performance --- a self-hosted
alternative combining the core capabilities of SageMaker, Vertex AI, MLflow, Feast,
and Seldon Core. The platform provides a unified API, CLI, and dashboard so that
any team can go from trained model to monitored production endpoint in minutes,
not weeks.

This is not a toy project. By the end, you will have a working Kubernetes-based
platform with a feature store, model registry, multi-model serving, canary deployments,
infrastructure as code, multi-tenant API gateway, and a monitoring dashboard.

---

## Why This Project?

| Question | Answer |
|----------|--------|
| Why build a platform? | ML platform engineering is one of the highest-demand roles in the industry. This project proves you can design and build the infrastructure that entire ML teams depend on. |
| Why not just use SageMaker? | Using a managed service is a click-and-deploy exercise. Building one teaches you what happens beneath the surface: serving infrastructure, feature pipelines, deployment strategies. |
| Is this realistic for one person? | Yes --- you are building a "mini" version. The scope is deliberately constrained to run on minikube, but the architecture mirrors production systems. |
| What makes this Expert+? | You are combining Kubernetes, Terraform, feature stores, model serving, multi-tenancy, and observability into a coherent platform. Each of these is a specialty; combining them is rare. |
| Will this help me get hired? | Platform engineering and MLOps roles pay top-of-market. This project is a direct portfolio piece for Staff MLE and ML Platform Engineer positions. |

---

## Architecture Overview

```
                    ┌──────────────────────────────────────────┐
                    │         React Dashboard                  │
                    │  (model catalog, deploy, monitor)        │
                    └─────────────────┬────────────────────────┘
                                      │ REST + WebSocket
                                      ▼
                    ┌──────────────────────────────────────────┐
                    │         API Gateway (FastAPI)             │
                    │  auth │ rate limit │ usage tracking       │
                    └─────────┬──────────┬──────────┬──────────┘
                              │          │          │
              ┌───────────────┘          │          └───────────────┐
              ▼                          ▼                          ▼
    ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
    │   Feature Store  │   │  Model Registry  │   │  Serving Layer   │
    │   (Feast)        │   │  (MLflow)        │   │  (Seldon/Bento)  │
    │                  │   │                  │   │                  │
    │  online: Redis   │   │  artifacts: S3/  │   │  canary │ A/B   │
    │  offline: PG/    │   │    local FS      │   │  autoscale      │
    │    Parquet       │   │  metadata: PG    │   │  multi-model    │
    └──────────────────┘   └──────────────────┘   └────────┬─────────┘
                                                           │
    ┌──────────────────────────────────────────────────────┘
    │          Kubernetes (minikube)
    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │  │Model A │ │Model B │ │Model C │ │Shadow  │
    │  │Pod     │ │Pod     │ │Pod     │ │Deploy  │
    │  └────────┘ └────────┘ └────────┘ └────────┘
    │
    │  Managed by: Helm Charts + Terraform
    └──────────────────────────────────────────────────────┐
                                                           │
    ┌──────────────────────────────────────────────────────┘
    │          Observability Stack
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐
    │  │Prometheus  │  │ Grafana    │  │ AlertManager│
    │  │(metrics)   │  │(dashboards)│  │(alerts)     │
    │  └────────────┘  └────────────┘  └────────────┘
    └──────────────────────────────────────────────────────
```

---

## Tech Stack

| Category | Tool | Why This One |
|----------|------|-------------|
| Language | Python 3.11+ | ML ecosystem standard |
| Container Orchestration | Kubernetes (minikube) | Industry standard for ML serving at scale |
| Feature Store | Feast 0.38+ | Open-source, supports online + offline serving |
| Model Registry | MLflow 2.17+ | Widely adopted, model versioning and lineage |
| Model Serving | Seldon Core / BentoML | Multi-model serving, canary, A/B testing |
| API Framework | FastAPI | High-performance async API with auto docs |
| Frontend | React 18 + Next.js 14 | Dashboard for model catalog and monitoring |
| Database | PostgreSQL 16 | Feature store offline, model metadata, platform state |
| Cache / Online Store | Redis 7 | Feature store online serving, session cache |
| Object Storage | MinIO (S3-compatible) | Model artifacts, training data |
| Infrastructure as Code | Terraform | Reproducible environment provisioning |
| Package Management | Helm 3 | Kubernetes application packaging |
| Monitoring | Prometheus + Grafana | Metrics, dashboards, alerting |
| CI/CD | GitHub Actions | Automated testing, model deployment |
| CLI | Click / Typer | Developer-facing CLI tool |
| SDK | Python package | Programmatic model registration and deployment |
| Testing | pytest + k8s test framework | Unit, integration, and Kubernetes tests |
| Code Quality | Ruff + mypy | Fast linting + type checking |

---

## Project Structure

```
ml-platform/
│
├── doc/
│   ├── DESIGN_DOC.md               # Platform design, API contracts, user personas
│   ├── PROJECT_PLAN.md              # This file
│   ├── GETTING_STARTED.md           # Onboarding guide for platform users
│   └── API_REFERENCE.md             # Auto-generated API docs
│
├── pyproject.toml                   # Platform dependencies
├── docker-compose.yaml              # Local development stack (no K8s)
│
├── sdk/
│   ├── pyproject.toml               # SDK package metadata
│   ├── mlplatform/
│   │   ├── __init__.py
│   │   ├── client.py               # Platform API client
│   │   ├── model.py                # Model registration helpers
│   │   ├── features.py             # Feature store helpers
│   │   └── deploy.py               # Deployment helpers
│   └── tests/
│       └── test_sdk.py
│
├── cli/
│   ├── pyproject.toml
│   ├── mlplatform_cli/
│   │   ├── __init__.py
│   │   ├── main.py                 # CLI entry point (Typer)
│   │   ├── commands/
│   │   │   ├── model.py            # mlp model register/list/deploy
│   │   │   ├── feature.py          # mlp feature create/list/serve
│   │   │   ├── deploy.py           # mlp deploy create/rollback/status
│   │   │   └── monitor.py          # mlp monitor metrics/alerts
│   │   └── config.py               # CLI configuration
│   └── tests/
│       └── test_cli.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── app.py                  # FastAPI application
│   │   ├── routes/
│   │   │   ├── models.py           # Model CRUD endpoints
│   │   │   ├── features.py         # Feature store endpoints
│   │   │   ├── deployments.py      # Deployment management
│   │   │   ├── predictions.py      # Unified prediction proxy
│   │   │   ├── monitoring.py       # Metrics and alerts
│   │   │   └── admin.py            # Tenant management, settings
│   │   ├── middleware/
│   │   │   ├── auth.py             # JWT / API key authentication
│   │   │   ├── rate_limit.py       # Per-tenant rate limiting
│   │   │   ├── usage.py            # Request counting and cost tracking
│   │   │   └── cors.py             # CORS configuration
│   │   ├── schemas.py              # Pydantic models
│   │   └── dependencies.py         # Dependency injection
│   │
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── service.py              # Model registry service layer
│   │   ├── mlflow_backend.py       # MLflow integration
│   │   ├── lineage.py              # Model lineage tracking
│   │   ├── approval.py             # Model approval workflow
│   │   └── models.py               # SQLAlchemy ORM models
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── service.py              # Feature store service layer
│   │   ├── feast_backend.py        # Feast integration
│   │   ├── definitions.py          # Feature definition helpers
│   │   └── validation.py           # Feature quality checks
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── service.py              # Serving orchestration
│   │   ├── seldon_backend.py       # Seldon Core integration
│   │   ├── bentoml_backend.py      # BentoML alternative
│   │   ├── canary.py               # Canary deployment logic
│   │   ├── ab_test.py              # A/B test traffic splitting
│   │   └── autoscale.py            # HPA configuration
│   │
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── pipeline.py             # CI/CD for model deployment
│   │   ├── testing.py              # Automated model testing
│   │   ├── rollback.py             # Rollback mechanisms
│   │   └── strategies.py           # Blue-green, canary, shadow
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Prometheus metric definitions
│   │   ├── alerts.py               # Alert rule definitions
│   │   ├── drift.py                # Data and concept drift detection
│   │   ├── cost.py                 # Cost per model tracking
│   │   └── quality.py              # Data quality monitoring
│   │
│   └── dashboard/
│       ├── package.json
│       ├── src/
│       │   ├── components/
│       │   │   ├── ModelCatalog.tsx
│       │   │   ├── DeploymentStatus.tsx
│       │   │   ├── FeatureExplorer.tsx
│       │   │   ├── MetricsDashboard.tsx
│       │   │   └── CostTracker.tsx
│       │   └── pages/
│       │       ├── index.tsx        # Platform overview
│       │       ├── models/
│       │       │   ├── index.tsx    # Model catalog
│       │       │   └── [id].tsx     # Model detail
│       │       ├── features/
│       │       │   └── index.tsx    # Feature store browser
│       │       ├── deployments/
│       │       │   └── index.tsx    # Active deployments
│       │       └── monitoring/
│       │           └── index.tsx    # Monitoring dashboard
│       └── next.config.js
│
├── infrastructure/
│   ├── terraform/
│   │   ├── main.tf                 # Provider config, modules
│   │   ├── variables.tf            # Input variables
│   │   ├── outputs.tf              # Output values
│   │   ├── environments/
│   │   │   ├── dev.tfvars
│   │   │   ├── staging.tfvars
│   │   │   └── prod.tfvars
│   │   └── modules/
│   │       ├── kubernetes/         # K8s cluster provisioning
│   │       ├── database/           # PostgreSQL setup
│   │       ├── storage/            # MinIO/S3 setup
│   │       └── monitoring/         # Prometheus/Grafana setup
│   │
│   ├── helm/
│   │   ├── ml-platform/
│   │   │   ├── Chart.yaml
│   │   │   ├── values.yaml
│   │   │   ├── values-dev.yaml
│   │   │   ├── values-staging.yaml
│   │   │   ├── values-prod.yaml
│   │   │   └── templates/
│   │   │       ├── deployment.yaml
│   │   │       ├── service.yaml
│   │   │       ├── ingress.yaml
│   │   │       ├── configmap.yaml
│   │   │       ├── secret.yaml
│   │   │       └── hpa.yaml
│   │   └── model-serving/
│   │       ├── Chart.yaml
│   │       ├── values.yaml
│   │       └── templates/
│   │           ├── seldon-deployment.yaml
│   │           └── istio-virtualservice.yaml
│   │
│   └── docker/
│       ├── Dockerfile.gateway
│       ├── Dockerfile.dashboard
│       ├── Dockerfile.worker
│       └── Dockerfile.model-server
│
├── tests/
│   ├── unit/
│   │   ├── test_registry_service.py
│   │   ├── test_feature_service.py
│   │   ├── test_serving_service.py
│   │   ├── test_canary.py
│   │   └── test_rate_limiter.py
│   ├── integration/
│   │   ├── test_model_lifecycle.py
│   │   ├── test_feature_serving.py
│   │   └── test_deployment_pipeline.py
│   ├── e2e/
│   │   ├── test_full_workflow.py    # Register -> deploy -> predict -> monitor
│   │   └── test_multi_tenant.py
│   └── conftest.py
│
├── examples/
│   ├── notebooks/
│   │   ├── 01_register_model.ipynb
│   │   ├── 02_define_features.ipynb
│   │   ├── 03_deploy_model.ipynb
│   │   └── 04_monitor_model.ipynb
│   └── scripts/
│       ├── train_and_register.py
│       └── batch_predict.py
│
├── scripts/
│   ├── setup_minikube.sh
│   ├── install_deps.sh
│   ├── seed_demo_data.py
│   └── run_local.sh
│
└── .github/
    └── workflows/
        ├── ci.yaml
        ├── cd-staging.yaml
        └── cd-prod.yaml
```

---

## Phase 1: Setup & Design Doc

**Duration:** 2--3 days
**Objective:** Define what the platform does, who uses it, and how the components
interact before writing any infrastructure code.

### Task 1.1: Initialize Repository and Tooling

Set up the monorepo structure with separate packages for the platform core, SDK, and CLI.

```bash
mkdir ml-platform && cd ml-platform
git init
python -m venv .venv && source .venv/bin/activate
```

**pyproject.toml core dependencies:**

```toml
[project]
name = "ml-platform"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic>=2.9.0",
    "sqlalchemy>=2.0.35",
    "asyncpg>=0.30.0",
    "redis>=5.2.0",
    "mlflow>=2.17.0",
    "feast>=0.38.0",
    "httpx>=0.27.0",
    "kubernetes>=31.0.0",
    "prometheus-client>=0.21.0",
    "structlog>=24.4.0",
    "pyyaml>=6.0.2",
    "boto3>=1.35.0",         # S3/MinIO
    "click>=8.1.0",
    "typer>=0.12.0",
    "python-jose>=3.3.0",   # JWT auth
    "passlib>=1.7.4",       # password hashing
]
```

### Task 1.2: Write the Platform Design Document

The design doc must answer:

1. **User Personas** --- Who uses this platform? ML engineers, data scientists, ML ops?
2. **Platform Capabilities** --- What can users do? Register models, deploy, monitor.
3. **API Design** --- RESTful endpoints for all CRUD operations.
4. **Multi-Tenancy Model** --- How are teams isolated? API keys, namespaces, quotas.
5. **Deployment Model** --- How do models get from registry to serving endpoint?
6. **Failure Modes** --- What happens when a model deployment fails? Auto-rollback?

### Task 1.3: Define User Personas

| Persona | Role | Needs |
|---------|------|-------|
| **ML Engineer** | Trains and registers models | SDK/CLI for model registration, experiment tracking |
| **Data Scientist** | Defines features, explores models | Feature store access, model catalog, notebooks |
| **ML Ops** | Deploys and monitors models | Deployment pipeline, monitoring dashboard, alerts |
| **Platform Admin** | Manages tenants and quotas | Tenant CRUD, resource allocation, cost tracking |

### Task 1.4: Design the Platform API

Define the core API contracts that all components will implement.

```
# Model Registry API
POST   /api/v1/models                    # Register a new model
GET    /api/v1/models                    # List models (with filters)
GET    /api/v1/models/{id}               # Get model details + lineage
POST   /api/v1/models/{id}/versions      # Register a new version
PATCH  /api/v1/models/{id}/versions/{v}  # Update stage (staging/prod)
POST   /api/v1/models/{id}/approve       # Approve for production

# Feature Store API
POST   /api/v1/features                  # Define a new feature
GET    /api/v1/features                  # List features
GET    /api/v1/features/{name}/online    # Get online feature values
POST   /api/v1/features/batch            # Get offline feature batch

# Deployment API
POST   /api/v1/deployments               # Create deployment
GET    /api/v1/deployments               # List deployments
GET    /api/v1/deployments/{id}          # Deployment status
POST   /api/v1/deployments/{id}/canary   # Start canary rollout
POST   /api/v1/deployments/{id}/rollback # Rollback to previous version
DELETE /api/v1/deployments/{id}          # Tear down deployment

# Prediction API (unified proxy)
POST   /api/v1/predict/{model_name}      # Route to correct serving endpoint

# Monitoring API
GET    /api/v1/monitoring/{model_name}/metrics   # Model performance metrics
GET    /api/v1/monitoring/{model_name}/drift      # Drift detection results
GET    /api/v1/monitoring/costs                   # Cost per model/team
```

**Deliverables:**
- [ ] Repository structure with monorepo layout
- [ ] DESIGN_DOC.md with all sections above
- [ ] API contract definitions (OpenAPI spec or markdown)
- [ ] User persona definitions with user stories
- [ ] Architecture diagram

---

## Phase 2: Feature Store

**Duration:** 4--5 days
**Objective:** Set up Feast as the feature store with online (Redis) and offline
(PostgreSQL/Parquet) serving, feature definitions, and a sharing mechanism so multiple
teams can discover and reuse features.

### Task 2.1: Feast Project Setup

Initialize a Feast feature repository with online and offline stores configured.

```python
# feature_repo/feature_store.yaml
project: ml_platform
provider: local
registry: data/registry.db
online_store:
  type: redis
  connection_string: "redis://localhost:6379"
offline_store:
  type: file     # Parquet files for local dev; swap to BigQuery/Redshift in prod
entity_key_serialization_version: 2
```

### Task 2.2: Define Feature Entities and Views

Create example feature definitions that platform users will register through the API.

```python
# feature_repo/features/user_features.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String


user = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="Unique user identifier",
)

user_profile_source = FileSource(
    path="data/user_profiles.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

user_profile_features = FeatureView(
    name="user_profile",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="signup_days_ago", dtype=Int64),
        Field(name="account_type", dtype=String),
        Field(name="lifetime_value", dtype=Float32),
        Field(name="avg_session_duration_sec", dtype=Float32),
    ],
    online=True,
    source=user_profile_source,
    tags={"team": "growth", "owner": "data-science"},
)
```

### Task 2.3: Feature Store Service Layer

Build the platform service that wraps Feast and adds team-based access control, feature
discovery, and usage tracking.

```python
# src/features/service.py
from typing import Any
from datetime import datetime
import structlog

from feast import FeatureStore

logger = structlog.get_logger()


class FeatureService:
    """Platform-level wrapper around Feast for multi-tenant feature management."""

    def __init__(self, feast_repo_path: str):
        self.store = FeatureStore(repo_path=feast_repo_path)

    def list_features(
        self, team: str | None = None, tags: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """List all feature views, optionally filtered by team or tags."""
        feature_views = self.store.list_feature_views()
        results = []

        for fv in feature_views:
            fv_tags = fv.tags or {}

            if team and fv_tags.get("team") != team:
                continue

            if tags and not all(fv_tags.get(k) == v for k, v in tags.items()):
                continue

            results.append({
                "name": fv.name,
                "entities": [e.name for e in fv.entities],
                "features": [f.name for f in fv.schema],
                "ttl": str(fv.ttl),
                "online": fv.online,
                "tags": fv_tags,
            })

        return results

    async def get_online_features(
        self,
        feature_refs: list[str],
        entity_rows: list[dict[str, Any]],
        team: str,
    ) -> dict[str, Any]:
        """Fetch online features for real-time inference."""
        logger.info(
            "features.online_fetch",
            features=feature_refs,
            entity_count=len(entity_rows),
            team=team,
        )

        result = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        )

        return result.to_dict()

    async def get_historical_features(
        self,
        feature_refs: list[str],
        entity_df: Any,  # pandas DataFrame
        team: str,
    ) -> Any:
        """Fetch historical features for training data generation."""
        logger.info(
            "features.historical_fetch",
            features=feature_refs,
            rows=len(entity_df),
            team=team,
        )

        return self.store.get_historical_features(
            features=feature_refs,
            entity_df=entity_df,
        ).to_df()

    def materialize(
        self,
        start_date: datetime,
        end_date: datetime,
        feature_views: list[str] | None = None,
    ) -> None:
        """Materialize features from offline to online store."""
        self.store.materialize(
            start_date=start_date,
            end_date=end_date,
            feature_views=feature_views,
        )
```

### Task 2.4: Feature Validation

Implement quality checks that run when features are registered or materialized:
schema validation, null rate checks, value range validation, and freshness monitoring.

### Task 2.5: Feature Discovery API

Expose feature metadata through the API so teams can discover and reuse features
created by other teams.

**Deliverables:**
- [ ] Feast project with online (Redis) and offline (Parquet) stores
- [ ] Example feature definitions for user and product entities
- [ ] `FeatureService` with online/offline fetching and materialization
- [ ] Feature quality validation (null rates, ranges, freshness)
- [ ] Feature discovery API endpoints
- [ ] Integration test: materialize features, fetch online and offline

---

## Phase 3: Model Registry

**Duration:** 4--5 days
**Objective:** Build an MLflow-backed model registry with versioning, metadata tracking,
lineage, and an approval workflow for promoting models to production.

### Task 3.1: MLflow Registry Setup

Configure MLflow with PostgreSQL backend for metadata and MinIO for artifact storage.

```python
# src/registry/mlflow_backend.py
import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(
    tracking_uri: str = "postgresql://platform:pass@localhost/mlflow",
    artifact_root: str = "s3://ml-platform-artifacts/",
) -> MlflowClient:
    """Configure MLflow with PostgreSQL tracking and S3 artifacts."""
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)
```

### Task 3.2: Model Registry Service

Wrap MLflow in a platform service that adds multi-tenant isolation, approval workflows,
and enhanced metadata.

```python
# src/registry/service.py
from typing import Any
from datetime import datetime
from enum import Enum
import structlog
from mlflow.tracking import MlflowClient

logger = structlog.get_logger()


class ModelStage(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ModelRegistryService:
    """Platform model registry built on MLflow with approval workflows."""

    def __init__(self, mlflow_client: MlflowClient, db_session: Any):
        self.mlflow = mlflow_client
        self.db = db_session

    async def register_model(
        self,
        name: str,
        description: str,
        team: str,
        owner: str,
        model_uri: str,
        tags: dict[str, str] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Register a new model version in the registry."""
        logger.info(
            "registry.register",
            model=name,
            team=team,
            owner=owner,
        )

        # Register in MLflow
        result = self.mlflow.create_model_version(
            name=name,
            source=model_uri,
            description=description,
            tags={
                **(tags or {}),
                "team": team,
                "owner": owner,
                "registered_at": datetime.utcnow().isoformat(),
            },
        )

        # Store extended metadata in platform DB
        await self._store_metadata(
            model_name=name,
            version=result.version,
            team=team,
            owner=owner,
            metrics=metrics or {},
        )

        return {
            "model_name": name,
            "version": result.version,
            "stage": ModelStage.DEVELOPMENT,
            "approval_status": ApprovalStatus.PENDING,
        }

    async def approve_model(
        self,
        model_name: str,
        version: str,
        approver: str,
        target_stage: ModelStage,
        notes: str = "",
    ) -> dict[str, Any]:
        """Approve a model version for promotion to a target stage."""
        logger.info(
            "registry.approve",
            model=model_name,
            version=version,
            approver=approver,
            target=target_stage,
        )

        # Validate: model must pass automated checks before approval
        checks = await self._run_approval_checks(model_name, version)
        if not all(c["passed"] for c in checks):
            failed = [c["name"] for c in checks if not c["passed"]]
            return {
                "approved": False,
                "reason": f"Failed checks: {failed}",
                "checks": checks,
            }

        # Promote in MLflow
        self.mlflow.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=target_stage.value,
        )

        # Record approval in platform DB
        await self._record_approval(
            model_name=model_name,
            version=version,
            approver=approver,
            stage=target_stage,
            notes=notes,
        )

        return {
            "approved": True,
            "model_name": model_name,
            "version": version,
            "stage": target_stage,
            "approver": approver,
        }

    async def _run_approval_checks(
        self, model_name: str, version: str
    ) -> list[dict[str, Any]]:
        """Run automated checks before allowing model promotion."""
        checks = []

        # Check 1: Model artifact exists and is loadable
        checks.append(await self._check_artifact_integrity(model_name, version))

        # Check 2: Minimum metric thresholds met
        checks.append(await self._check_metric_thresholds(model_name, version))

        # Check 3: No data quality issues in training data
        checks.append(await self._check_training_data_quality(model_name, version))

        # Check 4: Model size within limits
        checks.append(await self._check_model_size(model_name, version))

        return checks

    async def get_lineage(
        self, model_name: str, version: str
    ) -> dict[str, Any]:
        """Get the full lineage of a model version."""
        # Retrieve: training data version, feature definitions used,
        # parent model (if fine-tuned), training config, environment
        ...
```

### Task 3.3: Model Lineage Tracking

Track the full provenance of every model: which data it was trained on, which features
it uses, which experiments led to it, and which other models it descends from.

```python
# src/registry/lineage.py
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelLineage:
    """Complete provenance record for a model version."""

    model_name: str
    version: str

    # Data lineage
    training_data_version: str         # DVC hash or dataset version
    training_data_size: int            # number of rows
    feature_definitions: list[str]     # Feast feature view names used

    # Training lineage
    experiment_id: str                 # MLflow experiment
    run_id: str                        # MLflow run
    training_config: dict              # hyperparameters, augmentation
    training_duration_seconds: float
    training_environment: dict         # Python version, GPU type, libraries

    # Model lineage
    parent_model: str | None = None    # if fine-tuned from another model
    framework: str = ""                # pytorch, tensorflow, sklearn
    model_type: str = ""               # classification, regression, etc.

    # Evaluation lineage
    eval_metrics: dict[str, float] = field(default_factory=dict)
    eval_dataset_version: str = ""
    eval_slices: dict[str, dict[str, float]] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)
```

### Task 3.4: Model Approval Workflow

Implement a multi-stage approval process:
1. **Automated checks** --- artifact integrity, metric thresholds, data quality
2. **Peer review** --- Another ML engineer reviews the model card
3. **ML Ops approval** --- Ops team confirms deployment readiness

**Deliverables:**
- [ ] MLflow configured with PostgreSQL + MinIO
- [ ] `ModelRegistryService` with register, approve, list, get, lineage
- [ ] Model lineage tracking with full provenance
- [ ] Multi-stage approval workflow with automated checks
- [ ] API endpoints for model CRUD and approval
- [ ] Integration test: register model, run checks, approve, promote

---

## Phase 4: Model Serving

**Duration:** 5--6 days
**Objective:** Build multi-model serving infrastructure with canary deployments,
A/B testing, auto-scaling, and a unified prediction endpoint.

### Task 4.1: Serving Backend Abstraction

Create an abstraction over Seldon Core and BentoML so the platform can use either.

```python
# src/serving/service.py
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum
import structlog

logger = structlog.get_logger()


class DeploymentStrategy(str, Enum):
    RECREATE = "recreate"        # Replace all at once
    CANARY = "canary"            # Gradual traffic shift
    BLUE_GREEN = "blue_green"    # Switch traffic instantly
    SHADOW = "shadow"            # Mirror traffic to new version


class ServingBackend(ABC):
    """Abstract interface for model serving backends."""

    @abstractmethod
    async def deploy(
        self,
        model_name: str,
        model_version: str,
        model_uri: str,
        resources: dict[str, Any],
        replicas: int = 1,
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def get_status(self, deployment_id: str) -> dict[str, Any]: ...

    @abstractmethod
    async def predict(
        self, deployment_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...

    @abstractmethod
    async def delete(self, deployment_id: str) -> None: ...

    @abstractmethod
    async def scale(self, deployment_id: str, replicas: int) -> None: ...


class ServingService:
    """Orchestrates model deployments across the serving backend."""

    def __init__(self, backend: ServingBackend, registry: Any, db: Any):
        self.backend = backend
        self.registry = registry
        self.db = db

    async def create_deployment(
        self,
        model_name: str,
        model_version: str,
        team: str,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        resources: dict[str, Any] | None = None,
        replicas: int = 2,
    ) -> dict[str, Any]:
        """Deploy a model version with the specified strategy."""
        logger.info(
            "serving.deploy",
            model=model_name,
            version=model_version,
            strategy=strategy,
            team=team,
        )

        # Verify model is approved for production
        model_info = await self.registry.get_model(model_name, model_version)
        if model_info.get("stage") != "production":
            raise ValueError(
                f"Model {model_name} v{model_version} is not approved for production. "
                f"Current stage: {model_info.get('stage')}"
            )

        # Default resource allocation
        resources = resources or {
            "cpu": "500m",
            "memory": "1Gi",
            "gpu": "0",
        }

        # Deploy via backend
        deployment = await self.backend.deploy(
            model_name=model_name,
            model_version=model_version,
            model_uri=model_info["artifact_uri"],
            resources=resources,
            replicas=replicas,
        )

        # If canary, set up gradual traffic shifting
        if strategy == DeploymentStrategy.CANARY:
            await self._setup_canary(deployment["id"], model_name)

        return deployment
```

### Task 4.2: Canary Deployment Implementation

Gradually shift traffic from the old model version to the new one, monitoring metrics
at each step. Automatically roll back if error rate or latency spikes.

```python
# src/serving/canary.py
import asyncio
from typing import Any
import structlog

logger = structlog.get_logger()


class CanaryController:
    """Manages gradual traffic shifting for canary deployments."""

    def __init__(
        self,
        traffic_steps: list[int] = None,
        step_duration_seconds: int = 300,
        rollback_thresholds: dict[str, float] = None,
    ):
        self.traffic_steps = traffic_steps or [5, 10, 25, 50, 75, 100]
        self.step_duration = step_duration_seconds
        self.rollback_thresholds = rollback_thresholds or {
            "error_rate": 0.05,        # 5% error rate
            "p99_latency_ms": 5000,    # 5 second p99
            "prediction_drift": 0.15,  # 15% distribution drift
        }

    async def execute_canary(
        self,
        deployment_id: str,
        old_version: str,
        new_version: str,
        traffic_manager: Any,
        metrics_collector: Any,
    ) -> dict[str, Any]:
        """Execute a canary rollout with automatic rollback."""
        for step_pct in self.traffic_steps:
            logger.info(
                "canary.step",
                deployment=deployment_id,
                traffic_pct=step_pct,
                new_version=new_version,
            )

            # Shift traffic
            await traffic_manager.set_weights(
                deployment_id,
                {old_version: 100 - step_pct, new_version: step_pct},
            )

            # Wait and collect metrics
            await asyncio.sleep(self.step_duration)
            metrics = await metrics_collector.get_metrics(
                deployment_id, new_version, window_seconds=self.step_duration
            )

            # Check rollback thresholds
            should_rollback, reason = self._check_thresholds(metrics)
            if should_rollback:
                logger.warning(
                    "canary.rollback",
                    deployment=deployment_id,
                    reason=reason,
                    step_pct=step_pct,
                    metrics=metrics,
                )
                await traffic_manager.set_weights(
                    deployment_id, {old_version: 100, new_version: 0}
                )
                return {
                    "status": "rolled_back",
                    "reason": reason,
                    "failed_at_step": step_pct,
                    "metrics_at_failure": metrics,
                }

        # Canary complete --- new version gets 100% traffic
        logger.info("canary.complete", deployment=deployment_id)
        return {
            "status": "completed",
            "new_version": new_version,
            "steps_completed": len(self.traffic_steps),
        }

    def _check_thresholds(
        self, metrics: dict[str, float]
    ) -> tuple[bool, str]:
        """Check if any metric exceeds rollback thresholds."""
        for metric_name, threshold in self.rollback_thresholds.items():
            actual = metrics.get(metric_name, 0)
            if actual > threshold:
                return True, (
                    f"{metric_name}={actual:.4f} exceeds threshold={threshold}"
                )
        return False, ""
```

### Task 4.3: A/B Testing Support

Allow users to split traffic between model versions and collect comparison metrics.

```python
# src/serving/ab_test.py
from dataclasses import dataclass
from typing import Any
import hashlib


@dataclass
class ABTestConfig:
    name: str
    model_a: str    # model_name:version
    model_b: str    # model_name:version
    traffic_split: float = 0.5  # fraction going to model_b
    metric_to_compare: str = "accuracy"
    min_sample_size: int = 1000


class ABTestRouter:
    """Routes prediction requests to A or B based on consistent hashing."""

    def __init__(self, config: ABTestConfig):
        self.config = config

    def route(self, request_id: str) -> str:
        """Deterministically route a request to model A or B."""
        hash_val = int(hashlib.sha256(request_id.encode()).hexdigest(), 16)
        fraction = (hash_val % 10000) / 10000.0

        if fraction < self.config.traffic_split:
            return self.config.model_b
        return self.config.model_a

    async def get_results(
        self, metrics_store: Any
    ) -> dict[str, Any]:
        """Compute A/B test statistical results."""
        metrics_a = await metrics_store.get(self.config.model_a)
        metrics_b = await metrics_store.get(self.config.model_b)

        return {
            "model_a": {
                "model": self.config.model_a,
                "samples": metrics_a["count"],
                "metric_value": metrics_a[self.config.metric_to_compare],
            },
            "model_b": {
                "model": self.config.model_b,
                "samples": metrics_b["count"],
                "metric_value": metrics_b[self.config.metric_to_compare],
            },
            "sufficient_data": min(
                metrics_a["count"], metrics_b["count"]
            ) >= self.config.min_sample_size,
        }
```

### Task 4.4: Auto-Scaling Configuration

Set up Horizontal Pod Autoscaler (HPA) for model serving pods based on CPU, memory,
and custom metrics (requests per second, queue depth).

### Task 4.5: Model Warm-Up

Implement model warm-up that sends synthetic requests to newly deployed models before
they receive real traffic, preventing cold-start latency spikes.

**Deliverables:**
- [ ] `ServingBackend` abstraction with Seldon Core implementation
- [ ] `CanaryController` with automatic rollback on metric degradation
- [ ] `ABTestRouter` with consistent hashing and statistical results
- [ ] Auto-scaling configuration (HPA with custom metrics)
- [ ] Model warm-up on deployment
- [ ] Integration test: deploy model, run canary, verify traffic shifting

---

## Phase 5: Deployment Pipeline

**Duration:** 3--4 days
**Objective:** Build the CI/CD pipeline for models --- from registry to production endpoint
with automated testing, canary rollout, and rollback capability.

### Task 5.1: Model Testing Framework

Automated tests that run before every model deployment:

```python
# src/deployment/testing.py
from typing import Any
import structlog

logger = structlog.get_logger()


class ModelTestSuite:
    """Automated tests for model deployment readiness."""

    async def run_all(
        self, model_name: str, model_version: str, model_uri: str
    ) -> list[dict[str, Any]]:
        """Run all deployment readiness tests."""
        results = []

        results.append(await self.test_model_loadable(model_uri))
        results.append(await self.test_inference_latency(model_uri))
        results.append(await self.test_output_schema(model_uri))
        results.append(await self.test_edge_cases(model_uri))
        results.append(await self.test_resource_usage(model_uri))

        passed = all(r["passed"] for r in results)
        logger.info(
            "deployment.tests",
            model=model_name,
            version=model_version,
            passed=passed,
            results=results,
        )
        return results

    async def test_model_loadable(self, model_uri: str) -> dict[str, Any]:
        """Verify the model artifact can be loaded without errors."""
        try:
            import mlflow
            model = mlflow.pyfunc.load_model(model_uri)
            return {"name": "model_loadable", "passed": True}
        except Exception as e:
            return {"name": "model_loadable", "passed": False, "error": str(e)}

    async def test_inference_latency(
        self, model_uri: str, max_p99_ms: float = 500
    ) -> dict[str, Any]:
        """Verify inference latency is within acceptable bounds."""
        import time
        import mlflow

        model = mlflow.pyfunc.load_model(model_uri)
        sample_input = self._generate_sample_input(model)

        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            model.predict(sample_input)
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        p99 = latencies[98]

        return {
            "name": "inference_latency",
            "passed": p99 <= max_p99_ms,
            "p50_ms": latencies[49],
            "p99_ms": p99,
            "threshold_ms": max_p99_ms,
        }

    async def test_output_schema(self, model_uri: str) -> dict[str, Any]:
        """Verify model output matches expected schema."""
        ...

    async def test_edge_cases(self, model_uri: str) -> dict[str, Any]:
        """Test with edge case inputs: nulls, out-of-range, adversarial."""
        ...

    async def test_resource_usage(self, model_uri: str) -> dict[str, Any]:
        """Verify model memory and CPU usage are within limits."""
        ...
```

### Task 5.2: Deployment Pipeline Orchestration

Chain the steps: test -> build container -> deploy to staging -> run integration tests
-> canary to production.

```python
# src/deployment/pipeline.py
from enum import Enum
from typing import Any
import structlog

logger = structlog.get_logger()


class PipelineStage(str, Enum):
    TESTING = "testing"
    BUILDING = "building"
    STAGING = "staging"
    INTEGRATION = "integration_testing"
    CANARY = "canary"
    PRODUCTION = "production"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentPipeline:
    """End-to-end model deployment pipeline."""

    def __init__(self, test_suite, serving_service, canary_controller, db):
        self.test_suite = test_suite
        self.serving = serving_service
        self.canary = canary_controller
        self.db = db

    async def execute(
        self,
        model_name: str,
        model_version: str,
        model_uri: str,
        team: str,
    ) -> dict[str, Any]:
        """Run the full deployment pipeline."""
        pipeline_id = f"deploy-{model_name}-{model_version}"

        # Stage 1: Automated testing
        await self._update_stage(pipeline_id, PipelineStage.TESTING)
        test_results = await self.test_suite.run_all(
            model_name, model_version, model_uri
        )
        if not all(r["passed"] for r in test_results):
            await self._update_stage(pipeline_id, PipelineStage.FAILED)
            return {"status": "failed", "stage": "testing", "results": test_results}

        # Stage 2: Deploy to staging
        await self._update_stage(pipeline_id, PipelineStage.STAGING)
        staging_deploy = await self.serving.create_deployment(
            model_name=model_name,
            model_version=model_version,
            team=team,
            environment="staging",
        )

        # Stage 3: Integration tests against staging
        await self._update_stage(pipeline_id, PipelineStage.INTEGRATION)
        integration_ok = await self._run_integration_tests(staging_deploy["endpoint"])
        if not integration_ok:
            await self.serving.delete_deployment(staging_deploy["id"])
            await self._update_stage(pipeline_id, PipelineStage.FAILED)
            return {"status": "failed", "stage": "integration_testing"}

        # Stage 4: Canary deployment to production
        await self._update_stage(pipeline_id, PipelineStage.CANARY)
        canary_result = await self.canary.execute_canary(
            deployment_id=staging_deploy["id"],
            old_version=await self._get_current_prod_version(model_name),
            new_version=model_version,
            traffic_manager=self.serving.traffic_manager,
            metrics_collector=self.serving.metrics_collector,
        )

        if canary_result["status"] == "rolled_back":
            await self._update_stage(pipeline_id, PipelineStage.ROLLED_BACK)
            return {"status": "rolled_back", "details": canary_result}

        await self._update_stage(pipeline_id, PipelineStage.COMPLETE)
        return {"status": "deployed", "deployment": staging_deploy}
```

### Task 5.3: Rollback Mechanism

One-command rollback that reverts to the previous model version, including traffic
routing and health verification.

### Task 5.4: GitHub Actions CD Pipeline

```yaml
# .github/workflows/cd-staging.yaml
name: Deploy Model to Staging
on:
  workflow_dispatch:
    inputs:
      model_name:
        description: "Model name in registry"
        required: true
      model_version:
        description: "Model version to deploy"
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run model tests
        run: |
          pip install -e ".[dev]"
          python -m pytest tests/deployment/ \
            --model-name=${{ inputs.model_name }} \
            --model-version=${{ inputs.model_version }}

      - name: Deploy to staging
        run: |
          mlp deploy create \
            --model ${{ inputs.model_name }} \
            --version ${{ inputs.model_version }} \
            --environment staging \
            --strategy canary

      - name: Run integration tests
        run: |
          python -m pytest tests/integration/ \
            --endpoint=https://staging.ml-platform.internal
```

**Deliverables:**
- [ ] `ModelTestSuite` with 5 automated checks
- [ ] `DeploymentPipeline` orchestrating test -> stage -> canary -> prod
- [ ] One-command rollback mechanism
- [ ] GitHub Actions CD pipeline for staging and production
- [ ] End-to-end test: model goes from registry to live production endpoint

---

## Phase 6: Kubernetes Infrastructure

**Duration:** 4--5 days
**Objective:** Set up the Kubernetes cluster with minikube, create Helm charts for all
platform components, and configure resource management and auto-scaling.

### Task 6.1: Minikube Setup

Create a repeatable setup script that initializes minikube with the required addons.

```bash
#!/bin/bash
# scripts/setup_minikube.sh

set -euo pipefail

echo "==> Starting minikube with required resources..."
minikube start \
    --cpus=4 \
    --memory=8192 \
    --disk-size=50g \
    --kubernetes-version=v1.29.0 \
    --driver=docker

echo "==> Enabling required addons..."
minikube addons enable metrics-server
minikube addons enable ingress
minikube addons enable storage-provisioner

echo "==> Creating platform namespaces..."
kubectl create namespace ml-platform || true
kubectl create namespace ml-serving || true
kubectl create namespace ml-monitoring || true

echo "==> Installing Helm repos..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

echo "==> Minikube setup complete!"
kubectl cluster-info
```

### Task 6.2: Helm Chart for ML Platform

Create a Helm chart that packages the API gateway, worker processes, and configuration.

```yaml
# infrastructure/helm/ml-platform/Chart.yaml
apiVersion: v2
name: ml-platform
description: Internal ML Platform - API Gateway and Workers
type: application
version: 0.1.0
appVersion: "1.0.0"
dependencies:
  - name: postgresql
    version: "15.x.x"
    repository: https://charts.bitnami.com/bitnami
  - name: redis
    version: "19.x.x"
    repository: https://charts.bitnami.com/bitnami
```

```yaml
# infrastructure/helm/ml-platform/values.yaml
replicaCount: 2

image:
  repository: ml-platform/gateway
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: ml-platform.local
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: "1"
    memory: 2Gi
  requests:
    cpu: 250m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilization: 70
  targetMemoryUtilization: 80

postgresql:
  auth:
    database: ml_platform
    username: platform
    password: platform_pass
  primary:
    persistence:
      size: 10Gi

redis:
  auth:
    enabled: false
  master:
    persistence:
      size: 2Gi
```

### Task 6.3: Kubernetes Deployment Templates

```yaml
# infrastructure/helm/ml-platform/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ml-platform.fullname" . }}
  labels:
    {{- include "ml-platform.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "ml-platform.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "ml-platform.selectorLabels" . | nindent 8 }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          ports:
            - containerPort: {{ .Values.service.port }}
          livenessProbe:
            httpGet:
              path: /health
              port: {{ .Values.service.port }}
            initialDelaySeconds: 15
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: {{ .Values.service.port }}
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          envFrom:
            - configMapRef:
                name: {{ include "ml-platform.fullname" . }}-config
            - secretRef:
                name: {{ include "ml-platform.fullname" . }}-secrets
```

### Task 6.4: Horizontal Pod Autoscaler

```yaml
# infrastructure/helm/ml-platform/templates/hpa.yaml
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "ml-platform.fullname" . }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "ml-platform.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilization }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilization }}
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
{{- end }}
```

### Task 6.5: Resource Quotas and Limit Ranges

Set up per-namespace resource quotas so teams cannot consume more than their allocation.

**Deliverables:**
- [ ] Minikube setup script with required addons
- [ ] Helm chart for ML platform (gateway, workers, DB, cache)
- [ ] Helm chart for model serving (Seldon/BentoML deployments)
- [ ] HPA configuration with custom metrics
- [ ] Resource quotas per namespace/team
- [ ] Verified: `helm install` deploys the full platform on minikube

---

## Phase 7: Infrastructure as Code

**Duration:** 3--4 days
**Objective:** Use Terraform to provision and manage all infrastructure components
across dev, staging, and production environments.

### Task 7.1: Terraform Project Structure

```hcl
# infrastructure/terraform/main.tf
terraform {
  required_version = ">= 1.7.0"

  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.16"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  backend "local" {
    path = "terraform.tfstate"
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
  config_context = var.kube_context
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
    config_context = var.kube_context
  }
}

module "database" {
  source = "./modules/database"

  namespace    = var.namespace
  environment  = var.environment
  storage_size = var.db_storage_size
}

module "storage" {
  source = "./modules/storage"

  namespace   = var.namespace
  environment = var.environment
  bucket_name = "ml-platform-${var.environment}"
}

module "monitoring" {
  source = "./modules/monitoring"

  namespace   = "ml-monitoring"
  environment = var.environment

  grafana_admin_password = var.grafana_admin_password
  alert_email            = var.alert_email
}

module "platform" {
  source = "./modules/kubernetes"

  namespace   = var.namespace
  environment = var.environment

  image_tag       = var.platform_image_tag
  replica_count   = var.platform_replicas
  db_url          = module.database.connection_url
  redis_url       = module.database.redis_url
  storage_url     = module.storage.endpoint_url
}
```

### Task 7.2: Environment-Specific Variables

```hcl
# infrastructure/terraform/variables.tf
variable "environment" {
  type        = string
  description = "Deployment environment (dev, staging, prod)"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "namespace" {
  type    = string
  default = "ml-platform"
}

variable "kube_context" {
  type    = string
  default = "minikube"
}

variable "platform_replicas" {
  type    = number
  default = 2
}

variable "db_storage_size" {
  type    = string
  default = "10Gi"
}

variable "platform_image_tag" {
  type    = string
  default = "latest"
}

variable "grafana_admin_password" {
  type      = string
  sensitive = true
}

variable "alert_email" {
  type    = string
  default = ""
}
```

```hcl
# infrastructure/terraform/environments/dev.tfvars
environment         = "dev"
kube_context        = "minikube"
platform_replicas   = 1
db_storage_size     = "5Gi"
platform_image_tag  = "dev-latest"
```

```hcl
# infrastructure/terraform/environments/prod.tfvars
environment         = "prod"
kube_context        = "prod-cluster"
platform_replicas   = 3
db_storage_size     = "50Gi"
platform_image_tag  = "v1.0.0"
```

### Task 7.3: Terraform Modules

Create reusable modules for database, storage, monitoring, and the platform itself.
Each module encapsulates the Kubernetes resources it manages.

### Task 7.4: State Management and Locking

Configure remote state storage (S3/MinIO) with locking to prevent concurrent
modifications in team environments.

**Deliverables:**
- [ ] Terraform project with modular structure
- [ ] Environment-specific tfvars for dev, staging, prod
- [ ] Modules for database, storage, monitoring, platform
- [ ] `terraform plan` produces a clean diff for each environment
- [ ] Verified: `terraform apply` provisions the full stack on minikube

---

## Phase 8: Multi-Tenant API Gateway

**Duration:** 4--5 days
**Objective:** Build the API gateway with JWT authentication, per-tenant rate limiting,
usage tracking, and cost allocation.

### Task 8.1: Authentication Middleware

```python
# src/gateway/middleware/auth.py
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime
from typing import Any
import structlog

logger = structlog.get_logger()
security = HTTPBearer()


class AuthMiddleware:
    """JWT and API key authentication for multi-tenant access."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm

    async def verify_token(
        self, credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> dict[str, Any]:
        """Verify JWT token and return tenant context."""
        try:
            payload = jwt.decode(
                credentials.credentials,
                self.secret_key,
                algorithms=[self.algorithm],
            )

            if payload.get("exp", 0) < datetime.utcnow().timestamp():
                raise HTTPException(status_code=401, detail="Token expired")

            return {
                "tenant_id": payload["tenant_id"],
                "team": payload["team"],
                "role": payload.get("role", "user"),
                "permissions": payload.get("permissions", []),
            }

        except JWTError as e:
            logger.warning("auth.invalid_token", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid token")

    async def verify_api_key(self, request: Request) -> dict[str, Any]:
        """Alternative: verify API key from header."""
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            raise HTTPException(status_code=401, detail="API key required")

        # Look up API key in database
        tenant = await self._lookup_api_key(api_key)
        if not tenant:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return tenant
```

### Task 8.2: Per-Tenant Rate Limiting

```python
# src/gateway/middleware/rate_limit.py
import time
from typing import Any
import redis.asyncio as redis
from fastapi import Request, HTTPException
import structlog

logger = structlog.get_logger()


class RateLimiter:
    """Sliding window rate limiter backed by Redis."""

    def __init__(self, redis_client: redis.Redis):
        self._redis = redis_client

    async def check_rate_limit(
        self,
        tenant_id: str,
        endpoint: str,
        limit: int,
        window_seconds: int = 60,
    ) -> dict[str, Any]:
        """Check if the request is within rate limits."""
        key = f"ratelimit:{tenant_id}:{endpoint}"
        now = time.time()
        window_start = now - window_seconds

        pipe = self._redis.pipeline()
        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current requests in window
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set expiry on the key
        pipe.expire(key, window_seconds)

        results = await pipe.execute()
        current_count = results[1]

        if current_count >= limit:
            logger.warning(
                "rate_limit.exceeded",
                tenant=tenant_id,
                endpoint=endpoint,
                count=current_count,
                limit=limit,
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + window_seconds)),
                },
            )

        return {
            "limit": limit,
            "remaining": limit - current_count - 1,
            "reset": int(now + window_seconds),
        }
```

### Task 8.3: Usage Tracking and Cost Allocation

Track every API call per tenant, including prediction requests, feature fetches, and
model registry operations. Compute cost per team based on resource usage.

```python
# src/gateway/middleware/usage.py
from datetime import datetime
from typing import Any
import structlog

logger = structlog.get_logger()


class UsageTracker:
    """Tracks API usage per tenant for cost allocation."""

    def __init__(self, db_session: Any):
        self.db = db_session

    async def record(
        self,
        tenant_id: str,
        team: str,
        endpoint: str,
        method: str,
        model_name: str | None = None,
        response_time_ms: float = 0,
        tokens_used: int = 0,
        compute_seconds: float = 0,
    ) -> None:
        """Record a single API usage event."""
        await self.db.execute(
            """
            INSERT INTO usage_events
                (tenant_id, team, endpoint, method, model_name,
                 response_time_ms, tokens_used, compute_seconds, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            tenant_id, team, endpoint, method, model_name,
            response_time_ms, tokens_used, compute_seconds,
            datetime.utcnow(),
        )

    async def get_team_usage(
        self,
        team: str,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, Any]:
        """Get aggregated usage for a team in a time period."""
        result = await self.db.fetchrow(
            """
            SELECT
                COUNT(*) as total_requests,
                SUM(response_time_ms) as total_response_time_ms,
                SUM(tokens_used) as total_tokens,
                SUM(compute_seconds) as total_compute_seconds,
                COUNT(DISTINCT model_name) as models_used,
                COUNT(DISTINCT endpoint) as endpoints_used
            FROM usage_events
            WHERE team = $1 AND timestamp BETWEEN $2 AND $3
            """,
            team, start_date, end_date,
        )

        return {
            "team": team,
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total_requests": result["total_requests"],
            "total_compute_seconds": result["total_compute_seconds"],
            "estimated_cost_usd": self._compute_cost(result),
        }

    def _compute_cost(self, usage: dict[str, Any]) -> float:
        """Estimate cost based on compute usage."""
        # Pricing: $0.05 per 1000 requests + $0.10 per compute-minute
        request_cost = (usage["total_requests"] / 1000) * 0.05
        compute_cost = (usage["total_compute_seconds"] / 60) * 0.10
        return round(request_cost + compute_cost, 4)
```

### Task 8.4: Tenant Management API

Endpoints for creating tenants, managing API keys, setting quotas, and viewing usage.

**Deliverables:**
- [ ] JWT + API key authentication middleware
- [ ] Sliding window rate limiter with per-tenant Redis keys
- [ ] Usage tracking with PostgreSQL event log
- [ ] Cost allocation: compute estimated cost per team
- [ ] Tenant management API (create, update quotas, revoke keys)
- [ ] Integration test: multi-tenant requests with rate limiting

---

## Phase 9: Monitoring Dashboard

**Duration:** 4--5 days
**Objective:** Build comprehensive monitoring with model performance tracking,
infrastructure metrics, cost visibility, and data quality alerts.

### Task 9.1: Prometheus Metrics

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary

# Prediction metrics
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["model_name", "model_version", "team", "status"],
)

prediction_latency_seconds = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
    ["model_name", "model_version"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

prediction_confidence = Summary(
    "prediction_confidence",
    "Model prediction confidence scores",
    ["model_name"],
)

# Model performance metrics
model_accuracy = Gauge(
    "model_accuracy",
    "Current model accuracy (from ground truth feedback)",
    ["model_name", "model_version"],
)

model_drift_score = Gauge(
    "model_drift_score",
    "Feature or prediction drift score (0=no drift, 1=max drift)",
    ["model_name", "drift_type"],  # drift_type: feature, prediction, concept
)

# Infrastructure metrics
serving_pod_count = Gauge(
    "serving_pod_count",
    "Number of active serving pods",
    ["model_name", "model_version"],
)

feature_store_latency_seconds = Histogram(
    "feature_store_latency_seconds",
    "Feature fetch latency",
    ["feature_view", "store_type"],  # store_type: online, offline
)

# Cost metrics
model_cost_usd = Counter(
    "model_cost_usd_total",
    "Cumulative cost per model",
    ["model_name", "cost_type"],  # cost_type: compute, storage, api
)

team_cost_usd = Counter(
    "team_cost_usd_total",
    "Cumulative cost per team",
    ["team", "cost_type"],
)
```

### Task 9.2: Data and Concept Drift Detection

```python
# src/monitoring/drift.py
from typing import Any
import numpy as np
from scipy import stats


class DriftDetector:
    """Detects feature drift and prediction drift using statistical tests."""

    def detect_feature_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        method: str = "ks",
        threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Detect drift between reference and current feature distributions."""
        if method == "ks":
            statistic, p_value = stats.ks_2samp(reference, current)
        elif method == "chi2":
            # For categorical features
            ref_counts = np.bincount(reference.astype(int))
            cur_counts = np.bincount(current.astype(int))
            max_len = max(len(ref_counts), len(cur_counts))
            ref_padded = np.pad(ref_counts, (0, max_len - len(ref_counts)))
            cur_padded = np.pad(cur_counts, (0, max_len - len(cur_counts)))
            statistic, p_value = stats.chisquare(cur_padded, f_exp=ref_padded + 1)
        elif method == "psi":
            statistic = self._population_stability_index(reference, current)
            p_value = 1.0 if statistic < 0.1 else 0.0  # PSI threshold
        else:
            raise ValueError(f"Unknown method: {method}")

        drift_detected = p_value < threshold
        return {
            "drift_detected": drift_detected,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "threshold": threshold,
            "method": method,
        }

    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> dict[str, Any]:
        """Detect drift in model predictions over time."""
        return self.detect_feature_drift(
            reference_predictions, current_predictions, method="ks"
        )

    def _population_stability_index(
        self, reference: np.ndarray, current: np.ndarray, bins: int = 10
    ) -> float:
        """Calculate Population Stability Index (PSI)."""
        ref_percents, bin_edges = np.histogram(reference, bins=bins)
        cur_percents, _ = np.histogram(current, bins=bin_edges)

        ref_percents = ref_percents / len(reference) + 1e-6
        cur_percents = cur_percents / len(current) + 1e-6

        psi = np.sum(
            (cur_percents - ref_percents) * np.log(cur_percents / ref_percents)
        )
        return float(psi)
```

### Task 9.3: Alert Definitions

Create alert rules for common failure modes.

```yaml
# infrastructure/prometheus/alert_rules.yaml
groups:
  - name: model_alerts
    rules:
      - alert: HighModelErrorRate
        expr: rate(prediction_requests_total{status="error"}[5m]) / rate(prediction_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model {{ $labels.model_name }} error rate above 5%"

      - alert: HighPredictionLatency
        expr: histogram_quantile(0.99, rate(prediction_latency_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Model {{ $labels.model_name }} p99 latency above 2s"

      - alert: FeatureDriftDetected
        expr: model_drift_score{drift_type="feature"} > 0.3
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Feature drift detected for {{ $labels.model_name }}"

      - alert: ModelAccuracyDrop
        expr: model_accuracy < 0.8
        for: 30m
        labels:
          severity: critical
        annotations:
          summary: "Model {{ $labels.model_name }} accuracy dropped below 80%"
```

### Task 9.4: Grafana Dashboard Configuration

Create pre-built dashboards:

1. **Platform Overview** --- Total models, deployments, requests/sec, overall cost
2. **Model Performance** --- Per-model accuracy, latency, error rate, drift scores
3. **Infrastructure** --- Pod counts, CPU/memory usage, feature store latency
4. **Cost Management** --- Cost per model, cost per team, cost trends

### Task 9.5: React Monitoring Dashboard

Build the frontend dashboard with:

- Model catalog with search, filter, and health status indicators
- Deployment status with canary progress and rollback buttons
- Real-time metrics charts (latency, error rate, throughput)
- Cost breakdown per team and per model
- Alert history and current active alerts

**Deliverables:**
- [ ] Prometheus metrics for predictions, infrastructure, and costs
- [ ] `DriftDetector` with KS test, chi-square, and PSI methods
- [ ] Alert rules for error rate, latency, drift, and accuracy
- [ ] Four Grafana dashboards
- [ ] React monitoring dashboard with real-time charts
- [ ] Integration test: generate predictions, verify metrics appear in Prometheus

---

## Phase 10: Testing & CI/CD

**Duration:** 3--4 days
**Objective:** Build a comprehensive test suite covering unit, integration, end-to-end,
and Kubernetes-specific tests, plus a full CI/CD pipeline.

### Task 10.1: Unit Tests

- **Registry service** --- Model registration, approval workflow, lineage queries
- **Feature service** --- Feature listing, online/offline fetch, validation
- **Serving service** --- Deployment creation, canary logic, A/B routing
- **Rate limiter** --- Sliding window correctness, concurrent access
- **Drift detector** --- KS test, PSI calculation, edge cases

### Task 10.2: Integration Tests

- **Model lifecycle** --- Register -> approve -> deploy -> predict -> monitor
- **Feature serving** --- Define features -> materialize -> fetch online
- **Multi-tenant** --- Two tenants with different quotas, verify isolation

### Task 10.3: End-to-End Tests

```python
# tests/e2e/test_full_workflow.py
import pytest
import httpx


@pytest.mark.e2e
class TestFullMLWorkflow:
    """End-to-end test: train a model and deploy it through the platform."""

    async def test_model_to_production(self, platform_url: str, api_key: str):
        async with httpx.AsyncClient(
            base_url=platform_url,
            headers={"X-API-Key": api_key},
        ) as client:
            # Step 1: Register model
            resp = await client.post("/api/v1/models", json={
                "name": "test-classifier",
                "description": "E2E test model",
                "model_uri": "runs:/abc123/model",
                "metrics": {"accuracy": 0.95, "f1": 0.93},
            })
            assert resp.status_code == 201
            model = resp.json()

            # Step 2: Approve for production
            resp = await client.post(
                f"/api/v1/models/{model['model_name']}/approve",
                json={
                    "version": model["version"],
                    "target_stage": "production",
                    "notes": "E2E test approval",
                },
            )
            assert resp.status_code == 200
            assert resp.json()["approved"] is True

            # Step 3: Deploy
            resp = await client.post("/api/v1/deployments", json={
                "model_name": model["model_name"],
                "model_version": model["version"],
                "strategy": "recreate",
                "replicas": 1,
            })
            assert resp.status_code == 201
            deployment = resp.json()

            # Step 4: Wait for deployment to be ready
            # ... poll deployment status ...

            # Step 5: Make a prediction
            resp = await client.post(
                f"/api/v1/predict/{model['model_name']}",
                json={"features": {"age": 25, "income": 50000}},
            )
            assert resp.status_code == 200
            assert "prediction" in resp.json()

            # Step 6: Verify metrics exist
            resp = await client.get(
                f"/api/v1/monitoring/{model['model_name']}/metrics"
            )
            assert resp.status_code == 200
            assert resp.json()["total_predictions"] >= 1
```

### Task 10.4: CI/CD Pipeline

```yaml
# .github/workflows/ci.yaml
name: CI
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install ruff mypy
      - run: ruff check src/ tests/ sdk/ cli/
      - run: mypy src/ --ignore-missing-imports

  unit-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit/ -v --cov=src --cov-report=xml

  integration-test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: ["6379:6379"]
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -v

  build:
    needs: [lint, unit-test, integration-test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker compose build
      - name: Run Helm lint
        run: |
          helm lint infrastructure/helm/ml-platform/
          helm lint infrastructure/helm/model-serving/
```

**Deliverables:**
- [ ] Unit tests with >80% coverage on core services
- [ ] Integration tests with real Redis/PostgreSQL
- [ ] End-to-end test covering the full model lifecycle
- [ ] GitHub Actions CI pipeline (lint, unit test, integration test, build)
- [ ] CD pipelines for staging and production

---

## Phase 11: Documentation & Developer Experience

**Duration:** 3--4 days
**Objective:** Build the SDK, CLI, and documentation that make the platform easy for
ML engineers to use. The platform is only as good as its developer experience.

### Task 11.1: Python SDK

Create a pip-installable SDK that wraps the platform API.

```python
# sdk/mlplatform/client.py
from typing import Any
import httpx


class MLPlatformClient:
    """Python SDK for the ML Platform."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": api_key},
            timeout=30.0,
        )

    def register_model(
        self,
        name: str,
        model_uri: str,
        description: str = "",
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Register a new model version in the platform."""
        resp = self._client.post("/api/v1/models", json={
            "name": name,
            "model_uri": model_uri,
            "description": description,
            "metrics": metrics or {},
            "tags": tags or {},
        })
        resp.raise_for_status()
        return resp.json()

    def deploy(
        self,
        model_name: str,
        model_version: str,
        strategy: str = "canary",
        replicas: int = 2,
    ) -> dict[str, Any]:
        """Deploy a model to production."""
        resp = self._client.post("/api/v1/deployments", json={
            "model_name": model_name,
            "model_version": model_version,
            "strategy": strategy,
            "replicas": replicas,
        })
        resp.raise_for_status()
        return resp.json()

    def predict(
        self, model_name: str, features: dict[str, Any]
    ) -> dict[str, Any]:
        """Make a prediction using a deployed model."""
        resp = self._client.post(
            f"/api/v1/predict/{model_name}",
            json={"features": features},
        )
        resp.raise_for_status()
        return resp.json()

    def get_model_metrics(self, model_name: str) -> dict[str, Any]:
        """Get monitoring metrics for a deployed model."""
        resp = self._client.get(
            f"/api/v1/monitoring/{model_name}/metrics"
        )
        resp.raise_for_status()
        return resp.json()
```

### Task 11.2: CLI Tool

Build a CLI that ML engineers use from their terminal.

```python
# cli/mlplatform_cli/commands/model.py
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Model registry commands")
console = Console()


@app.command("register")
def register_model(
    name: str = typer.Argument(..., help="Model name"),
    uri: str = typer.Option(..., "--uri", help="MLflow model URI"),
    description: str = typer.Option("", "--desc", help="Model description"),
):
    """Register a new model version."""
    from mlplatform import MLPlatformClient

    client = MLPlatformClient.from_config()
    result = client.register_model(name=name, model_uri=uri, description=description)

    console.print(f"[green]Model registered successfully![/green]")
    console.print(f"  Name:    {result['model_name']}")
    console.print(f"  Version: {result['version']}")
    console.print(f"  Stage:   {result['stage']}")


@app.command("list")
def list_models(
    team: str = typer.Option(None, "--team", help="Filter by team"),
    stage: str = typer.Option(None, "--stage", help="Filter by stage"),
):
    """List all registered models."""
    from mlplatform import MLPlatformClient

    client = MLPlatformClient.from_config()
    models = client.list_models(team=team, stage=stage)

    table = Table(title="Registered Models")
    table.add_column("Name", style="cyan")
    table.add_column("Latest Version", style="green")
    table.add_column("Stage", style="yellow")
    table.add_column("Team", style="blue")
    table.add_column("Accuracy", justify="right")

    for m in models:
        table.add_row(
            m["name"],
            str(m["latest_version"]),
            m["stage"],
            m["team"],
            f"{m.get('accuracy', 'N/A')}",
        )

    console.print(table)
```

**Example usage:**

```bash
# Register a model
mlp model register my-classifier --uri "runs:/abc123/model" --desc "XGBoost classifier v2"

# List models
mlp model list --team growth --stage production

# Deploy
mlp deploy create --model my-classifier --version 3 --strategy canary

# Check deployment status
mlp deploy status --model my-classifier

# Rollback
mlp deploy rollback --model my-classifier

# View metrics
mlp monitor metrics --model my-classifier --period 24h
```

### Task 11.3: Getting Started Guide

Write a comprehensive onboarding document that walks a new user through:
1. Installing the SDK and CLI
2. Configuring authentication
3. Registering their first model
4. Deploying to staging
5. Promoting to production
6. Setting up monitoring alerts

### Task 11.4: Example Notebooks

Create Jupyter notebooks demonstrating common workflows:
- `01_register_model.ipynb` --- Train a simple model and register it
- `02_define_features.ipynb` --- Create feature definitions and materialize
- `03_deploy_model.ipynb` --- Deploy with canary and monitor rollout
- `04_monitor_model.ipynb` --- Check drift, view metrics, set alerts

**Deliverables:**
- [ ] Python SDK package (`pip install mlplatform`)
- [ ] CLI tool with model, deploy, feature, and monitor commands
- [ ] GETTING_STARTED.md onboarding guide
- [ ] Four example notebooks
- [ ] SDK and CLI unit tests
- [ ] Verified: a new user can go from install to deployed model following the guide

---

## Skills Checklist

By completing this project, you will have demonstrated:

| Skill | Where You Used It |
|-------|-------------------|
| **Platform Engineering** | Designed and built a multi-service ML platform |
| **Kubernetes** | Minikube cluster, Helm charts, HPA, resource quotas |
| **Infrastructure as Code** | Terraform modules, multi-environment provisioning |
| **Feature Store** | Feast setup, online/offline serving, feature discovery |
| **Model Registry** | MLflow-based registry with approval workflows and lineage |
| **Model Serving** | Multi-model serving, canary, A/B testing, auto-scaling |
| **Multi-Tenant Architecture** | JWT auth, per-tenant rate limiting, cost allocation |
| **CI/CD for ML** | Automated model testing, staging, canary rollout |
| **Monitoring & Observability** | Prometheus, Grafana, drift detection, alerting |
| **API Design** | RESTful API with auth, rate limiting, versioning |
| **Developer Experience** | SDK, CLI, documentation, example notebooks |
| **Cost Management** | Per-model and per-team cost tracking and allocation |
| **Data Quality** | Feature validation, drift detection, freshness monitoring |
| **Deployment Strategies** | Canary, blue-green, shadow, rollback |
| **Helm Charts** | Kubernetes application packaging and configuration |
| **Testing** | Unit, integration, E2E with real infrastructure |
| **Database Design** | PostgreSQL schema for registry, usage, approvals |
