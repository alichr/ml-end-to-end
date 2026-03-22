"""Pick the best MLflow run and register/promote the model to Production.

Usage:
    python scripts/promote_model.py
    python scripts/promote_model.py --run-id <specific_run_id>
"""

import argparse

import mlflow
from mlflow.tracking import MlflowClient


MODEL_NAME = "cat-dog-classifier"


def promote_best_model(run_id: str | None = None) -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name("cat-vs-dog")

    if experiment is None:
        print("No 'cat-vs-dog' experiment found. Run training first.")
        return

    if run_id is None:
        # Find the run with the highest best_val_accuracy
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.best_val_accuracy DESC"],
            max_results=1,
        )
        if not runs:
            print("No runs found.")
            return
        best_run = runs[0]
        run_id = best_run.info.run_id
        best_acc = best_run.data.metrics.get("best_val_accuracy", 0)
        run_name = best_run.data.tags.get("mlflow.runName", "unnamed")
        print(f"Best run: {run_name} (id={run_id}, val_acc={best_acc:.4f})")
    else:
        run = client.get_run(run_id)
        best_acc = run.data.metrics.get("best_val_accuracy", 0)
        run_name = run.data.tags.get("mlflow.runName", "unnamed")
        print(f"Selected run: {run_name} (id={run_id}, val_acc={best_acc:.4f})")

    # Register the model
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)
    print(f"Registered model version {result.version}")

    # Promote to Production alias
    client.set_registered_model_alias(MODEL_NAME, "production", result.version)
    print(f"Model version {result.version} promoted to 'production'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Specific MLflow run ID to promote")
    args = parser.parse_args()
    promote_best_model(args.run_id)
