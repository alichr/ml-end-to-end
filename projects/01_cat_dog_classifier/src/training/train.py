"""Training loop with validation, checkpointing, early stopping, and MLflow tracking."""

import multiprocessing
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import yaml  # type: ignore[import-untyped]
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.data.dataset import CatDogDataset
from src.data.transforms import get_train_transform, inference_transform
from src.model.classifier import CatDogClassifier

# Prevent DataLoader worker deadlocks when running multiple training runs
multiprocessing.set_start_method("spawn", force=True)
matplotlib.use("Agg")


def load_config(config_path: str = "configs/train_config.yaml") -> dict:  # type: ignore[type-arg]
    with open(config_path) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def compute_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
) -> None:
    """Generate and save a confusion matrix plot."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["cat", "dog"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title("Validation Confusion Matrix")
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: dict
) -> torch.optim.lr_scheduler.LRScheduler:
    name = train_cfg.get("scheduler", "cosine")
    if name == "step":
        return StepLR(optimizer, step_size=train_cfg.get("step_size", 3), gamma=0.1)
    return CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])


def train(config_path: str = "configs/train_config.yaml") -> None:
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    augmentation = data_cfg.get("augmentation", "default")
    train_dataset = CatDogDataset(
        data_cfg["train_dir"], transform=get_train_transform(augmentation)
    )
    val_dataset = CatDogDataset(data_cfg["val_dir"], transform=inference_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    # Model
    model = CatDogClassifier(
        num_classes=model_cfg["num_classes"],
        freeze_backbone=model_cfg["freeze_backbone"],
        unfreeze_last_n=model_cfg.get("unfreeze_last_n", 0),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
    )
    scheduler = build_scheduler(optimizer, train_cfg)

    # Checkpointing
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0

    # MLflow tracking
    experiment_name = config.get("experiment_name", "cat-vs-dog")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=config.get("run_name")):
        mlflow.log_params(
            {
                "model": model_cfg["name"],
                "freeze_backbone": model_cfg["freeze_backbone"],
                "unfreeze_last_n": model_cfg.get("unfreeze_last_n", 0),
                "epochs": train_cfg["epochs"],
                "batch_size": train_cfg["batch_size"],
                "learning_rate": train_cfg["learning_rate"],
                "optimizer": train_cfg["optimizer"],
                "scheduler": train_cfg.get("scheduler", "cosine"),
                "augmentation": augmentation,
                "device": str(device),
                "trainable_params": trainable,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
            }
        )

        for epoch in range(train_cfg["epochs"]):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{train_cfg['epochs']} — "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"LR: {lr:.6f}"
            )

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": lr,
                },
                step=epoch,
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
                print(f"  -> New best model saved (val_acc={val_acc:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= train_cfg["early_stopping_patience"]:
                print(
                    f"Early stopping at epoch {epoch + 1} (no improvement for "
                    f"{train_cfg['early_stopping_patience']} epochs)"
                )
                break

        # Log best accuracy and artifacts
        mlflow.log_metric("best_val_accuracy", best_val_acc)

        # Load best model and generate confusion matrix
        model.load_state_dict(torch.load(checkpoint_dir / "best_model.pth", weights_only=True))
        cm_path = checkpoint_dir / "confusion_matrix.png"
        compute_confusion_matrix(model, val_loader, device, cm_path)

        mlflow.log_artifact(str(checkpoint_dir / "best_model.pth"))
        mlflow.log_artifact(str(cm_path))

        # Log model to MLflow for registry
        mlflow.pytorch.log_model(model, "model")

        print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
        print(f"Confusion matrix saved to {cm_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    train(args.config)
