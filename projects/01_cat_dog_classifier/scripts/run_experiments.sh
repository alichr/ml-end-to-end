#!/usr/bin/env bash
# Run all Phase 3 experiments sequentially.
set -e

echo "=== Experiment 1/5: Baseline (frozen backbone) ==="
python -m src.training.train --config configs/experiment_baseline.yaml

echo ""
echo "=== Experiment 2/5: Fine-tune (unfreeze last 3 layers) ==="
python -m src.training.train --config configs/experiment_finetune.yaml

echo ""
echo "=== Experiment 3/5: Strong augmentation ==="
python -m src.training.train --config configs/experiment_augmentation.yaml

echo ""
echo "=== Experiment 4/5: Batch size 64 ==="
python -m src.training.train --config configs/experiment_batch_size.yaml

echo ""
echo "=== Experiment 5/5: StepLR scheduler ==="
python -m src.training.train --config configs/experiment_scheduler.yaml

echo ""
echo "All experiments complete! Run 'mlflow ui' to compare results."
