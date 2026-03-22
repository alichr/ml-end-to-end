"""Split processed images into train/val/test sets (stratified).

Usage:
    python -m src.data.split
"""

import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_dataset(
    input_dir: Path = PROCESSED_DIR,
    output_dir: Path = SPLITS_DIR,
    seed: int = 42,
) -> None:
    """Split images into train/val/test with stratified sampling."""
    rng = np.random.default_rng(seed)

    for split in ["train", "val", "test"]:
        for label in ["cat", "dog"]:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    for label in ["cat", "dog"]:
        images = sorted(input_dir.glob(f"{label}.*.jpg"))
        if not images:
            logger.warning("No %s images found in %s", label, input_dir)
            continue

        # Shuffle
        indices = rng.permutation(len(images))
        n_train = int(len(images) * TRAIN_RATIO)
        n_val = int(len(images) * VAL_RATIO)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        for split_name, split_indices in [
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ]:
            for i in split_indices:
                src = images[i]
                dst = output_dir / split_name / label / src.name
                shutil.copy2(src, dst)

        logger.info(
            "%s — train: %d, val: %d, test: %d",
            label,
            len(train_idx),
            len(val_idx),
            len(test_idx),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    split_dataset()
