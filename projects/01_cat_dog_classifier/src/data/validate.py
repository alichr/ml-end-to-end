"""Scan for and remove corrupted images from the dataset.

Usage:
    python -m src.data.validate
"""

import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def validate_images(input_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR) -> None:
    """Copy valid images to output_dir, skipping corrupted ones."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(input_dir.glob("*.jpg"))
    if not all_images:
        logger.error("No images found in %s", input_dir)
        return

    valid_count = 0
    corrupted_count = 0

    for img_path in all_images:
        try:
            with Image.open(img_path) as img:
                img.verify()
            # Re-open after verify (verify closes the file)
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                dest = output_dir / img_path.name
                img.save(dest, "JPEG")
            valid_count += 1
        except Exception as e:
            logger.warning("Corrupted image %s: %s", img_path.name, e)
            corrupted_count += 1

    logger.info(
        "Validation complete. Valid: %d, Corrupted: %d, Output: %s",
        valid_count,
        corrupted_count,
        output_dir,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validate_images()
