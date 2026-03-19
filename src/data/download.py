"""Download the Kaggle Dogs vs Cats dataset.

Usage:
    python -m src.data.download

Prerequisites:
    1. Create a Kaggle account at https://www.kaggle.com
    2. Go to https://www.kaggle.com/settings → API → Create New Token
    3. This downloads a kaggle.json file. Place it at ~/.kaggle/kaggle.json
    4. Run: chmod 600 ~/.kaggle/kaggle.json
"""

import logging
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET = "shaunthesheep/microsoft-catsvsdogs-dataset"
RAW_DIR = Path("data/raw")
EXPECTED_COUNT = 25000


def download_dataset(output_dir: Path = RAW_DIR) -> None:
    """Download and extract the Dogs vs Cats dataset from Kaggle.

    This function is idempotent — if the data already exists and looks
    complete, it skips the download.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    image_files = list(output_dir.glob("*.jpg"))
    if len(image_files) >= EXPECTED_COUNT:
        logger.info("Dataset already exists with %d images, skipping download.", len(image_files))
        return

    logger.info("Downloading '%s' dataset from Kaggle...", DATASET)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    # Download dataset files
    api.dataset_download_files(DATASET, path=output_dir)

    # Extract the downloaded zip
    for zip_file in output_dir.glob("*.zip"):
        logger.info("Extracting %s...", zip_file.name)
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(output_dir)
        zip_file.unlink()

    # The Microsoft dataset has PetImages/Cat/ and PetImages/Dog/ structure
    # Flatten into output_dir as cat.0.jpg, cat.1.jpg, dog.0.jpg, dog.1.jpg, etc.
    import shutil

    pet_images = output_dir / "PetImages"
    if pet_images.is_dir():
        for label_dir in pet_images.iterdir():
            if not label_dir.is_dir():
                continue
            label = label_dir.name.lower()  # "cat" or "dog"
            for i, img in enumerate(sorted(label_dir.iterdir())):
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    img.rename(output_dir / f"{label}.{i}.jpg")
        shutil.rmtree(pet_images)

    # Clean up non-image files
    for f in output_dir.iterdir():
        if f.is_file() and f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            f.unlink()

    # Verify
    image_files = list(output_dir.glob("*.jpg"))
    logger.info("Download complete. %d images in %s", len(image_files), output_dir)

    if len(image_files) < EXPECTED_COUNT:
        logger.warning(
            "Expected ~%d images but found %d. Check the download.",
            EXPECTED_COUNT,
            len(image_files),
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_dataset()
