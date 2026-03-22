"""PyTorch Dataset for cat vs dog classification."""

from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CatDogDataset(Dataset):
    """Dataset that lazily loads cat/dog images from a split directory.

    Expected directory structure:
        root/
            cat/
                cat.0.jpg
                cat.1.jpg
                ...
            dog/
                dog.0.jpg
                dog.1.jpg
                ...
    """

    CLASS_TO_IDX = {"cat": 0, "dog": 1}
    IDX_TO_CLASS = {0: "cat", 1: "dog"}

    def __init__(
        self,
        root: str | Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_name, class_idx in self.CLASS_TO_IDX.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
