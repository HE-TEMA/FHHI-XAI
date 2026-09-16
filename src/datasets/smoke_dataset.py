"""Reference dataset and inference preprocessing shared with pidnet_smoke_BRK.ipynb."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SmokeDataset(Dataset):
    class_names = ("background", "smoke")
    ignore_label = -1
    target_size = (720, 1280)  # height, width
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, root_dir, min_smoke_coverage=0.0):
        root = Path(root_dir)
        image_dir = root / "images"
        mask_dir = root / "segmenation" / "smoke"  # directory spelling on disk
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Smoke image/mask directories missing under {root}")

        def by_stem(directory):
            return {p.stem.casefold(): p for p in directory.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}

        self.images = by_stem(image_dir)
        self.masks = by_stem(mask_dir)
        paired = sorted(self.images.keys() & self.masks.keys())
        self.coverages = {}
        for stem in paired:
            mask = np.asarray(Image.open(self.masks[stem]).convert("L"))
            self.coverages[stem] = float((mask >= 128).mean())
        self.stems = [stem for stem in paired
                      if self.coverages[stem] > min_smoke_coverage]
        self.files = [{"img": self.images[s], "label": self.masks[s],
                       "name": s, "smoke_coverage": self.coverages[s]}
                      for s in self.stems]
        if not self.stems:
            raise ValueError(f"No paired smoke images found under {root}")

    def __len__(self):
        return len(self.stems)

    @classmethod
    def transform(cls, image):
        """Match the notebook: RGB, normalize, then bilinear resize."""
        rgb = np.asarray(Image.fromarray(np.asarray(image)).convert("RGB"))
        x = rgb.astype(np.float32) / 255.0
        x = (x - cls.mean) / cls.std
        x = cv2.resize(x, (cls.target_size[1], cls.target_size[0]),
                       interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()

    def __getitem__(self, index):
        stem = self.stems[index]
        rgb = np.asarray(Image.open(self.images[stem]).convert("RGB"))
        raw_mask = np.asarray(Image.open(self.masks[stem]).convert("L"))
        x = self.transform(rgb)
        y = cv2.resize((raw_mask >= 128).astype(np.uint8),
                       (self.target_size[1], self.target_size[0]),
                       interpolation=cv2.INTER_NEAREST)
        return x, torch.from_numpy(np.ascontiguousarray(y)).long()

    def reverse_augmentation(self, tensor):
        x = tensor.detach().cpu().float()
        mean = torch.as_tensor(self.mean).view(3, 1, 1)
        std = torch.as_tensor(self.std).view(3, 1, 1)
        return ((x * std + mean).clamp(0, 1) * 255.0)

    reverse_normalization = reverse_augmentation
