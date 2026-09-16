"""RGB burnt-area data shipped as Segmentation_Burnt.zip / Segmentation_Blaze.

Blue is burnt and orange is background (verified against image/mask pairs).
Black and unknown colors are unlabelled. RGB normalization and 720x1280
inference resize follow PIDNet_burnt/tools/eval.py in burnt_segmentation.zip.
The supplied RAS training loader uses binary masks, so its grayscale threshold
must not be applied to the colored masks in this reference dataset.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


class BurntDataset:
    class_names = ["background", "burnt"]
    color_list = [[216, 82, 24], [0, 113, 188]]
    num_classes = 2

    def __init__(self, root="data/Segmentation_Blaze", split="train",
                 require_burnt=True, min_burnt_coverage=0.0,
                 crop_size=(720, 1280), channel_order="RGB", ignore_label=-1,
                 mean=(0.4951, 0.509, 0.4397), std=(0.176, 0.1487, 0.218)):
        self.root = Path(root)
        self.split = split
        self.crop_size = tuple(int(x) for x in crop_size)
        self.channel_order = channel_order.upper()
        self.ignore_label = int(ignore_label)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.require_burnt = bool(require_burnt)
        self.min_burnt_coverage = float(min_burnt_coverage)
        if not 0 <= self.min_burnt_coverage < 1:
            raise ValueError("min_burnt_coverage must be in [0, 1)")
        if self.channel_order not in ("RGB", "BGR"):
            raise ValueError("channel_order must be RGB or BGR")
        if len(self.crop_size) != 2 or min(self.crop_size) <= 0:
            raise ValueError("crop_size must contain positive height and width")
        if self.mean.shape != (3,) or self.std.shape != (3,) or (self.std <= 0).any():
            raise ValueError("mean/std must contain three values with positive std")
        self.files = self._pair_files()
        if self.require_burnt or self.min_burnt_coverage > 0:
            kept = []
            for item in self.files:
                label = self._load_label(item["label"])
                # Fraction of the entire image, as in the fire coverage filter.
                if (label == 1).mean() > self.min_burnt_coverage:
                    kept.append(item)
            self.files = kept
        if not self.files:
            raise ValueError("No burnt samples remain after pairing/coverage filtering")

    @staticmethod
    def _key(path):
        stem = path.stem.replace("_label_ground-truth_coco-panoptic", "")
        return re.sub(r"[^a-z0-9]", "", stem.lower())

    def _pair_files(self):
        def scan(folder):
            result = {}
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                key = self._key(path)
                if key in result:
                    raise ValueError(f"Ambiguous burnt pairing: {result[key]} and {path}")
                result[key] = path
            return result

        images = scan(self.root / "RGB" / self.split)
        masks = scan(self.root / "masks" / self.split)
        if images.keys() != masks.keys():
            raise ValueError(f"Unpaired burnt files: images={sorted(images.keys() - masks.keys())}, "
                             f"masks={sorted(masks.keys() - images.keys())}")
        return [{"img": str(images[k]), "label": str(masks[k]), "name": images[k].stem}
                for k in sorted(images)]

    def __len__(self):
        return len(self.files)

    def color2label(self, colors):
        label = np.full(colors.shape[:2], self.ignore_label, dtype=np.int32)
        for class_id, color in enumerate(self.color_list):
            label[cv2.inRange(colors, tuple(color), tuple(color)) != 0] = class_id
        return label

    def _load_label(self, path):
        with Image.open(path) as mask:
            return self.color2label(np.asarray(mask.convert("RGB")))

    def load_sample(self, index):
        item = self.files[index]
        with Image.open(item["img"]) as image:
            rgb = np.array(image.convert("RGB"))
        return rgb, self._load_label(item["label"])

    def transform(self, image):
        """Shared preprocessing for reference samples and live RGB images."""
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Burnt preprocessing expects an HWC uint8 RGB image")
        h, w = self.crop_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.channel_order == "BGR":
            image = image[:, :, ::-1]
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(image.transpose(2, 0, 1).copy())

    def __getitem__(self, index):
        image, label = self.load_sample(index)
        h, w = self.crop_size
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        return self.transform(image), torch.from_numpy(label.astype(np.int64))

    def reverse_normalization(self, data):
        tensor = data.detach().cpu().float()
        mean = torch.from_numpy(self.mean).view(3, 1, 1)
        std = torch.from_numpy(self.std).view(3, 1, 1)
        rgb = ((tensor * std + mean) * 255).clamp(0, 255)
        if self.channel_order == "BGR":
            rgb = rgb.flip(-3)
        return rgb

    reverse_augmentation = reverse_normalization

    def bank_metadata(self):
        """Portable record of the exact subset/order and preprocessing."""
        return {
            "dataset": "burnt", "split": self.split,
            "files": [item["name"] for item in self.files],
            "crop_size": list(self.crop_size), "channel_order": self.channel_order,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "color_list": self.color_list, "ignore_label": self.ignore_label,
            "require_burnt": self.require_burnt,
            "min_burnt_coverage": self.min_burnt_coverage,
        }
