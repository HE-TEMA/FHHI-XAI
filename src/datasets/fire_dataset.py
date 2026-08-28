import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


class FireDataset:
    class_names = ["background", "fire"]

    def __init__(
        self,
        root: str = "data/FireSeg",
        split: str = "train",
        list_path: Optional[str] = None,
        modality: str = "rgb",
        require_fire: bool = False,
        crop_size: Tuple[int, int] = (720, 1280),   # CROP_SIZE from AUTH hydra config
        ignore_label: int = 255,
        mean_rgb: List[float] = [0.485, 0.456, 0.406],
        std_rgb: List[float] = [0.229, 0.224, 0.225],
        mean_ir: float = 0.5,
        std_ir: float = 0.25,
    ):
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = np.asarray(list(mean_rgb) + [mean_ir], dtype=np.float32)
        self.std = np.asarray(list(std_rgb) + [std_ir], dtype=np.float32)
        self.num_classes = len(self.class_names)

        list_file = list_path or os.path.join(root, "lists", f"fireseg_{split}.txt")
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Split list not found: {list_file}")
        with open(list_file) as handle:
            self.files = [line.strip() for line in handle if line.strip()]
        if not self.files:
            raise ValueError(f"Split list is empty: {list_file}")

        # The checkpoint segments RGB well (fire IoU 0.82) but barely responds to
        # the IR-only samples (0.16), so restrict the split unless asked otherwise.
        if modality not in ("rgb", "ir", "all"):
            raise ValueError(f"modality must be rgb, ir or all; got {modality!r}")
        self.modality = modality
        if modality != "all":
            want_ir = modality == "ir"
            self.files = [f for f in self.files if self._has_variant(f, "ir") == want_ir]
            if not self.files:
                raise ValueError(f"No {modality} samples in {list_file}")

        # remove samples without fire to get more meaningful prototypes
        self.require_fire = require_fire
        if require_fire:
            keep = self._fire_flags(list_file, modality)
            self.files = [f for f, has in zip(self.files, keep) if has]
            if not self.files:
                raise ValueError(f"No sample with fire in {list_file}")

    def __len__(self):
        return len(self.files)

    def _resolve(self, entry: str) -> str:
        """List entries are written relative to the parent of the FireSeg folder."""
        prefix = os.path.basename(os.path.normpath(self.root)) + "/"
        if entry.startswith(prefix):
            entry = entry[len(prefix):]
        return os.path.join(self.root, entry)

    def _fire_flags(self, list_file: str, modality: str) -> List[bool]:
        cache = Path(self.root) / f".fire_flags_{Path(list_file).stem}_{modality}.json"
        if cache.is_file():
            cached = json.loads(cache.read_text())
            if cached.get("files") == self.files:
                return cached["flags"]

        flags = []
        for entry in self.files:
            label_path = self._resolve(
                entry.replace("XXX", "gt").replace(".jpg", ".png").replace(".JPG", ".png")
            )
            label = np.asarray(Image.open(label_path).convert("L"))
            flags.append(bool((label > 0.1).any()))
        try:
            cache.write_text(json.dumps({"files": self.files, "flags": flags}))
        except OSError:
            pass
        return flags

    def _has_variant(self, entry: str, variant: str) -> bool:
        base, _ = os.path.splitext(self._resolve(entry.replace("XXX", variant)))
        return any(os.path.isfile(base + ext) for ext in (".png", ".jpg", ".JPG"))

    def _find_image(self, path_base: str) -> str:
        base, _ = os.path.splitext(path_base)
        for ext in (".png", ".jpg", ".JPG"):
            if os.path.isfile(base + ext):
                return base + ext
        raise FileNotFoundError(base)

    def load_sample(self, index: int):
        """HWC uint8 image with four channels, plus the binary label."""
        entry = self.files[index]

        label_path = self._resolve(
            entry.replace("XXX", "gt").replace(".jpg", ".png").replace(".JPG", ".png")
        )
        label = np.asarray(Image.open(label_path).convert("L")).astype(np.uint8)
        # Their threshold: any non-zero pixel counts as fire.
        label = (label > 0.1).astype(np.uint8)
        h, w = label.shape[:2]

        try:
            rgb_path = self._find_image(self._resolve(entry.replace("XXX", "rgb")))
            rgb = np.asarray(Image.open(rgb_path).convert("RGB")).copy()
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        except (FileNotFoundError, OSError):
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

        try:
            ir_path = self._find_image(self._resolve(entry.replace("XXX", "ir")))
            # The IR files are false-coloured RGB; convert('L') is their reduction.
            ir = np.asarray(Image.open(ir_path).convert("L")).copy()
            ir = cv2.resize(ir, (w, h), interpolation=cv2.INTER_LINEAR)
            ir = ir.reshape(h, w, 1)
        except (FileNotFoundError, OSError):
            ir = np.zeros((h, w, 1), dtype=np.uint8)

        return np.concatenate([rgb, ir], axis=2), label

    def __getitem__(self, index: int):
        image, label = self.load_sample(index)

        target_h, target_w = self.crop_size
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)

        return torch.from_numpy(image.copy()), torch.from_numpy(label.astype(np.int64))

    def reverse_normalization(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.detach().cpu().float()
        mean = torch.tensor(self.mean).view(-1, 1, 1)
        std = torch.tensor(self.std).view(-1, 1, 1)
        return ((tensor * std + mean) * 255).clamp(0, 255)

    def reverse_augmentation(self, data: torch.Tensor) -> torch.Tensor:
        """Displayable RGB tensor. `plotpcx_gpu` calls this for every panel, and
        its drawing utilities expect three channels, so the IR plane is dropped."""
        image = self.reverse_normalization(data)
        # Keep a batch dimension if it is there: plotpcx_gpu indexes the result
        # as `sample[:3, :, :][0]`, which needs 4D to end up with three channels.
        if image.ndim == 3 and image.shape[0] >= 3:
            image = image[:3]
        elif image.ndim == 4 and image.shape[1] >= 3:
            image = image[:, :3]
        return image.to(torch.float32)
