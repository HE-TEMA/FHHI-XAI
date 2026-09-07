"""Metric view of the shared PIDNet flood preprocessing.

All image and mask preprocessing is inherited from ``flood_dataset`` so the
inputs used for CRP/PCX and evaluation cannot silently diverge.
"""

import numpy as np

from src.datasets.flood_dataset import FloodDataset as _CRPFloodDataset


class FloodDataset(_CRPFloodDataset):
    """Return PIDNet evaluation metadata in addition to image/label tensors."""

    def __init__(self, *args, return_or_dims=True, **kwargs):
        super().__init__(*args, return_or_dims=return_or_dims, **kwargs)

    def __getitem__(self, index):
        image, label, edge, size, image_or, name = self._load_sample(index)
        if self.return_or_dims:
            return image.clone(), label.clone(), edge.copy(), np.array(size), image_or.copy(), name
        return image.clone(), label.clone(), edge.copy(), np.array(size), name

    @classmethod
    def from_config(cls, cfg, split="val", **kwargs):
        """Build from PIDNet config, converting IMAGE_SIZE [W,H] to (H,W)."""
        list_path = kwargs.pop("list_path", None)
        if list_path is None:
            list_path = cfg.DATASET.TEST_SET if split == "val" else cfg.DATASET.TRAIN_SET
        image_size = cfg.TEST.IMAGE_SIZE if split == "val" else cfg.TRAIN.IMAGE_SIZE
        base_size = cfg.TEST.BASE_SIZE if split == "val" else cfg.TRAIN.BASE_SIZE
        return cls(
            root=cfg.DATASET.ROOT,
            split=split,
            num_classes=cfg.DATASET.NUM_CLASSES,
            multi_scale=False if split == "val" else cfg.TRAIN.MULTI_SCALE,
            flip=False if split == "val" else cfg.TRAIN.FLIP,
            ignore_label=cfg.TRAIN.IGNORE_LABEL,
            base_size=base_size,
            crop_size=(image_size[1], image_size[0]),
            list_path=list_path,
            **kwargs,
        )


Flood = FloodDataset
