"""Manifest-backed dataset view used consistently by CRP, PCX, and serving."""

import torch
from torch.utils.data import Dataset


class DetectionSubset(Dataset):
    """Expose selected samples with their validated detector-class targets."""

    def __init__(self, dataset, indices, predicted_classes):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.predicted_classes = {
            int(index): tuple(int(class_id) for class_id in class_ids)
            for index, class_ids in predicted_classes.items()
        }
        self.class_names = dataset.class_names

        missing = [
            index for index in self.indices
            if index not in self.predicted_classes
        ]
        if missing:
            raise ValueError(
                f"Missing validated detector classes for {len(missing)} subset indices."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]
        image, _ = self.dataset[original_index]
        class_ids = torch.tensor(
            self.predicted_classes[original_index],
            dtype=torch.long,
        )
        targets = class_ids[:, None].expand(class_ids.shape[0], 2)
        return image, targets

    def reverse_normalization(self, data):
        return self.dataset.reverse_normalization(data)
