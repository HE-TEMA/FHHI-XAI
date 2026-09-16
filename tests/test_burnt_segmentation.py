"""Local tests; no broker, MinIO or GPU required."""
import json
import logging

import numpy as np
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from PIL import Image

from src.datasets.burnt_dataset import BurntDataset
from src.pidnet_bank import (burnt_artifact_metadata, load_bank_indices,
                             validate_burnt_artifacts)


def make_dataset_root(tmp_path):
    for folder in ("RGB/train", "masks/train"):
        (tmp_path / folder).mkdir(parents=True)
    image = np.full((8, 12, 3), [40, 100, 200], dtype=np.uint8)
    mask = np.full((8, 12, 3), [216, 82, 24], dtype=np.uint8)
    mask[:, :6] = [0, 113, 188]
    mask[0, 0] = 0
    Image.fromarray(image).save(tmp_path / "RGB/train/aerial_image.jpg")
    Image.fromarray(mask).save(tmp_path / "masks/train/aerial-image_label_ground-truth_coco-panoptic.png")
    return tmp_path


class BurntSegmentationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)
        self.dataset_root = make_dataset_root(self.tmp_path)

    def test_mask_mapping_pairing_and_shared_preprocessing(self):
        dataset_root = self.dataset_root
        dataset = BurntDataset(dataset_root, crop_size=(8, 12))
        raw, label = dataset.load_sample(0)
        assert len(dataset) == 1
        assert label[0, 0] == -1
        assert label[1, 0] == 1
        assert label[1, 10] == 0
        image, target = dataset[0]
        torch.testing.assert_close(image, dataset.transform(raw))
        assert image.dtype == torch.float32 and target.dtype == torch.int64
        torch.testing.assert_close(dataset.reverse_augmentation(image),
                                   torch.from_numpy(raw.transpose(2, 0, 1).copy()).float())
        assert dataset.reverse_augmentation(image[None]).shape == (1, 3, 8, 12)
        bgr = BurntDataset(dataset_root, crop_size=(8, 12), channel_order="BGR")
        torch.testing.assert_close(bgr.reverse_augmentation(bgr.transform(raw)),
                                   dataset.reverse_augmentation(image))


    def test_coverage_and_ambiguous_pairs_fail_clearly(self):
        dataset_root = self.dataset_root
        with self.assertRaisesRegex(ValueError, "No burnt samples"):
            BurntDataset(dataset_root, min_burnt_coverage=0.5)
        Image.new("RGB", (12, 8)).save(dataset_root / "RGB/train/aerial-image.png")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            BurntDataset(dataset_root)


    def test_bank_indices_preserve_skipped_samples(self):
        tmp_path = self.tmp_path
        np.save(tmp_path / "sample_indices.npy", np.array([0, 2, 5]))
        assert load_bank_indices(tmp_path, 3, 6).tolist() == [0, 2, 5]
        with self.assertRaisesRegex(ValueError, "Invalid"):
            load_bank_indices(tmp_path, 3, 5)
        (tmp_path / "sample_indices.npy").unlink()
        with self.assertRaisesRegex(ValueError, "do not match"):
            load_bank_indices(tmp_path, 3, 6)
        assert load_bank_indices(tmp_path, 3, 3).tolist() == [0, 1, 2]

    def test_bank_builder_records_indices_when_a_row_is_nonfinite(self):
        from examples import build_crp_pcx_multilayer as builder

        def attribute(sample, *args, **kwargs):
            index = int(sample.item())
            relevance = torch.tensor([1., 2.]).reshape(1, 2, 1, 1)
            if index == 1:
                relevance[:] = float("nan")
            return SimpleNamespace(relevances={"layer": relevance})

        dataset = [(torch.tensor([float(i)]), torch.tensor(1)) for i in range(3)]
        with patch.dict(builder.ATTRIBUTORS, {"pidnet": lambda model: attribute}), \
             patch.dict(builder.CANONIZERS, {"pidnet": lambda: None}), \
             patch.dict(builder.COMPOSITES, {"pidnet": lambda **kwargs: None}):
            builder.build_pcx_banks(None, dataset, ["layer"], 1, "cpu", self.tmp_path)
        directory = self.tmp_path / "layer"
        assert np.load(directory / "sample_indices.npy").tolist() == [0, 2]
        bank = np.load(directory / "attributions.npy")
        assert bank.shape == (2, 2) and np.isfinite(bank).all()


    def test_metadata_rejects_preprocessing_or_checkpoint_drift(self):
        dataset_root, tmp_path = self.dataset_root, self.tmp_path
        dataset = BurntDataset(dataset_root, crop_size=(8, 12))
        ckpt = tmp_path / "checkpoint.pt"
        ckpt.write_bytes(b"checkpoint identity")
        crp, pcx = tmp_path / "crp", tmp_path / "pcx"
        for folder in (crp, pcx / "layer"):
            folder.mkdir(parents=True)
            (folder / "burnt_metadata.json").write_text(json.dumps(burnt_artifact_metadata(dataset, ckpt)))
        validate_burnt_artifacts(dataset, ckpt, crp, pcx, "layer")
        changed = BurntDataset(dataset_root, crop_size=(16, 24))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_burnt_artifacts(changed, ckpt, crp, pcx, "layer")
        ckpt.write_bytes(b"different checkpoint")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_burnt_artifacts(dataset, ckpt, crp, pcx, "layer")


    def test_service_dispatch_and_entity(self):
        dataset_root, tmp_path = self.dataset_root, self.tmp_path
        import matplotlib.pyplot as plt
        import src.explanator as module

        dataset = BurntDataset(dataset_root, crop_size=(8, 12))
        explainer = module.Explanator(str(tmp_path), logging.getLogger("burnt-test"))
        explainer._burnt_dataset = dataset
        explainer._burnt_model = torch.nn.Conv2d(3, 2, 1)
        with torch.no_grad():
            explainer._burnt_model.weight.zero_()
            explainer._burnt_model.bias.copy_(torch.tensor([0., 1.]))
        explainer.burnt_checkpoint = str(tmp_path / "checkpoint.pt")
        (tmp_path / "checkpoint.pt").write_bytes(b"test")
        validation_patch = patch.object(module, "validate_burnt_artifacts", lambda *args: None)
        validation_patch.start()
        self.addCleanup(validation_patch.stop)

        def render(*args, **kwargs):
            assert args[2] is dataset
            assert kwargs["precision"] == "fp32" and kwargs["use_rf"]
            assert kwargs["layer_name"] == "layer4.0.conv1"
            assert kwargs["outlier_percentile"] == 2.0
            fig = plt.figure()
            fig._n_refimgs_used = 4
            return fig

        render_patch = patch.object(module, "plot_pcx_explanations_pidnet", render)
        render_patch.start()
        self.addCleanup(render_patch.stop)
        raw, _ = dataset.load_sample(0)
        entity, images, filenames = explainer.explain(
            "BurntSegmentation", "use", "flight/image.jpg", raw,
            "bm", "uav", "flight", "urn:alert:1",
        )
        assert entity[0]["type"] == "BurntSegmentationExplanation"
        assert entity[0]["explainsEntity"]["object"] == "urn:ngsi-ld:tema:AUTH:TFA-06:BurntSegmentation"
        assert entity[0]["original_detection"]["value"]["class_categories"] == ["burnt"]
        assert entity[0]["explanation"]["value"]["n_refimgs"] == 4
        assert entity[0]["explanation"]["value"]["layer"] == "layer4.0.conv1"
        assert entity[0]["alert_ref"]["object"] == "urn:alert:1"
        assert filenames == ["tfa02/BurntSegmentation/flight/image.jpg"]
        assert len(images) == 1 and images[0].ndim == 3
        assert explainer.latest_kpi_log_paths

        with torch.no_grad():
            explainer._burnt_model.bias.copy_(torch.tensor([1., 0.]))
        assert explainer.explain("BurntSegmentation", "use", "empty.jpg", raw,
                                 "bm", "uav", "flight", "urn:alert:1") == (None, [], [])

    def test_preprocessing_matches_supplied_inference_wrapper(self):
        import cv2
        dataset = BurntDataset(self.dataset_root)
        raw, _ = dataset.load_sample(0)
        # Reproduce tools/eval.py: float32 /255 followed by float64 mean/std.
        resized = cv2.resize(raw, (1280, 720), interpolation=cv2.INTER_LINEAR)
        expected = torch.from_numpy(resized).permute(2, 0, 1).float() / 255
        mean = torch.tensor([0.4951, 0.509, 0.4397], dtype=torch.float64)[:, None, None]
        std = torch.tensor([0.176, 0.1487, 0.218], dtype=torch.float64)[:, None, None]
        expected = ((expected - mean) / std).float()
        torch.testing.assert_close(dataset.transform(raw), expected, atol=1e-6, rtol=1e-6)

if __name__ == "__main__":
    unittest.main()
