import pytest
import torch

from src.yolo_class_mapping import (
    class_matched_box_or_fallback,
    class_name,
    matching_detection_index,
    resolve_display_class_names,
)


class _PersonCarDataset:
    class_names = ("person", "car")


def test_validated_detector_class_order_matches_crp_pcx_artifacts():
    names = resolve_display_class_names("yolov6s6", _PersonCarDataset())
    assert names == ("person", "car")
    assert class_name(names, 0) == "person"
    assert class_name(names, 1) == "car"


def test_prototype_box_is_selected_from_the_requested_class():
    # Detection zero is a car; detection one is a person.
    scores = torch.tensor([[0.05, 0.95], [0.90, 0.10]])
    assert matching_detection_index(scores, class_id=0) == 1
    assert matching_detection_index(scores, class_id=1) == 0


def test_missing_prototype_class_never_falls_back_to_another_class():
    scores = torch.tensor([[0.05, 0.95], [0.10, 0.90]])
    with pytest.raises(ValueError, match="class_id=0"):
        matching_detection_index(scores, class_id=0)


def test_unknown_class_id_is_rejected():
    with pytest.raises(ValueError, match="has no entry"):
        class_name(("car", "person"), 2)


def test_missing_live_detection_uses_same_class_stored_prototype():
    scores = torch.tensor([[0.05, 0.95]])  # live detection is car
    boxes = torch.tensor([[1.0, 2.0, 10.0, 20.0]])
    fallback = [30.0, 40.0, 80.0, 100.0]
    box, used_fallback = class_matched_box_or_fallback(
        scores, boxes, class_id=0, fallback_class_id=0, fallback_box=fallback
    )
    assert used_fallback is True
    assert box.tolist() == fallback


def test_cross_class_fallback_is_never_allowed():
    with pytest.raises(ValueError, match="cross-class"):
        class_matched_box_or_fallback(
            torch.tensor([[0.05, 0.95]]),
            torch.tensor([[1.0, 2.0, 10.0, 20.0]]),
            class_id=0,
            fallback_class_id=1,
            fallback_box=[30.0, 40.0, 80.0, 100.0],
        )
