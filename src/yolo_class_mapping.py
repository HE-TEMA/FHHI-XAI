"""Class identity invariants shared by YOLO explanation code."""

MODEL_CLASS_NAME_OVERRIDES = {
    # The validated two-class BRK checkpoint and its CRP/PCX artifacts use
    # this detector-output order. Keep detector IDs separate from any external
    # ontology IDs used by downstream entities.
    "yolov6s6": ("person", "car"),
}


def resolve_display_class_names(model_name, dataset):
    names = tuple(
        MODEL_CLASS_NAME_OVERRIDES.get(model_name)
        or getattr(dataset, "class_names", ())
    )
    if not names:
        raise ValueError(f"No class-name mapping is configured for model {model_name!r}.")
    if len(names) != len(set(names)):
        raise ValueError(f"Class-name mapping contains duplicates: {names!r}")
    return names


def class_name(class_names, class_id):
    """Resolve a detector class ID, rejecting unknown IDs instead of mislabelling it."""
    class_id = int(class_id)
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(
            f"Detector class_id={class_id} has no entry in class mapping {tuple(class_names)!r}."
        )
    return class_names[class_id]


def matching_detection_index(scores, class_id, occurrence=0):
    """Return the original detection index for a class-local occurrence."""
    matching = (scores.argmax(dim=1) == int(class_id)).nonzero(as_tuple=True)[0]
    if matching.numel() == 0:
        raise ValueError(f"No predicted detections found for class_id={class_id}.")
    if occurrence < 0 or occurrence >= matching.numel():
        raise IndexError(
            f"Detection occurrence {occurrence} is out of range for class_id={class_id}; "
            f"found {matching.numel()}."
        )
    return int(matching[occurrence].item())


def class_matched_box_or_fallback(scores, boxes, class_id, fallback_class_id, fallback_box):
    """Select a live box of the requested class, otherwise its same-class stored box."""
    class_id = int(class_id)
    fallback_class_id = int(fallback_class_id)
    if fallback_class_id != class_id:
        raise ValueError(
            f"Refusing cross-class prototype fallback: requested class_id={class_id}, "
            f"fallback class_id={fallback_class_id}."
        )

    matching = (scores.argmax(dim=1) == class_id).nonzero(as_tuple=True)[0]
    if matching.numel():
        return boxes[int(matching[0].item())], False

    fallback = boxes.new_tensor(fallback_box).flatten()
    if fallback.numel() != 4:
        raise ValueError(
            f"Stored fallback box for class_id={class_id} must contain four coordinates."
        )
    if not bool((fallback[2:] > fallback[:2]).all()):
        raise ValueError(f"Stored fallback box for class_id={class_id} is invalid: {fallback.tolist()}")
    return fallback, True
