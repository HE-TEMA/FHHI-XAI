import math

import cv2
import numpy as np
import torch
from PIL import Image

from yolov6.data.data_augment import letterbox


def rescale_boxes(boxes, letterbox_shape, original_shape):
    """
    Rescale boxes from letterbox coordinates to original image coordinates.
    Matches inferer.py rescale method exactly.
    """

    if torch.is_tensor(boxes):
        boxes = boxes.detach().cpu().numpy()

    boxes = np.array(boxes).copy()

    ratio = min(letterbox_shape[0] / original_shape[0],
                letterbox_shape[1] / original_shape[1])
    padding = (
        (letterbox_shape[1] - original_shape[1] * ratio) / 2,
        (letterbox_shape[0] - original_shape[0] * ratio) / 2
    )

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0] = np.clip(boxes[:, 0], 0, original_shape[1])
    boxes[:, 1] = np.clip(boxes[:, 1], 0, original_shape[0])
    boxes[:, 2] = np.clip(boxes[:, 2], 0, original_shape[1])
    boxes[:, 3] = np.clip(boxes[:, 3], 0, original_shape[0])

    return boxes

def check_img_size(img_size, stride=32, floor=0):
    """Make sure image size is a multiple of stride s in each dimension.
    Exact copy from inferer.py's check_img_size method.
    """
    def make_divisible(x, divisor):
        return math.ceil(x / divisor) * divisor

    if isinstance(img_size, int):
        new_size = max(make_divisible(img_size, int(stride)), floor)
    elif isinstance(img_size, list):
        new_size = [max(make_divisible(x, int(stride)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    return new_size if isinstance(img_size, list) else [new_size, new_size]


def letterbox_transform(
    img,
    target_size=640,
    stride=32,
    half=False,
    auto=False,
    scaleup=True,
):
    """Apply the deterministic image preprocessing used by YOLOv6 training.

    This matches ``TrainValDataset.load_image`` and the non-random part of
    ``TrainValDataset.__getitem__`` followed by ``Trainer.prepro_data``:
    integer-truncated initial resizing, letterbox padding with value 114,
    BGR-to-RGB and HWC-to-CHW conversion, float conversion, and division by
    255.

    PIL images are interpreted as RGB. NumPy images are interpreted as BGR,
    matching images loaded by OpenCV in the YOLOv6 data loader.
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img)[:, :, ::-1]  # PIL RGB to BGR
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        raise TypeError(f"Expected a PIL image or NumPy array, got {type(img)!r}")

    img_size = check_img_size(target_size, stride=stride)

    # Match TrainValDataset.load_image exactly. This first resize matters:
    # YOLOv6 uses int() truncation here, whereas letterbox uses round().
    # Skipping this stage can change the resized content by one pixel.
    h0, w0 = img_np.shape[:2]
    initial_ratio = target_size / max(h0, w0)
    if initial_ratio != 1:
        img_np = cv2.resize(
            img_np,
            (int(w0 * initial_ratio), int(h0 * initial_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

    img_letterbox = letterbox(
        img_np,
        new_shape=img_size,
        stride=stride,
        auto=auto,
        scaleup=scaleup,
    )[0]

    # Match TrainValDataset: HWC to CHW and OpenCV BGR to model RGB.
    image = img_letterbox.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image)
    image = image.half() if half else image.float()
    image /= 255.0

    return image


class YOLOv6TrainPreprocess:
    """Callable transform configured like YOLOv6's 640-pixel train loader."""

    def __init__(self, target_size=640, stride=32, half=False):
        self.target_size = target_size
        self.stride = stride
        self.half = half

    def __call__(self, image):
        return letterbox_transform(
            image,
            target_size=self.target_size,
            stride=self.stride,
            half=self.half,
            auto=False,
            scaleup=True,
        )
