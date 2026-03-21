#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import os.path as osp
import json
import torch

ROOT = "/home/jovyan/FHHI-XAI/YOLOV6_fhhi_new"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
print(sys.path)

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Inference.', add_help=add_help)

    parser.add_argument('--weights', type=str,
                        default='/home/jovyan/FHHI-XAI/models/best_ckpt.pt',
                        help='model path(s) for inference.')
    parser.add_argument('--source', type=str,
                        default='/home/jovyan/FHHI-XAI/data/RAS_TrialData/images/train/set00_DJI_20250619113314_0001_W.JPG',
                        help='the source path, e.g. image-file/dir.')
    parser.add_argument('--webcam', action='store_true', help='whether to use webcam.')
    parser.add_argument('--webcam-addr', type=str, default='0', help='the web camera address, local camera or rtsp address.')
    parser.add_argument('--yaml', type=str,
                        default='/home/jovyan/FHHI-XAI/YOLOV6_fhhi_new/data/visdrone_twoclasses_first.yaml',
                        help='data yaml file.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='the image-size(h,w) in inference size.')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold for inference.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold for inference.')
    parser.add_argument('--max-det', type=int, default=1000, help='maximal inferences per image.')
    parser.add_argument('--device', default='0', help='device to run our model i.e. 0 or 0,1,2,3 or cpu.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt.')
    parser.add_argument('--not-save-img', action='store_true', help='do not save visuallized inference results.')
    parser.add_argument('--save-dir', type=str, help='directory to save predictions in. See --save-txt.')
    parser.add_argument('--view-img', action='store_true', help='show inference results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by classes, e.g. --classes 0, or --classes 0 2 3.')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS.')
    parser.add_argument('--project', default='/home/jovyan/FHHI-XAI/YOLOV6_fhhi_new/runs', help='save inference results to project/name.')
    parser.add_argument('--name', default='exp', help='save inference results to project/name.')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels.')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences.')
    parser.add_argument('--half', action='store_true', help='whether to use FP16 half-precision inference.')

    # ---- tiled inference switches (no str2bool, still True/False) ----
    parser.add_argument('--tile-infer', dest='tile_infer', action='store_true',
                        help='enable tiled inference')
    parser.add_argument('--no-tile-infer', dest='tile_infer', action='store_false',
                        help='disable tiled inference (use normal inference)')
    parser.set_defaults(tile_infer=True)  # DEFAULT TRUE

    parser.add_argument('--tile-size', type=int, default=640)
    parser.add_argument('--overlap-px', type=int, default=64)
    parser.add_argument('--overlap-ratio', type=float, default=0.10)

    parser.add_argument('--pad-to-tile', dest='pad_to_tile', action='store_true',
                        help='pad border tiles to tile-size')
    parser.add_argument('--no-pad-to-tile', dest='pad_to_tile', action='store_false',
                        help='do not pad border tiles')
    parser.set_defaults(pad_to_tile=True)  # DEFAULT TRUE

    args = parser.parse_args()
    LOGGER.info(args)
    return args


@torch.no_grad()
def run(weights=osp.join(ROOT, 'yolov6s6.pt'),
        source=osp.join(ROOT, 'data/images'),
        webcam=False,
        webcam_addr=0,
        yaml=None,
        img_size=640,
        conf_thres=0.4,
        iou_thres=0.45,
        max_det=1000,
        device='',
        save_txt=True,
        not_save_img=False,
        save_dir=None,
        view_img=True,
        classes=None,
        agnostic_nms=False,
        project=osp.join(ROOT, 'runs/inference'),
        name='exp',
        hide_labels=False,
        hide_conf=False,
        half=False,
        json_output="bboxes.json",
        tile_infer=True,
        tile_size=640,
        overlap_px=64,
        overlap_ratio=0.10,
        pad_to_tile=True):
    """
    Inference process (normal OR tiled).
    Saves detections to JSON (list of dicts).
    """
    # create save dir
    if save_dir is None:
        save_dir = osp.join(project, name)
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    inferer = Inferer(source, webcam, webcam_addr, weights, device, yaml, img_size, half)
    LOGGER.info("Running YOLOv6 inference...")

    if tile_infer:
        detections = inferer.infer_tiling(
            conf_thres, iou_thres, classes, agnostic_nms, max_det,
            save_dir, save_txt, not not_save_img, hide_labels, hide_conf, view_img,
            tile_size=tile_size,
            overlap_px=overlap_px,
            overlap_ratio=overlap_ratio,
            pad_to_tile=pad_to_tile,
            pad_value=114
        )
    else:
        detections = inferer.infer(
            conf_thres, iou_thres, classes, agnostic_nms, max_det,
            save_dir, save_txt, not not_save_img, hide_labels, hide_conf, view_img
        )

    if not detections:
        LOGGER.error("❌ No detections found. JSON file will not be created.")
        return

    with open(json_output, "w") as json_file:
        json.dump(detections, json_file, indent=4)

    LOGGER.info(f"✅ Bounding boxes saved to {json_output}")


def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
