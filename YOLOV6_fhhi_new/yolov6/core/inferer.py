#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import cv2
import time
import math
import torch
import numpy as np
import os.path as osp
import torchvision

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from collections import deque

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.data.datasets import LoadData
from yolov6.utils.nms import non_max_suppression


# =========================
# Helpers (match your tiler)
# =========================

def compute_stride(tile: int, overlap_px: int = 0, overlap_ratio: float = None) -> int:
    if overlap_ratio is not None:
        if not (0.0 <= overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be in [0,1).")
        overlap_px = int(round(tile * overlap_ratio))
    if overlap_px < 0 or overlap_px >= tile:
        raise ValueError("overlap_px must be in [0, tile-1].")
    stride = tile - overlap_px
    if stride <= 0:
        raise ValueError("Stride <= 0. Reduce overlap.")
    return stride


def generate_positions(length: int, tile: int, stride: int):
    # EXACT SAME AS YOUR TILER
    if length <= tile:
        return [0]
    positions = []
    pos = 0
    while True:
        positions.append(pos)
        if pos + tile >= length:
            break
        pos += stride
        if pos + tile > length:
            pos = length - tile
    return positions


def tile_ownership_region(x0: int, y0: int, tile: int, W: int, H: int, margin: int):
    """
    Same idea as your training tiler --assign_unique_by_center:
    interior tiles shrink by margin; border tiles don't shrink on the image boundary side.
    """
    is_left   = (x0 == 0)
    is_top    = (y0 == 0)
    is_right  = (x0 + tile >= W)
    is_bottom = (y0 + tile >= H)

    left_m   = 0 if is_left else margin
    top_m    = 0 if is_top else margin
    right_m  = 0 if is_right else margin
    bottom_m = 0 if is_bottom else margin

    ox1 = x0 + left_m
    oy1 = y0 + top_m
    ox2 = x0 + tile - right_m
    oy2 = y0 + tile - bottom_m

    ox1 = max(0, min(ox1, W))
    oy1 = max(0, min(oy1, H))
    ox2 = max(0, min(ox2, W))
    oy2 = max(0, min(oy2, H))

    return ox1, oy1, ox2, oy2


def global_nms_xyxy(det: torch.Tensor, iou_thres: float = 0.25, agnostic: bool = False) -> torch.Tensor:
    """
    NMS for decoded detections:
      det shape: [N, 6] = [x1, y1, x2, y2, conf, cls]
    """
    if det is None or det.numel() == 0 or len(det) == 0:
        return det

    boxes = det[:, :4]
    scores = det[:, 4]
    classes = det[:, 5]

    if agnostic:
        idxs = torch.zeros_like(classes)  # all same class
    else:
        idxs = classes

    keep = torchvision.ops.batched_nms(boxes, scores, idxs, iou_thres)
    return det[keep]



class Inferer:
    def __init__(self, source, webcam, webcam_addr, weights, device, yaml, img_size, half):

        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size
        self.half = half

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        # Half precision
        if self.half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        # Load data
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        self.files = LoadData(source, webcam, webcam_addr)
        self.source = source


    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

        LOGGER.info("Switch model to deploy modality.")

    def infer(self, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf, view_img=True):
        ''' Model Inference and results visualization '''
        vid_path, vid_writer, windows = None, None, []
        fps_calculator = CalcFPS()
        for img_src, img_path, vid_cap in tqdm(self.files):
            img, img_src = self.process_image(img_src, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            t1 = time.time()
            pred_results = self.model(img)
            det = non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            t2 = time.time()

            if self.webcam:
                save_path = osp.join(save_dir, self.webcam_addr)
                txt_path = osp.join(save_dir, self.webcam_addr)
            else:
                # Create output files in nested dirs that mirrors the structure of the images' dirs
                rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(self.source))
                save_path = osp.join(save_dir, rel_path, osp.basename(img_path))  # im.jpg
                txt_path = osp.join(save_dir, rel_path, 'labels', osp.splitext(osp.basename(img_path))[0])
                os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)

            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_ori = img_src.copy()

            # check image and font
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            self.font_check()

            if len(det):
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:
                        class_num = int(cls)  # integer class
                        label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')

                        self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=self.generate_colors(class_num, True))

                img_src = np.asarray(img_ori)

            # FPS counter
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()

            if self.files.type == 'video':
                self.draw_text(
                    img_src,
                    f"FPS: {avg_fps:0.1f}",
                    pos=(20, 20),
                    font_scale=1.0,
                    text_color=(204, 85, 17),
                    text_color_bg=(255, 255, 255),
                    font_thickness=2,
                )

            if view_img:
                if img_path not in windows:
                    windows.append(img_path)
                    cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
                cv2.imshow(str(img_path), img_src)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if self.files.type == 'image':
                    cv2.imwrite(save_path, img_src)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(img_src)



    @torch.no_grad()
    def infer_tiling(
        self,
        conf_thres, iou_thres, classes, agnostic_nms, max_det,
        save_dir, save_txt, save_img, hide_labels, hide_conf, view_img=True,
        tile_size=640, overlap_px=64, overlap_ratio=0.10, pad_to_tile=True, pad_value=114
    ):
        """
        Tile-based inference:
        - Split exactly like your tiler (generate_positions snapping)
        - Run model per tile
        - Convert boxes to global coords
        - Remove duplicates using OWNERSHIP REGION (same as your training tiler)
        - Optional safety-net: correct global NMS for [N,6]
        - Draw/save like infer()
        - Returns JSON-friendly detections
        """
        vid_path, vid_writer, windows = None, None, []
        fps_calculator = CalcFPS()

        results_for_json = []

        if tile_size is None:
            tile_size = int(self.img_size[0])

        stride = compute_stride(tile_size, overlap_px=overlap_px, overlap_ratio=overlap_ratio)
        overlap_used = tile_size - stride
        margin = max(0, overlap_used // 2)

        for img_src, img_path, vid_cap in tqdm(self.files):
            H, W = img_src.shape[:2]
            xs = generate_positions(W, tile_size, stride)
            ys = generate_positions(H, tile_size, stride)

            if self.webcam:
                save_path = osp.join(save_dir, self.webcam_addr)
                txt_path = osp.join(save_dir, self.webcam_addr)
            else:
                rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(self.source))
                save_path = osp.join(save_dir, rel_path, osp.basename(img_path))
                txt_path = osp.join(save_dir, rel_path, 'labels', osp.splitext(osp.basename(img_path))[0])
                os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)
                os.makedirs(osp.join(save_dir, rel_path, 'labels'), exist_ok=True)

            img_ori = img_src.copy()
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Use np.ascontiguousarray.'
            self.font_check()

            all_det = []

            t1 = time.time()

            for y0 in ys:
                for x0 in xs:
                    tile = img_src[y0:y0 + tile_size, x0:x0 + tile_size]
                    th, tw = tile.shape[:2]

                    if pad_to_tile and (th != tile_size or tw != tile_size):
                        padded = np.full((tile_size, tile_size, 3), pad_value, dtype=tile.dtype)
                        padded[:th, :tw] = tile
                        tile = padded

                    img_t, _ = self.process_image(tile, [tile_size, tile_size], self.stride, self.half)
                    img_t = img_t.to(self.device)
                    if len(img_t.shape) == 3:
                        img_t = img_t[None]

                    pred = self.model(img_t)
                    det_tile = non_max_suppression(
                        pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
                    )[0]

                    if det_tile is None or len(det_tile) == 0:
                        continue

                    # rescale model coords -> tile pixel coords
                    det_tile[:, :4] = self.rescale(img_t.shape[2:], det_tile[:, :4], (tile_size, tile_size, 3)).round()

                    # shift to GLOBAL coords
                    det_tile[:, [0, 2]] += x0
                    det_tile[:, [1, 3]] += y0

                    # clamp to image bounds
                    det_tile[:, 0].clamp_(0, W)
                    det_tile[:, 1].clamp_(0, H)
                    det_tile[:, 2].clamp_(0, W)
                    det_tile[:, 3].clamp_(0, H)

                    # ============================
                    # DEDUP LIKE YOUR TRAINING TILE
                    # ownership region by center
                    # ============================
                    ox1, oy1, ox2, oy2 = tile_ownership_region(x0, y0, tile_size, W, H, margin)
                    cx = 0.5 * (det_tile[:, 0] + det_tile[:, 2])
                    cy = 0.5 * (det_tile[:, 1] + det_tile[:, 3])
                    keep = (cx >= ox1) & (cx < ox2) & (cy >= oy1) & (cy < oy2)
                    det_tile = det_tile[keep]

                    if det_tile is None or len(det_tile) == 0:
                        continue

                    all_det.append(det_tile)

            if len(all_det):
                det = torch.cat(all_det, dim=0)

                # 1) Normal NMS (your normal iou_thres)
                det = global_nms_xyxy(det, iou_thres=iou_thres, agnostic=agnostic_nms)

                # 2) EXTRA merge-NMS for tile duplicates (LOWER threshold removes more duplicates)
                #    This is the important part.
                merge_iou = 0.1  # <- try 0.25, if still duplicates go 0.20; if over-suppress, go 0.30
                det = global_nms_xyxy(det, iou_thres=merge_iou, agnostic=agnostic_nms)

                # cap max_det
                if det is not None and len(det) > max_det:
                    det = det[det[:, 4].argsort(descending=True)[:max_det]]
            else:
                det = torch.empty((0, 6), device=self.device)

            t2 = time.time()

            gn = torch.tensor(img_ori.shape)[[1, 0, 1, 0]]  # whwh

            det_list = []
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    det_list.append({
                        "cls": int(cls),
                        "conf": float(conf),
                        "xyxy": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    })

                    if save_txt:
                        xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:
                        class_num = int(cls)
                        label = None if hide_labels else (
                            self.class_names[class_num] if hide_conf
                            else f'{self.class_names[class_num]} {conf:.2f}'
                        )
                        self.plot_box_and_label(
                            img_ori,
                            max(round(sum(img_ori.shape) / 2 * 0.003), 2),
                            xyxy,
                            label,
                            color=self.generate_colors(class_num, True)
                        )

            results_for_json.append({"image": str(img_path), "detections": det_list})

            img_vis = np.asarray(img_ori)

            fps_calculator.update(1.0 / max(t2 - t1, 1e-9))
            avg_fps = fps_calculator.accumulate()

            if self.files.type == 'video':
                self.draw_text(
                    img_vis,
                    f"FPS: {avg_fps:0.1f}",
                    pos=(20, 20),
                    font_scale=1.0,
                    text_color=(204, 85, 17),
                    text_color_bg=(255, 255, 255),
                    font_thickness=2,
                )

            if view_img:
                if img_path not in windows:
                    windows.append(img_path)
                    cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(img_path), img_vis.shape[1], img_vis.shape[0])
                cv2.imshow(str(img_path), img_vis)
                cv2.waitKey(1)

            if save_img:
                if self.files.type == 'image':
                    cv2.imwrite(save_path, img_vis)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, img_vis.shape[1], img_vis.shape[0]
                        save_path_mp4 = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer = cv2.VideoWriter(save_path_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(img_vis)

        return results_for_json


    # =======================
    # Original YOLOv6 helpers
    # =======================
    @staticmethod
    def process_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def font_check(font='/home/jovyan/FHHI-XAI/yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=True):
        # Correct class IDs (COCO labels)
        PERSON = 0
        VEHICLE_CLASSES = [1]   # car, motorcycle, bus, truck

        # Define fixed colors
        if i == PERSON:
            rgb = (255, 0, 0)  # red
        elif i in VEHICLE_CLASSES:
            rgb = (0, 255, 0)  # green
        else:
            rgb = (0, 255, 255)  # yellow for all other classes

    # Convert RGB → BGR for OpenCV if needed
        return (rgb[2], rgb[1], rgb[0]) if bgr else rgb


class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
