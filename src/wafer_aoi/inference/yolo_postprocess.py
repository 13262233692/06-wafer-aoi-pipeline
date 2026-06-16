from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from wafer_aoi.config import InferenceConfig
from wafer_aoi.utils import DetectedDefect


def preprocess_batch(
    frames: List[np.ndarray],
    input_w: int,
    input_h: int,
) -> np.ndarray:
    """Preprocess a list of BGR frames into an NCHW float32 batch.

    Steps: letterbox resize -> BGR to RGB -> HWC to CHW -> normalize to [0,1].
    Uses vectorized operations for speed.
    """
    batch = np.zeros((len(frames), 3, input_h, input_w), dtype=np.float32)
    for i, frame in enumerate(frames):
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = min(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = rgb

        batch[i] = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
    return batch


def yolo_decode(
    output: np.ndarray,
    config: InferenceConfig,
    scale: float = 1.0,
    pad: Tuple[int, int] = (0, 0),
) -> List[List[DetectedDefect]]:
    """Decode YOLOv8 output (B, C+4, 8400) -> per-image list of Defect detections.

    Args:
        output: Raw model output with shape (batch, num_classes+4, num_anchors)
        config: Inference configuration
        scale: Scale factor from letterbox resize
        pad: (pad_x, pad_y) from letterbox

    Returns:
        List (length = batch_size) of DetectedDefect lists
    """
    batch_size = output.shape[0]
    results: List[List[DetectedDefect]] = [[] for _ in range(batch_size)]

    for b in range(batch_size):
        pred = output[b]
        boxes_xywh = pred[:4, :]
        cls_scores = pred[4:, :]

        class_ids = np.argmax(cls_scores, axis=0)
        max_scores = np.max(cls_scores, axis=0)
        keep = max_scores > config.conf_threshold

        if not np.any(keep):
            continue

        cx = boxes_xywh[0, keep]
        cy = boxes_xywh[1, keep]
        w = boxes_xywh[2, keep]
        h = boxes_xywh[3, keep]

        x1 = (cx - w / 2 - pad[0]) / scale
        y1 = (cy - h / 2 - pad[1]) / scale
        x2 = (cx + w / 2 - pad[0]) / scale
        y2 = (cy + h / 2 - pad[1]) / scale

        scores = max_scores[keep]
        cids = class_ids[keep]

        nms_keep = nms(x1, y1, x2, y2, scores, config.nms_threshold)

        for idx in nms_keep:
            cid = int(cids[idx])
            results[b].append(
                DetectedDefect(
                    class_id=cid,
                    class_name=config.class_names[cid] if cid < len(config.class_names) else str(cid),
                    confidence=float(scores[idx]),
                    bbox=(float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])),
                )
            )
    return results


def nms(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> List[int]:
    """Vectorized non-maximum suppression."""
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
