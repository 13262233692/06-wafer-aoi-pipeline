from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Dict, List, Optional

import cv2
import numpy as np

from wafer_aoi.ohem.ohem_filter import OhemVerdict
from wafer_aoi.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class HardExamplePackage:
    camera_id: int
    frame_id: int
    timestamp: float
    verdict: OhemVerdict
    frame: np.ndarray
    context_bbox: tuple


class HardExampleArchiver:
    """Asynchronously persists hard examples to the local training set directory.

    For each hard example the archiver writes:
      - A cropped image of the detection region (with context padding).
      - A JSON sidecar containing full metadata (class, confidence, bbox,
        OHEM similarity, suppression status, camera/frame ID, timestamp).

    Directory layout:
        <output_dir>/
            images/
                <camera_id>/
                    <timestamp>_<uuid>.png
            labels/
                <camera_id>/
                    <timestamp>_<uuid>.json

    A background thread drains the internal queue so the inference hot path
    is never blocked by disk I/O.
    """

    def __init__(
        self,
        output_dir: str = "data/hard_examples",
        queue_size: int = 512,
        context_padding_ratio: float = 0.25,
    ):
        self.output_dir = output_dir
        self.context_padding_ratio = context_padding_ratio

        self._queue: Queue = Queue(maxsize=queue_size)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._total_archived = 0
        self._total_dropped = 0

    def start(self):
        if self._worker_thread is not None:
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="HardExampleArchiver", daemon=True
        )
        self._worker_thread.start()
        logger.info("Hard example archiver started (dir=%s)", self.output_dir)

    def stop(self):
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None
        logger.info(
            "Hard example archiver stopped (archived=%d, dropped=%d)",
            self._total_archived,
            self._total_dropped,
        )

    def submit(
        self,
        camera_id: int,
        frame_id: int,
        timestamp: float,
        verdict: OhemVerdict,
        frame: np.ndarray,
        context_bbox: Optional[tuple] = None,
    ) -> bool:
        """Enqueue a hard example for async archival. Non-blocking."""
        bbox = context_bbox or verdict.defect.bbox
        package = HardExamplePackage(
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=timestamp,
            verdict=verdict,
            frame=frame,
            context_bbox=bbox,
        )
        try:
            self._queue.put_nowait(package)
            return True
        except Full:
            self._total_dropped += 1
            return False

    def submit_batch(
        self,
        camera_id: int,
        frame_id: int,
        timestamp: float,
        verdicts: List[OhemVerdict],
        frame: np.ndarray,
        context_bboxes: Optional[List[tuple]] = None,
    ) -> int:
        """Enqueue multiple hard examples. Returns count actually enqueued."""
        enqueued = 0
        for i, v in enumerate(verdicts):
            bbox = context_bboxes[i] if context_bboxes else None
            if self.submit(camera_id, frame_id, timestamp, v, frame, bbox):
                enqueued += 1
        return enqueued

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                package = self._queue.get(timeout=0.05)
            except Empty:
                continue

            try:
                self._archive_package(package)
                self._total_archived += 1
            except Exception as e:
                logger.debug("Failed to archive hard example: %s", e)
            finally:
                self._queue.task_done()

        while True:
            try:
                package = self._queue.get_nowait()
                self._archive_package(package)
                self._total_archived += 1
            except Empty:
                break
            except Exception:
                break

    def _archive_package(self, package: HardExamplePackage):
        cam_dir_img = Path(self.output_dir) / "images" / str(package.camera_id)
        cam_dir_lbl = Path(self.output_dir) / "labels" / str(package.camera_id)
        cam_dir_img.mkdir(parents=True, exist_ok=True)
        cam_dir_lbl.mkdir(parents=True, exist_ok=True)

        ts = int(package.timestamp * 1000)
        uid = uuid.uuid4().hex[:8]
        stem = f"{ts}_{uid}"

        crop = self._crop_with_context(package.frame, package.context_bbox)
        if crop is None:
            return

        img_path = cam_dir_img / f"{stem}.png"
        cv2.imwrite(str(img_path), crop)

        d = package.verdict.defect
        label = {
            "image": str(img_path),
            "camera_id": package.camera_id,
            "frame_id": package.frame_id,
            "timestamp": package.timestamp,
            "class_id": d.class_id,
            "class_name": d.class_name,
            "confidence": d.confidence,
            "bbox_xyxy": list(d.bbox),
            "context_bbox_xyxy": list(package.context_bbox),
            "ohem_suppressed": package.verdict.suppressed,
            "ohem_max_similarity": package.verdict.max_similarity,
            "ohem_matched_fp_ids": package.verdict.matched_fp_ids,
        }

        lbl_path = cam_dir_lbl / f"{stem}.json"
        with open(lbl_path, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)

        logger.debug(
            "Archived hard example: cam=%d cls=%s conf=%.3f sup=%s -> %s",
            package.camera_id,
            d.class_name,
            d.confidence,
            package.verdict.suppressed,
            img_path,
        )

    def _crop_with_context(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad_w = int(bw * self.context_padding_ratio)
        pad_h = int(bh * self.context_padding_ratio)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def stats(self) -> Dict:
        return {
            "total_archived": self._total_archived,
            "total_dropped": self._total_dropped,
            "queue_size": self._queue.qsize(),
            "queue_capacity": self._queue.maxsize,
            "output_dir": self.output_dir,
        }
