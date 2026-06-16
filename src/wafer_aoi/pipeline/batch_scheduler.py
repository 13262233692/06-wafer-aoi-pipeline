from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from wafer_aoi.config import InferenceConfig, SchedulerConfig
from wafer_aoi.inference.cuda_pipeline import CudaAsyncPipeline
from wafer_aoi.inference.yolo_postprocess import preprocess_batch, yolo_decode
from wafer_aoi.utils import DetectedDefect, InferenceResult, ThroughputMeter, setup_logger

logger = setup_logger(__name__)


@dataclass
class PendingFrame:
    camera_id: int
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass
class BatchJob:
    slot_idx: int
    frames: List[PendingFrame]
    preprocessed: np.ndarray
    scales: List[float]
    pads: List[tuple]


class BatchScheduler:
    """Collects frames from multiple camera streams, batches them, and dispatches to CUDA streams.

    Features:
    - Dynamic batching: accumulate up to max_batch_size within batch_timeout_us
    - Multi-stream dispatch: round-robin assign batches to free CUDA stream slots
    - Result dispatch: per-camera result queues with bounded depth
    """

    def __init__(
        self,
        pipeline: CudaAsyncPipeline,
        inf_config: InferenceConfig,
        sched_config: SchedulerConfig,
    ):
        self.pipeline = pipeline
        self.inf_config = inf_config
        self.sched_config = sched_config

        self._input_queues: Dict[int, Deque[PendingFrame]] = {}
        self._result_queues: Dict[int, queue.Queue] = {}
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._dispatch_thread: Optional[threading.Thread] = None

        self._meter = ThroughputMeter(window_size=500)
        self._last_stats_log = time.perf_counter()
        self._frames_processed = 0

    def register_camera(self, cam_id: int, result_queue_depth: int = 32):
        with self._lock:
            self._input_queues[cam_id] = deque(maxlen=result_queue_depth)
            self._result_queues[cam_id] = queue.Queue(maxsize=result_queue_depth)
            logger.info("Registered camera %d in scheduler", cam_id)

    def submit_frame(self, cam_id: int, frame: np.ndarray, frame_id: int):
        if cam_id not in self._input_queues:
            self.register_camera(cam_id)
        pending = PendingFrame(
            camera_id=cam_id,
            frame_id=frame_id,
            timestamp=time.perf_counter(),
            frame=frame,
        )
        try:
            self._input_queues[cam_id].append(pending)
        except Exception:
            pass

    def get_result(self, cam_id: int, timeout: float = 0.01) -> Optional[InferenceResult]:
        q = self._result_queues.get(cam_id)
        if q is None:
            return None
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_latest_result(self, cam_id: int) -> Optional[InferenceResult]:
        q = self._result_queues.get(cam_id)
        if q is None:
            return None
        latest = None
        while True:
            try:
                latest = q.get_nowait()
            except queue.Empty:
                break
        return latest

    def start(self):
        if self._dispatch_thread is not None:
            return
        self._stop_event.clear()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop, name="BatchScheduler", daemon=True
        )
        self._dispatch_thread.start()
        logger.info("Batch scheduler started")

    def stop(self):
        self._stop_event.set()
        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=2.0)
            self._dispatch_thread = None
        logger.info("Batch scheduler stopped")

    def stats(self) -> dict:
        with self._lock:
            pending = sum(len(q) for q in self._input_queues.values())
            results = sum(q.qsize() for q in self._result_queues.values())
        return {
            "pending_frames": pending,
            "result_queue_depth": results,
            "throughput_fps": self._meter.fps,
            "per_frame_latency_ms": self._meter.latency_ms,
            "frames_processed_total": self._frames_processed,
            **self.pipeline.stats(),
        }

    def _collect_batch(self) -> Optional[List[PendingFrame]]:
        batch: List[PendingFrame] = []
        max_batch = self.sched_config.max_batch_size
        deadline = time.perf_counter() + self.sched_config.batch_timeout_us / 1e6

        while len(batch) < max_batch and time.perf_counter() < deadline:
            got_one = False
            with self._lock:
                cam_ids = list(self._input_queues.keys())
            for cid in cam_ids:
                q = self._input_queues.get(cid)
                if q is None:
                    continue
                try:
                    pending = q.popleft()
                    batch.append(pending)
                    got_one = True
                    if len(batch) >= max_batch:
                        break
                except IndexError:
                    continue
            if not got_one and not batch:
                time.sleep(0.0001)
            else:
                break

        return batch if batch else None

    def _dispatch_loop(self):
        while not self._stop_event.is_set():
            batch = self._collect_batch()
            if batch is None:
                continue

            slot_idx = self.pipeline.acquire_slot(timeout=0.005)
            if slot_idx is None:
                for pf in reversed(batch):
                    try:
                        self._input_queues[pf.camera_id].appendleft(pf)
                    except Exception:
                        pass
                time.sleep(0.0005)
                continue

            job = self._prepare_batch(batch, slot_idx)
            if job is None:
                self.pipeline._slots[slot_idx].busy = False
                continue

            self._run_job(job)

    def _prepare_batch(self, frames: List[PendingFrame], slot_idx: int) -> Optional[BatchJob]:
        if not frames:
            return None
        iw, ih = self.inf_config.input_width, self.inf_config.input_height

        raw_frames = [pf.frame for pf in frames]
        scales = []
        pads = []
        for f in raw_frames:
            h, w = f.shape[:2]
            scale = min(iw / w, ih / h)
            nw, nh = int(w * scale), int(h * scale)
            pad_x = (iw - nw) // 2
            pad_y = (ih - nh) // 2
            scales.append(scale)
            pads.append((pad_x, pad_y))

        preprocessed = preprocess_batch(raw_frames, iw, ih)
        return BatchJob(
            slot_idx=slot_idx,
            frames=frames,
            preprocessed=preprocessed,
            scales=scales,
            pads=pads,
        )

    def _run_job(self, job: BatchJob):
        try:
            output, meta = self.pipeline.submit(
                job.slot_idx,
                job.preprocessed,
                meta={"num_frames": len(job.frames)},
            )
            infer_ms = meta.get("infer_time_ms", 0.0)

            all_defects = []
            for i, pf in enumerate(job.frames):
                single_out = output[i : i + 1]
                decoded = yolo_decode(
                    single_out,
                    self.inf_config,
                    scale=job.scales[i],
                    pad=job.pads[i],
                )
                all_defects.append(decoded[0] if decoded else [])

            now = time.perf_counter()
            for i, pf in enumerate(job.frames):
                result = InferenceResult(
                    camera_id=pf.camera_id,
                    frame_id=pf.frame_id,
                    timestamp=now,
                    defects=all_defects[i],
                    inference_time_ms=infer_ms / max(1, len(job.frames)),
                )
                self._meter.tick()
                self._frames_processed += 1
                try:
                    q = self._result_queues.get(pf.camera_id)
                    if q is not None:
                        if q.full():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                pass
                        q.put_nowait(result)
                except Exception:
                    pass

            if now - self._last_stats_log >= 10.0:
                s = self.stats()
                logger.info(
                    "Scheduler: %.1f FPS, latency=%.2f ms/img, pending=%d, busy_slots=%d/%d",
                    s["throughput_fps"],
                    s["per_frame_latency_ms"],
                    s["pending_frames"],
                    s["busy_slots"],
                    s["num_streams"],
                )
                self._last_stats_log = now
        except Exception as e:
            logger.exception("Batch job failed: %s", e)
