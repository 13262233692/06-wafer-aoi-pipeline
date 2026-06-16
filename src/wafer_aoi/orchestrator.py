from __future__ import annotations

import multiprocessing as mp
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from wafer_aoi.config import AppConfig
from wafer_aoi.camera.gige_capture import GigECameraProcess
from wafer_aoi.camera.shared_buffer import SharedBufferManager
from wafer_aoi.inference.cuda_context import get_cuda_context
from wafer_aoi.inference.cuda_pipeline import CudaAsyncPipeline
from wafer_aoi.inference.trt_engine import TensorRTEngine
from wafer_aoi.inference.yolo_postprocess import preprocess_batch, yolo_decode
from wafer_aoi.pipeline.batch_scheduler import BatchScheduler
from wafer_aoi.utils import DetectedDefect, InferenceResult, setup_logger

logger = setup_logger(__name__)


class PipelineOrchestrator:
    """Top-level orchestrator that wires together cameras, buffers, scheduler, and API.

    Process / Thread layout:
    - 4x camera subprocess (GigECameraProcess) writing to shared memory
    - Main process:
        - Frame fetcher thread (1) reading shared memory -> scheduler input
        - Batch scheduler thread (1) batching + dispatching to CUDA streams
        - Result collector thread (1 optional)
        - FastAPI server thread (1) for control panel
    """

    def __init__(self, config: AppConfig):
        self.config = config

        self._buffer_manager: Optional[SharedBufferManager] = None
        self._camera_processes: Dict[int, GigECameraProcess] = {}
        self._camera_stop_events: Dict[int, mp.Event] = {}

        self._trt_engine: Optional[TensorRTEngine] = None
        self._cuda_pipeline: Optional[CudaAsyncPipeline] = None
        self._scheduler: Optional[BatchScheduler] = None

        self._fetcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._is_running = False
        self._recent_results: Dict[int, InferenceResult] = {}
        self._result_lock = threading.Lock()

    def initialize(self):
        logger.info("Initializing pipeline orchestrator")

        self._buffer_manager = SharedBufferManager(self.config.camera)
        self._buffer_manager.create_buffers()

        self._trt_engine = TensorRTEngine(self.config.inference)
        self._trt_engine.load()

        self._cuda_pipeline = CudaAsyncPipeline(self._trt_engine, self.config.inference)
        self._cuda_pipeline.initialize()

        self._scheduler = BatchScheduler(
            self._cuda_pipeline, self.config.inference, self.config.scheduler
        )
        for cid in range(self.config.camera.num_cameras):
            self._scheduler.register_camera(cid)

        logger.info("Orchestrator initialization complete")

    def _start_camera_process(self, cam_id: int):
        if cam_id in self._camera_processes and self._camera_processes[cam_id].is_alive():
            return True
        stop_evt = mp.Event()
        proc = GigECameraProcess(cam_id, self.config.camera, stop_evt)
        proc.start()
        self._camera_processes[cam_id] = proc
        self._camera_stop_events[cam_id] = stop_evt
        logger.info("Started camera process %d (PID=%d)", cam_id, proc.pid)
        return True

    def _stop_camera_process(self, cam_id: int) -> bool:
        stop_evt = self._camera_stop_events.get(cam_id)
        proc = self._camera_processes.get(cam_id)
        if stop_evt is not None:
            stop_evt.set()
        if proc is not None and proc.is_alive():
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        if cam_id in self._camera_processes:
            del self._camera_processes[cam_id]
        if cam_id in self._camera_stop_events:
            del self._camera_stop_events[cam_id]
        logger.info("Stopped camera process %d", cam_id)
        return True

    def start_single_camera(self, cam_id: int) -> bool:
        return self._start_camera_process(cam_id)

    def stop_single_camera(self, cam_id: int) -> bool:
        return self._stop_camera_process(cam_id)

    def _start_all_cameras(self):
        for cid in range(self.config.camera.num_cameras):
            self._start_camera_process(cid)

    def _stop_all_cameras(self):
        for cid in list(self._camera_processes.keys()):
            self._stop_camera_process(cid)

    def _frame_fetcher_loop(self):
        """Continuously read frames from shared memory and feed into scheduler."""
        logger.info("Frame fetcher thread started")
        assert self._buffer_manager is not None
        assert self._scheduler is not None

        next_frame_id = {cid: -1 for cid in range(self.config.camera.num_cameras)}

        while not self._stop_event.is_set():
            worked = False
            for cid in range(self.config.camera.num_cameras):
                buf = self._buffer_manager.get_buffer(cid)
                if buf is None:
                    continue
                if buf.write_idx == buf.read_idx:
                    continue

                item = buf.get()
                if item is None:
                    continue

                raw_data, shared_fid = item
                next_frame_id[cid] += 1
                fid = next_frame_id[cid]
                try:
                    frame = raw_data[: self.config.camera.frame_size_bytes].reshape(
                        self.config.camera.frame_height,
                        self.config.camera.frame_width,
                        3,
                    )
                    self._scheduler.submit_frame(cid, frame.copy(), fid)
                    worked = True
                except Exception as e:
                    logger.debug("Frame fetch error cam %d: %s", cid, e)

            with self._result_lock:
                for cid in range(self.config.camera.num_cameras):
                    r = self._scheduler.get_latest_result(cid)
                    if r is not None:
                        self._recent_results[cid] = r

            if not worked:
                time.sleep(0.0005)

        logger.info("Frame fetcher thread stopped")

    def start_all(self):
        if self._is_running:
            return
        if self._buffer_manager is None:
            self.initialize()

        self._stop_event.clear()
        assert self._scheduler is not None
        self._scheduler.start()

        self._start_all_cameras()

        self._fetcher_thread = threading.Thread(
            target=self._frame_fetcher_loop, name="FrameFetcher", daemon=True
        )
        self._fetcher_thread.start()

        self._is_running = True
        logger.info("Pipeline orchestrator started")

    def stop_all(self):
        if not self._is_running:
            return
        self._stop_event.set()
        self._stop_all_cameras()

        if self._scheduler is not None:
            self._scheduler.stop()

        if self._fetcher_thread is not None:
            self._fetcher_thread.join(timeout=3.0)
            self._fetcher_thread = None

        self._is_running = False
        logger.info("Pipeline orchestrator stopped")

    def shutdown(self):
        self.stop_all()
        if self._cuda_pipeline is not None:
            self._cuda_pipeline.shutdown()
        if self._trt_engine is not None:
            self._trt_engine.destroy()
        if self._buffer_manager is not None:
            self._buffer_manager.close(unlink=True)
        logger.info("Pipeline orchestrator shut down")

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def camera_processes_alive(self) -> Dict[int, bool]:
        return {cid: p.is_alive() for cid, p in self._camera_processes.items()}

    def camera_stats(self) -> Dict:
        if self._buffer_manager is None:
            return {}
        return self._buffer_manager.stats()

    def scheduler_stats(self) -> Dict:
        if self._scheduler is None:
            return {}
        return self._scheduler.stats()

    def pipeline_stats(self) -> Dict:
        if self._cuda_pipeline is None:
            return {}
        return self._cuda_pipeline.stats()

    def get_latest_result(self, cam_id: int) -> Optional[InferenceResult]:
        if self._scheduler is None:
            return None
        r = self._scheduler.get_latest_result(cam_id)
        if r is not None:
            with self._result_lock:
                self._recent_results[cam_id] = r
            return r
        with self._result_lock:
            return self._recent_results.get(cam_id)

    def get_latest_frame(self, cam_id: int) -> Optional[np.ndarray]:
        if self._buffer_manager is None:
            return None
        return self._buffer_manager.get_latest_frame(cam_id)

    def recent_results_summary(self) -> Dict:
        with self._result_lock:
            result = {}
            for cid, r in self._recent_results.items():
                result[cid] = {
                    "frame_id": r.frame_id,
                    "num_defects": len(r.defects),
                    "defect_classes": [d.class_name for d in r.defects],
                    "inference_time_ms": r.inference_time_ms,
                }
            return result

    def rerun_inference(self, cam_id: int, frame: Optional[np.ndarray] = None) -> Optional[InferenceResult]:
        """Force a re-inspection on a specific frame.

        This method runs synchronously from the calling thread (e.g. a FastAPI
        request handler).  It explicitly acquires the CUDA context to avoid
        leaking implicit per-thread contexts—the core fix for the ~64MB/call
        memory leak that crashed the line after ~500 wafers.
        """
        if self._cuda_pipeline is None or self._trt_engine is None:
            return None

        import time

        if frame is None:
            frame = self.get_latest_frame(cam_id)
        if frame is None:
            return None

        cuda_ctx = get_cuda_context()
        with cuda_ctx.context_scope():
            cfg = self.config.inference

            batch = preprocess_batch([frame], cfg.input_width, cfg.input_height)

            slot_idx = self._cuda_pipeline.acquire_slot(timeout=0.1)
            if slot_idx is None:
                return None

            try:
                output, meta = self._cuda_pipeline.submit(slot_idx, batch)
            finally:
                pass

        h, w = frame.shape[:2]
        scale = min(cfg.input_width / w, cfg.input_height / h)
        pad_x = (cfg.input_width - int(w * scale)) // 2
        pad_y = (cfg.input_height - int(h * scale)) // 2

        decoded = yolo_decode(output, cfg, scale=scale, pad=(pad_x, pad_y))
        defects = decoded[0] if decoded else []

        result = InferenceResult(
            camera_id=cam_id,
            frame_id=-1,
            timestamp=time.perf_counter(),
            defects=defects,
            inference_time_ms=meta.get("infer_time_ms", 0.0),
        )

        with self._result_lock:
            self._recent_results[cam_id] = result

        logger.info(
            "Manual re-inspection camera %d: %d defects, %.2f ms",
            cam_id,
            len(defects),
            result.inference_time_ms,
        )
        return result

    def gpu_diagnostic(self) -> Dict:
        cuda_ctx = get_cuda_context()
        return cuda_ctx.diagnostic_dump()
