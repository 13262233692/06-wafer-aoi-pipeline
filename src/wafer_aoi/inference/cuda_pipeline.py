from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from wafer_aoi.config import InferenceConfig
from wafer_aoi.inference.cuda_context import CudaContextManager, get_cuda_context
from wafer_aoi.inference.trt_engine import TensorRTEngine
from wafer_aoi.utils import ThroughputMeter, setup_logger

logger = setup_logger(__name__)


@dataclass
class StreamSlot:
    """Per-stream resources: CUDA stream + pinned host memory + device buffers."""

    stream: object = None
    h_input: Optional[np.ndarray] = None
    h_output: Optional[np.ndarray] = None
    d_input: int = 0
    d_output: int = 0
    d_input_size: int = 0
    d_output_size: int = 0
    busy: bool = False
    meta: dict = field(default_factory=dict)


class CudaAsyncPipeline:
    """Multi-stream asynchronous CUDA pipeline: H2D copy -> inference -> D2H copy.

    Every CUDA operation (stream creation, host/device memory allocation,
    memcpy, inference submission) is explicitly wrapped in the singleton
    CudaContextManager's context_scope() so the correct CUcontext is always
    current on the calling thread. This eliminates the multi-threaded implicit
    context creation that caused the gradual ~64MB/context memory leak.
    """

    def __init__(self, engine: TensorRTEngine, config: InferenceConfig):
        self.engine = engine
        self.config = config
        self._cuda_ctx: CudaContextManager = get_cuda_context()
        self._slots: List[StreamSlot] = []
        self._initialized = False
        self._meter = ThroughputMeter(window_size=200)

    def initialize(self):
        if self._initialized:
            return

        if not self.engine.is_loaded:
            self.engine.load()

        num_streams = self.config.num_streams
        batch_size = self.config.max_batch_size
        input_size = self.engine.input_size * batch_size
        output_size = self.engine.output_size * batch_size

        logger.info(
            "Initializing CUDA pipeline: streams=%d, batch=%d, in=%d bytes, out=%d bytes",
            num_streams,
            batch_size,
            input_size * 4,
            output_size * 4,
        )

        if not self._cuda_ctx.is_available:
            logger.warning("PyCUDA not available, creating simulated slots")
            for i in range(num_streams):
                slot = StreamSlot(
                    h_input=np.zeros(
                        (batch_size, 3, self.config.input_height, self.config.input_width),
                        dtype=np.float32,
                    ),
                    h_output=np.zeros(output_size, dtype=np.float32),
                )
                self._slots.append(slot)
            self._initialized = True
            return

        self._cuda_ctx.initialize()

        with self._cuda_ctx.context_scope():
            cuda = self._cuda_ctx._cuda

            for i in range(num_streams):
                stream = cuda.Stream()

                h_input = cuda.pagelocked_empty(
                    (batch_size, 3, self.config.input_height, self.config.input_width),
                    dtype=np.float32,
                )
                h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)

                d_input_alloc = cuda.mem_alloc(input_size * 4)
                d_output_alloc = cuda.mem_alloc(output_size * 4)
                d_input_ptr = int(d_input_alloc)
                d_output_ptr = int(d_output_alloc)

                self._cuda_ctx.register_allocation(d_input_ptr, input_size * 4)
                self._cuda_ctx.register_allocation(d_output_ptr, output_size * 4)

                slot = StreamSlot(
                    stream=stream,
                    h_input=h_input,
                    h_output=h_output,
                    d_input=d_input_ptr,
                    d_output=d_output_ptr,
                    d_input_size=input_size * 4,
                    d_output_size=output_size * 4,
                )
                slot._d_input_alloc = d_input_alloc
                slot._d_output_alloc = d_output_alloc
                self._slots.append(slot)
                logger.debug("Stream slot %d created", i)

        info = self._cuda_ctx.get_memory_info()
        logger.info(
            "CUDA async pipeline initialized: %d streams, GPU used=%.1f/%.1f MB",
            num_streams,
            info.used_mb,
            info.total_mb,
        )
        self._initialized = True

    def acquire_slot(self, timeout: float = 0.001) -> Optional[int]:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            for idx, slot in enumerate(self._slots):
                if not slot.busy:
                    slot.busy = True
                    return idx
            time.sleep(0.0001)
        return None

    def submit(
        self,
        slot_idx: int,
        preprocessed_batch: np.ndarray,
        meta: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        slot = self._slots[slot_idx]
        batch_size = preprocessed_batch.shape[0]

        np.copyto(slot.h_input[:batch_size], preprocessed_batch)

        t0 = time.perf_counter()
        output = self.engine.infer_sync(
            slot.h_input[:batch_size],
            slot.d_input,
            slot.d_output,
            slot.h_output,
            stream=slot.stream,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self._meter.tick()
        result_meta = {
            "batch_size": batch_size,
            "infer_time_ms": dt_ms,
            "throughput_fps": self._meter.fps,
        }
        if meta:
            result_meta.update(meta)

        slot.busy = False
        slot.meta.clear()
        return output, result_meta

    @property
    def num_streams(self) -> int:
        return len(self._slots)

    @property
    def throughput_fps(self) -> float:
        return self._meter.fps

    @property
    def per_image_latency_ms(self) -> float:
        return self._meter.latency_ms

    def stats(self) -> dict:
        base = {
            "num_streams": self.num_streams,
            "throughput_fps": self._meter.fps,
            "per_batch_latency_ms": self._meter.latency_ms,
            "per_image_estimate_ms": self._meter.latency_ms / max(1, self.config.max_batch_size),
            "busy_slots": sum(1 for s in self._slots if s.busy),
        }
        if self._cuda_ctx.is_available:
            info = self._cuda_ctx.get_memory_info()
            base.update({
                "gpu_used_mb": info.used_mb,
                "gpu_free_mb": info.free_mb,
                "gpu_total_mb": info.total_mb,
            })
        return base

    def shutdown(self):
        if not self._cuda_ctx.is_available:
            self._slots.clear()
            self._initialized = False
            return

        if not self._initialized:
            return

        with self._cuda_ctx.context_scope():
            for slot in self._slots:
                try:
                    alloc_in = getattr(slot, "_d_input_alloc", None)
                    alloc_out = getattr(slot, "_d_output_alloc", None)
                    if alloc_in is not None:
                        self._cuda_ctx.unregister_allocation(slot.d_input)
                        alloc_in.free()
                    if alloc_out is not None:
                        self._cuda_ctx.unregister_allocation(slot.d_output)
                        alloc_out.free()
                except Exception as e:
                    logger.debug("Error freeing slot memory: %s", e)

        self._slots.clear()
        self._initialized = False
        info = self._cuda_ctx.get_memory_info()
        logger.info(
            "CUDA async pipeline shut down, GPU free=%.1f MB",
            info.free_mb,
        )
