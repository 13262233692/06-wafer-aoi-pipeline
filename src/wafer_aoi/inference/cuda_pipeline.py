from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from wafer_aoi.config import InferenceConfig
from wafer_aoi.inference.trt_engine import TensorRTEngine
from wafer_aoi.utils import setup_logger, ThroughputMeter

logger = setup_logger(__name__)


def _try_import_pycuda():
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        return cuda
    except ImportError:
        return None


@dataclass
class StreamSlot:
    """Per-stream resources: CUDA stream + pinned host memory + device buffers."""

    stream: object = None
    h_input: Optional[np.ndarray] = None
    h_output: Optional[np.ndarray] = None
    d_input: int = 0
    d_output: int = 0
    busy: bool = False
    meta: dict = field(default_factory=dict)


class CudaAsyncPipeline:
    """Multi-stream asynchronous CUDA pipeline: H2D copy -> inference -> D2H copy.

    Uses N independent CUDA streams to overlap data transfer with compute,
    achieving near-full GPU utilization under sustained load.
    """

    def __init__(self, engine: TensorRTEngine, config: InferenceConfig):
        self.engine = engine
        self.config = config
        self._cuda = _try_import_pycuda()
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

        if self._cuda is None:
            logger.warning("PyCUDA not available, creating simulated slots")
            for i in range(num_streams):
                slot = StreamSlot(
                    h_input=np.zeros((batch_size, 3, self.config.input_height, self.config.input_width), dtype=np.float32),
                    h_output=np.zeros(output_size, dtype=np.float32),
                )
                self._slots.append(slot)
            self._initialized = True
            return

        self._cuda.init_device(0)

        for i in range(num_streams):
            stream = self._cuda.Stream()

            h_input = self._cuda.pagelocked_empty(
                (batch_size, 3, self.config.input_height, self.config.input_width),
                dtype=np.float32,
            )
            h_output = self._cuda.pagelocked_empty(output_size, dtype=np.float32)

            d_input = self._cuda.mem_alloc(input_size * 4)
            d_output = self._cuda.mem_alloc(output_size * 4)

            slot = StreamSlot(
                stream=stream,
                h_input=h_input,
                h_output=h_output,
                d_input=int(d_input),
                d_output=int(d_output),
            )
            self._slots.append(slot)
            logger.debug("Stream slot %d created", i)

        self._initialized = True
        logger.info("CUDA async pipeline initialized with %d streams", num_streams)

    def acquire_slot(self, timeout: float = 0.001) -> Optional[int]:
        """Try to acquire a free stream slot. Returns slot index or None."""
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
        """Submit a batch to a stream slot and block until completion.

        For simpler scheduling we run synchronously per slot; the multi-stream
        design allows the scheduler to overlap work across slots by pipelining.

        Returns:
            (output_array, metadata_dict)
        """
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
        return {
            "num_streams": self.num_streams,
            "throughput_fps": self._meter.fps,
            "per_batch_latency_ms": self._meter.latency_ms,
            "per_image_estimate_ms": self._meter.latency_ms / max(1, self.config.max_batch_size),
            "busy_slots": sum(1 for s in self._slots if s.busy),
        }

    def shutdown(self):
        if self._cuda is None:
            self._slots.clear()
            self._initialized = False
            return

        for slot in self._slots:
            try:
                if slot.d_input:
                    self._cuda.DeviceAllocation(slot.d_input).free()
                if slot.d_output:
                    self._cuda.DeviceAllocation(slot.d_output).free()
            except Exception:
                pass
        self._slots.clear()
        self._initialized = False
        logger.info("CUDA async pipeline shut down")
