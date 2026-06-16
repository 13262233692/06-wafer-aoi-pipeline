from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from multiprocessing.shared_memory import SharedMemory
import numpy as np


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(processName)-18s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class FrameRingBuffer:
    """Lock-free ring buffer backed by shared memory for single-producer/single-consumer."""

    def __init__(
        self,
        name: str,
        frame_size_bytes: int,
        buffer_size: int,
        create: bool = False,
    ):
        self.name = name
        self.frame_size_bytes = frame_size_bytes
        self.buffer_size = buffer_size
        self.total_bytes = frame_size_bytes * buffer_size + 16

        self._shm: Optional[SharedMemory] = None
        self._metadata: Optional[np.ndarray] = None
        self._frames: Optional[np.ndarray] = None

        if create:
            try:
                SharedMemory(name=name, create=False).close()
                SharedMemory(name=name, create=False).unlink()
            except FileNotFoundError:
                pass
            self._shm = SharedMemory(name=name, create=True, size=self.total_bytes)
        else:
            self._shm = SharedMemory(name=name, create=False)

        self._metadata = np.ndarray(
            (4,), dtype=np.uint64, buffer=self._shm.buf[:32]
        )
        self._frames = np.ndarray(
            (buffer_size, frame_size_bytes),
            dtype=np.uint8,
            buffer=self._shm.buf[32:],
        )

        if create:
            self._metadata[0] = 0
            self._metadata[1] = 0
            self._metadata[2] = 0
            self._metadata[3] = 0

    @property
    def write_idx(self) -> int:
        return int(self._metadata[0])

    @property
    def read_idx(self) -> int:
        return int(self._metadata[1])

    @property
    def latest_frame_idx(self) -> int:
        return int(self._metadata[2])

    @property
    def dropped_frames(self) -> int:
        return int(self._metadata[3])

    def put(self, frame_data: np.ndarray, frame_id: int) -> bool:
        next_write = (self.write_idx + 1) % self.buffer_size
        if next_write == self.read_idx:
            self._metadata[3] += 1
            return False

        self._frames[self.write_idx, : frame_data.size] = frame_data.flatten()
        self._metadata[0] = next_write
        self._metadata[2] = frame_id
        return True

    def get(self) -> Optional[tuple]:
        if self.read_idx == self.write_idx:
            return None

        frame = self._frames[self.read_idx].copy()
        frame_id = self.latest_frame_idx
        self._metadata[1] = (self.read_idx + 1) % self.buffer_size
        return frame, frame_id

    def get_latest(self) -> Optional[np.ndarray]:
        if self.write_idx == 0:
            return None
        latest = (self.write_idx - 1) % self.buffer_size
        return self._frames[latest].copy()

    def close(self, unlink: bool = False):
        if self._shm is not None:
            self._shm.close()
            if unlink:
                try:
                    self._shm.unlink()
                except FileNotFoundError:
                    pass
        self._shm = None
        self._metadata = None
        self._frames = None

    def __del__(self):
        self.close()


@dataclass
class DetectedDefect:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": [float(x) for x in self.bbox],
        }


@dataclass
class InferenceResult:
    camera_id: int
    frame_id: int
    timestamp: float
    defects: List[DetectedDefect] = field(default_factory=list)
    inference_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "inference_time_ms": self.inference_time_ms,
            "defects": [d.to_dict() for d in self.defects],
        }


class ThroughputMeter:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._timestamps: List[float] = []

    def tick(self):
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) > self.window_size:
            self._timestamps.pop(0)

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / dt

    @property
    def latency_ms(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self._timestamps)):
            total += self._timestamps[i] - self._timestamps[i - 1]
        return (total / (len(self._timestamps) - 1)) * 1000.0
