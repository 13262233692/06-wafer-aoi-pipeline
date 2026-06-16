from __future__ import annotations

import time
from multiprocessing import Event
from typing import Dict, List, Optional, Tuple

import numpy as np

from wafer_aoi.config import CameraConfig
from wafer_aoi.utils import FrameRingBuffer, setup_logger

logger = setup_logger(__name__)


class SharedBufferManager:
    """Manages shared memory ring buffers for all camera streams."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._buffers: Dict[int, FrameRingBuffer] = {}
        self._stop_event = Event()

    def create_buffers(self):
        for cam_id in range(self.config.num_cameras):
            name = f"{self.config.shared_memory_name_prefix}{cam_id}"
            buf = FrameRingBuffer(
                name=name,
                frame_size_bytes=self.config.frame_size_bytes,
                buffer_size=self.config.buffer_size,
                create=True,
            )
            self._buffers[cam_id] = buf
            logger.info("Created shared buffer: %s (%d frames)", name, self.config.buffer_size)

    def attach_buffers(self):
        for cam_id in range(self.config.num_cameras):
            name = f"{self.config.shared_memory_name_prefix}{cam_id}"
            buf = FrameRingBuffer(
                name=name,
                frame_size_bytes=self.config.frame_size_bytes,
                buffer_size=self.config.buffer_size,
                create=False,
            )
            self._buffers[cam_id] = buf
            logger.info("Attached to shared buffer: %s", name)

    def get_buffer(self, cam_id: int) -> Optional[FrameRingBuffer]:
        return self._buffers.get(cam_id)

    def put_frame(self, cam_id: int, frame: np.ndarray, frame_id: int) -> bool:
        buf = self._buffers.get(cam_id)
        if buf is None:
            return False
        return buf.put(frame, frame_id)

    def get_frame(self, cam_id: int) -> Optional[Tuple[np.ndarray, int]]:
        buf = self._buffers.get(cam_id)
        if buf is None:
            return None
        result = buf.get()
        if result is None:
            return None
        raw_data, frame_id = result
        frame = raw_data[: self.config.frame_size_bytes].reshape(
            self.config.frame_height, self.config.frame_width, 3
        )
        return frame, frame_id

    def get_latest_frame(self, cam_id: int) -> Optional[np.ndarray]:
        buf = self._buffers.get(cam_id)
        if buf is None:
            return None
        raw = buf.get_latest()
        if raw is None:
            return None
        return raw[: self.config.frame_size_bytes].reshape(
            self.config.frame_height, self.config.frame_width, 3
        )

    def get_all_latest(self) -> Dict[int, Optional[np.ndarray]]:
        return {cid: self.get_latest_frame(cid) for cid in self._buffers}

    def stats(self) -> Dict:
        result = {}
        for cid, buf in self._buffers.items():
            result[cid] = {
                "dropped_frames": buf.dropped_frames,
                "write_idx": buf.write_idx,
                "read_idx": buf.read_idx,
                "latest_frame_id": buf.latest_frame_idx,
            }
        return result

    def close(self, unlink: bool = True):
        for buf in self._buffers.values():
            buf.close(unlink=unlink)
        self._buffers.clear()
        logger.info("Shared buffers closed (unlink=%s)", unlink)


class CameraFrameProducer:
    """Produces frames into a shared memory ring buffer (runs in camera process)."""

    def __init__(
        self,
        cam_id: int,
        config: CameraConfig,
    ):
        self.cam_id = cam_id
        self.config = config
        self._frame_id = 0
        self._buf: Optional[FrameRingBuffer] = None
        self._last_log = time.perf_counter()
        self._frames_since_log = 0

    def connect(self):
        name = f"{self.config.shared_memory_name_prefix}{self.cam_id}"
        self._buf = FrameRingBuffer(
            name=name,
            frame_size_bytes=self.config.frame_size_bytes,
            buffer_size=self.config.buffer_size,
            create=False,
        )
        logger.info("Camera %d producer connected to %s", self.cam_id, name)

    def submit(self, frame: np.ndarray):
        if self._buf is None:
            raise RuntimeError("Producer not connected")
        if frame.nbytes != self.config.frame_size_bytes:
            raise ValueError(
                f"Frame size mismatch: {frame.nbytes} != {self.config.frame_size_bytes}"
            )
        success = self._buf.put(frame, self._frame_id)
        self._frame_id += 1
        self._frames_since_log += 1

        now = time.perf_counter()
        if now - self._last_log >= 5.0:
            fps = self._frames_since_log / (now - self._last_log)
            logger.info(
                "Camera %d: %.1f FPS, dropped=%d",
                self.cam_id,
                fps,
                self._buf.dropped_frames,
            )
            self._frames_since_log = 0
            self._last_log = now
        return success

    def close(self):
        if self._buf is not None:
            self._buf.close(unlink=False)
            self._buf = None
