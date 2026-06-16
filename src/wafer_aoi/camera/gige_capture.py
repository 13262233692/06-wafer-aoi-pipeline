from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import time
from typing import Optional

import numpy as np

from wafer_aoi.config import CameraConfig
from wafer_aoi.camera.shared_buffer import CameraFrameProducer
from wafer_aoi.utils import setup_logger

logger = setup_logger(__name__)


def _try_import_pypylon():
    try:
        from pypylon import pylon

        return pylon
    except ImportError:
        return None


class _SimulatedCamera:
    """Fallback simulated camera when pypylon is not available."""

    def __init__(self, cam_id: int, config: CameraConfig):
        self.cam_id = cam_id
        self.config = config
        self._running = False
        self._frame_count = 0

    def Open(self):
        self._running = True

    def StartGrabbing(self):
        self._running = True

    def IsGrabbing(self):
        return self._running

    def RetrieveResult(self, timeout_ms, grab_result_out):
        time.sleep(1.0 / self.config.fps)
        w, h = self.config.frame_width, self.config.frame_height
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        self._frame_count += 1
        return _SimulatedGrabResult(frame)

    def StopGrabbing(self):
        self._running = False

    def Close(self):
        self._running = False


class _SimulatedGrabResult:
    def __init__(self, frame: np.ndarray):
        self.GrabSucceeded = True
        self._frame = frame

    def GetArray(self):
        return self._frame

    def Release(self):
        pass


class GigECameraProcess(mp.Process):
    """Standalone process for capturing from a single GigE Vision camera."""

    def __init__(
        self,
        cam_id: int,
        config: CameraConfig,
        stop_event: mp.Event,
        camera_serial: Optional[str] = None,
    ):
        super().__init__(name=f"Camera-{cam_id}", daemon=True)
        self.cam_id = cam_id
        self.config = config
        self.stop_event = stop_event
        self.camera_serial = camera_serial
        self._log = setup_logger(f"camera.{cam_id}", config=CameraConfig)

    def run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger = setup_logger(f"camera.{self.cam_id}")
        logger.info("Starting camera capture process (id=%d)", self.cam_id)

        pylon = _try_import_pypylon()
        producer = CameraFrameProducer(self.cam_id, self.config)
        producer.connect()

        camera = None
        try:
            if pylon is not None:
                camera = self._open_pylon_camera(pylon, logger)
            else:
                logger.warning("pypylon not found, using simulated camera")
                camera = _SimulatedCamera(self.cam_id, self.config)
                camera.Open()

            camera.StartGrabbing()

            while not self.stop_event.is_set():
                if not camera.IsGrabbing():
                    logger.error("Camera %d stopped grabbing unexpectedly", self.cam_id)
                    break

                grab_result = camera.RetrieveResult(5000, None)
                if grab_result is None:
                    continue
                if not grab_result.GrabSucceeded:
                    grab_result.Release()
                    continue

                frame = grab_result.GetArray()
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                if frame.shape != (self.config.frame_height, self.config.frame_width, 3):
                    import cv2
                    frame = cv2.resize(
                        frame,
                        (self.config.frame_width, self.config.frame_height),
                    )
                frame = frame.astype(np.uint8, copy=False)
                producer.submit(frame)
                grab_result.Release()

        except Exception as e:
            logger.exception("Camera %d failed: %s", self.cam_id, e)
        finally:
            if camera is not None:
                try:
                    camera.StopGrabbing()
                    camera.Close()
                except Exception:
                    pass
            producer.close()
            logger.info("Camera %d process stopped", self.cam_id)

    def _open_pylon_camera(self, pylon, logger):
        tl_factory = pylon.TlFactory.GetInstance()
        devices = tl_factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("No GigE cameras found")

        if self.camera_serial:
            for dev in devices:
                if dev.GetSerialNumber() == self.camera_serial:
                    camera = pylon.InstantCamera(tl_factory.CreateDevice(dev))
                    break
            else:
                raise RuntimeError(f"Camera with serial {self.camera_serial} not found")
        else:
            if self.cam_id >= len(devices):
                logger.warning(
                    "cam_id=%d exceeds available cameras (%d), using simulated",
                    self.cam_id,
                    len(devices),
                )
                return _SimulatedCamera(self.cam_id, self.config)
            camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[self.cam_id]))

        camera.Open()
        try:
            camera.Width.SetValue(self.config.frame_width)
            camera.Height.SetValue(self.config.frame_height)
            camera.AcquisitionFrameRateEnable.SetValue(True)
            camera.AcquisitionFrameRate.SetValue(self.config.fps)
            camera.PixelFormat.SetValue(self.config.pixel_format)
        except Exception as e:
            logger.warning("Could not set all camera params: %s", e)

        logger.info("Opened GigE camera %d: %s", self.cam_id, camera.GetDeviceInfo().GetFriendlyName())
        return camera
