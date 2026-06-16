from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class CameraConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CAMERA_")

    num_cameras: int = 4
    frame_width: int = 1920
    frame_height: int = 1200
    fps: int = 60
    pixel_format: str = "BGR8"
    shared_memory_name_prefix: str = "wafer_camera_"
    buffer_size: int = 10

    @property
    def frame_size_bytes(self) -> int:
        if self.pixel_format == "BGR8":
            return self.frame_width * self.frame_height * 3
        return self.frame_width * self.frame_height

    @property
    def total_buffer_bytes(self) -> int:
        return self.frame_size_bytes * self.buffer_size


class InferenceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERENCE_")

    engine_path: str = "models/yolov8n_defect.engine"
    input_width: int = 640
    input_height: int = 640
    max_batch_size: int = 8
    num_classes: int = 4
    class_names: List[str] = Field(
        default_factory=lambda: ["scratch", "particle", "stain", "crack"]
    )
    conf_threshold: float = 0.5
    nms_threshold: float = 0.45
    num_streams: int = 4

    @property
    def input_shape(self) -> tuple:
        return (3, self.input_height, self.input_width)

    @property
    def input_size(self) -> int:
        return 3 * self.input_height * self.input_width


class SchedulerConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCHEDULER_")

    batch_timeout_us: int = 2000
    max_batch_size: int = 8


class APIConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8000


class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOGGING_")

    level: str = "INFO"


class AppConfig:
    def __init__(self, config_path: Optional[str] = None):
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}

        self.camera = CameraConfig(**raw.get("camera", {}))
        self.inference = InferenceConfig(**raw.get("inference", {}))
        self.scheduler = SchedulerConfig(**raw.get("scheduler", {}))
        self.api = APIConfig(**raw.get("api", {}))
        self.logging = LoggingConfig(**raw.get("logging", {}))

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AppConfig":
        return cls(config_path)
