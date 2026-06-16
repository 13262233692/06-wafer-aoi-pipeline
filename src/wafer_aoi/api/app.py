from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2

from wafer_aoi.config import AppConfig

if TYPE_CHECKING:
    from wafer_aoi.orchestrator import PipelineOrchestrator


def create_app(config: AppConfig, orchestrator: "PipelineOrchestrator") -> FastAPI:
    """Create FastAPI control panel application."""
    app = FastAPI(
        title="Wafer AOI Pipeline",
        description="Semiconductor wafer defect detection control panel",
        version="0.1.0",
    )

    app.state.config = config
    app.state.orchestrator = orchestrator

    @app.get("/health")
    async def health() -> Dict:
        orch = app.state.orchestrator
        return {
            "status": "running" if orch.is_running else "stopped",
            "camera_processes_alive": orch.camera_processes_alive,
        }

    @app.get("/api/status")
    async def system_status() -> Dict:
        orch = app.state.orchestrator
        return {
            "running": orch.is_running,
            "cameras": orch.camera_stats(),
            "scheduler": orch.scheduler_stats(),
            "pipeline": orch.pipeline_stats(),
            "recent_results": orch.recent_results_summary(),
        }

    @app.get("/api/cameras")
    async def list_cameras() -> Dict:
        orch = app.state.orchestrator
        return {"cameras": orch.camera_stats()}

    @app.post("/api/cameras/{cam_id}/start")
    async def start_camera(cam_id: int) -> Dict:
        orch = app.state.orchestrator
        if cam_id >= config.camera.num_cameras:
            raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found")
        ok = orch.start_single_camera(cam_id)
        return {"success": ok, "camera_id": cam_id}

    @app.post("/api/cameras/{cam_id}/stop")
    async def stop_camera(cam_id: int) -> Dict:
        orch = app.state.orchestrator
        if cam_id >= config.camera.num_cameras:
            raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found")
        ok = orch.stop_single_camera(cam_id)
        return {"success": ok, "camera_id": cam_id}

    @app.get("/api/results/{cam_id}/latest")
    async def latest_result(cam_id: int) -> Dict:
        orch = app.state.orchestrator
        if cam_id >= config.camera.num_cameras:
            raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found")
        result = orch.get_latest_result(cam_id)
        if result is None:
            return {"camera_id": cam_id, "result": None}
        return {"camera_id": cam_id, "result": result.to_dict()}

    @app.get("/api/results/{cam_id}/stream")
    async def stream_results(cam_id: int):
        """SSE endpoint for streaming detection results (not implemented here fully)."""
        return JSONResponse(
            {"message": "SSE endpoint placeholder", "camera_id": cam_id}
        )

    @app.get("/api/frames/{cam_id}/latest")
    async def latest_frame(cam_id: int, annotated: bool = False) -> Response:
        """Return the latest camera frame as JPEG."""
        orch = app.state.orchestrator
        if cam_id >= config.camera.num_cameras:
            raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found")

        frame = orch.get_latest_frame(cam_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="No frame available")

        if annotated:
            result = orch.get_latest_result(cam_id)
            if result is not None:
                for defect in result.defects:
                    x1, y1, x2, y2 = [int(v) for v in defect.bbox]
                    color = (0, 0, 255) if defect.class_id == 0 else (
                        (0, 255, 0) if defect.class_id == 1 else (255, 0, 0)
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{defect.class_name}:{defect.confidence:.2f}"
                    cv2.putText(
                        frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.post("/api/pipeline/start")
    async def start_pipeline() -> Dict:
        orch = app.state.orchestrator
        orch.start_all()
        return {"success": True, "running": orch.is_running}

    @app.post("/api/pipeline/stop")
    async def stop_pipeline() -> Dict:
        orch = app.state.orchestrator
        orch.stop_all()
        return {"success": True, "running": orch.is_running}

    @app.get("/api/config")
    async def get_config() -> Dict:
        cfg = app.state.config
        return {
            "camera": {
                "num_cameras": cfg.camera.num_cameras,
                "frame_width": cfg.camera.frame_width,
                "frame_height": cfg.camera.frame_height,
                "fps": cfg.camera.fps,
                "pixel_format": cfg.camera.pixel_format,
            },
            "inference": {
                "input_width": cfg.inference.input_width,
                "input_height": cfg.inference.input_height,
                "max_batch_size": cfg.inference.max_batch_size,
                "num_classes": cfg.inference.num_classes,
                "class_names": cfg.inference.class_names,
                "conf_threshold": cfg.inference.conf_threshold,
                "nms_threshold": cfg.inference.nms_threshold,
                "num_streams": cfg.inference.num_streams,
            },
            "scheduler": {
                "batch_timeout_us": cfg.scheduler.batch_timeout_us,
                "max_batch_size": cfg.scheduler.max_batch_size,
            },
        }

    @app.post("/api/cameras/{cam_id}/reinspect")
    async def reinspect_camera(cam_id: int) -> Dict:
        """Force synchronous re-inspection of the latest frame from a camera.

        This endpoint safely invokes inference from the FastAPI request thread
        by using the process-wide CudaContextManager.  Without the explicit
        context push/pop inside orchestrator.rerun_inference, each request
        would implicitly create a new CUDA primary context on the request
        thread, leaking ~64 MB until the driver runs out of memory.
        """
        orch = app.state.orchestrator
        if cam_id >= config.camera.num_cameras:
            raise HTTPException(status_code=404, detail=f"Camera {cam_id} not found")

        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, orch.rerun_inference, cam_id, None)

        if result is None:
            raise HTTPException(status_code=503, detail="Inference pipeline not ready")
        return {"camera_id": cam_id, "result": result.to_dict()}

    @app.get("/api/gpu/diagnostic")
    async def gpu_diagnostic() -> Dict:
        """Return GPU memory and CUDA context diagnostics."""
        orch = app.state.orchestrator
        return orch.gpu_diagnostic()

    return app
