from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys

import uvicorn

from wafer_aoi.api.app import create_app
from wafer_aoi.config import AppConfig
from wafer_aoi.orchestrator import PipelineOrchestrator
from wafer_aoi.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Wafer AOI Pipeline")
    parser.add_argument(
        "-c", "--config", type=str, default="configs/default.yaml", help="Config file path"
    )
    parser.add_argument("--headless", action="store_true", help="Run without API server")
    args = parser.parse_args()

    logger = setup_logger("wafer_aoi")
    logger.info("Starting Wafer AOI Pipeline")

    config = AppConfig.load(args.config)
    orchestrator = PipelineOrchestrator(config)
    orchestrator.initialize()
    orchestrator.start_all()

    if args.headless:
        logger.info("Running in headless mode. Press Ctrl+C to stop.")

        def _shutdown(signum, frame):
            logger.info("Received signal %s, shutting down...", signum)
            orchestrator.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        import time
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            orchestrator.shutdown()
    else:
        app = create_app(config, orchestrator)
        logger.info(
            "Starting API server on %s:%d", config.api.host, config.api.port
        )
        try:
            uvicorn.run(app, host=config.api.host, port=config.api.port, log_level="info")
        finally:
            orchestrator.shutdown()


if __name__ == "__main__":
    mp.freeze_support()
    main()
