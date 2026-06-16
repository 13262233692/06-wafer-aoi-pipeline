from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from wafer_aoi.config import InferenceConfig
from wafer_aoi.inference.cuda_context import CudaContextManager, get_cuda_context
from wafer_aoi.utils import setup_logger

logger = setup_logger(__name__)


def _try_import_tensorrt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        return None


class TensorRTEngine:
    """TensorRT inference engine wrapper for YOLO-based detection models.

    All CUDA/TensorRT operations are guarded by CudaContextManager.context_scope()
    to guarantee that the correct CUcontext is current on the calling thread,
    preventing implicit context creation and the associated ~64MB/context leak.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self._trt = _try_import_tensorrt()
        self._cuda_ctx: CudaContextManager = get_cuda_context()

        self._runtime = None
        self._engine = None
        self._context = None
        self._input_bindings: List[int] = []
        self._output_bindings: List[int] = []
        self._input_shapes: List[tuple] = []
        self._output_shapes: List[tuple] = []
        self._stream = None
        self._loaded = False

    def load(self):
        if self._trt is None:
            logger.warning("TensorRT not available, running in simulation mode")
            self._loaded = True
            self._input_shapes = [
                (
                    self.config.max_batch_size,
                    3,
                    self.config.input_height,
                    self.config.input_width,
                )
            ]
            self._output_shapes = [
                (
                    self.config.max_batch_size,
                    self.config.num_classes + 4,
                    8400,
                )
            ]
            return

        if not self._cuda_ctx.is_available:
            raise RuntimeError("PyCUDA is required for TensorRT inference")

        self._cuda_ctx.initialize()

        engine_path = self.config.engine_path
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        logger.info("Loading TensorRT engine: %s", engine_path)

        with self._cuda_ctx.context_scope():
            trt_logger = self._trt.Logger(self._trt.Logger.WARNING)
            self._runtime = self._trt.Runtime(trt_logger)

            with open(engine_path, "rb") as f:
                engine_data = f.read()

            self._engine = self._runtime.deserialize_cuda_engine(engine_data)
            if self._engine is None:
                raise RuntimeError("Failed to deserialize TensorRT engine")

            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("Failed to create execution context")

            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                shape = tuple(self._engine.get_tensor_shape(name))
                if shape and shape[0] == -1:
                    shape = (self.config.max_batch_size,) + shape[1:]

                if mode == self._trt.TensorIOMode.INPUT:
                    self._context.set_input_shape(name, shape)
                    self._input_shapes.append(shape)
                    self._input_bindings.append(i)
                    logger.info("Input tensor %d: %s shape=%s", i, name, shape)
                else:
                    self._output_shapes.append(shape)
                    self._output_bindings.append(i)
                    logger.info("Output tensor %d: %s shape=%s", i, name, shape)

            self._stream = self._cuda_ctx._cuda.Stream()

        self._loaded = True
        logger.info("TensorRT engine loaded successfully")

    @property
    def input_size(self) -> int:
        if not self._input_shapes:
            return 0
        s = self._input_shapes[0]
        return int(np.prod(s[1:]))

    @property
    def output_size(self) -> int:
        if not self._output_shapes:
            return 0
        s = self._output_shapes[0]
        return int(np.prod(s[1:]))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def infer_sync(
        self,
        input_batch: np.ndarray,
        d_input: int,
        d_output: int,
        h_output: np.ndarray,
        stream=None,
    ) -> np.ndarray:
        """Run inference with pre-allocated device buffers.

        All CUDA operations are wrapped in context_scope() to guarantee the
        correct CUcontext is active on the calling thread.
        """
        if not self._loaded:
            return self._simulate_inference(input_batch)

        batch_size = input_batch.shape[0]

        with self._cuda_ctx.context_scope():
            use_stream = stream if stream is not None else self._stream
            cuda = self._cuda_ctx._cuda

            cuda.memcpy_htod_async(d_input, input_batch, use_stream)

            bindings = [d_input, d_output]
            if self._context is not None:
                if self._input_shapes and self._input_shapes[0][0] != batch_size:
                    name = self._engine.get_tensor_name(self._input_bindings[0])
                    self._context.set_input_shape(
                        name, (batch_size,) + self._input_shapes[0][1:]
                    )
                self._context.execute_async_v2(
                    bindings=bindings, stream_handle=use_stream.handle
                )

            cuda.memcpy_dtoh_async(h_output, d_output, use_stream)
            use_stream.synchronize()

        actual_output_shape = (batch_size,) + self._output_shapes[0][1:]
        return h_output[: int(np.prod(actual_output_shape))].reshape(actual_output_shape)

    def _simulate_inference(self, input_batch: np.ndarray) -> np.ndarray:
        batch_size = input_batch.shape[0]
        output = np.zeros(
            (batch_size, self.config.num_classes + 4, 8400), dtype=np.float32
        )
        return output

    def destroy(self):
        if not self._cuda_ctx.is_available or self._context is None:
            self._loaded = False
            return

        try:
            with self._cuda_ctx.context_scope():
                if self._context is not None:
                    del self._context
                    self._context = None
                if self._engine is not None:
                    del self._engine
                    self._engine = None
                if self._runtime is not None:
                    del self._runtime
                    self._runtime = None
        except Exception as e:
            logger.debug("Exception during TRT engine destroy: %s", e)
            self._context = None
            self._engine = None
            self._runtime = None

        self._input_bindings.clear()
        self._output_bindings.clear()
        self._input_shapes.clear()
        self._output_shapes.clear()
        self._stream = None
        self._loaded = False
        logger.info("TensorRT engine destroyed")
