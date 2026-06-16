from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from wafer_aoi.utils import setup_logger

logger = setup_logger(__name__)


def _try_import_pycuda():
    try:
        import pycuda.driver as cuda
        return cuda
    except ImportError:
        return None


@dataclass
class GpuMemoryInfo:
    total_mb: float
    used_mb: float
    free_mb: float


class _ThreadLocalState(threading.local):
    def __init__(self):
        self.stack_depth: int = 0


class CudaContextManager:
    """Process-wide singleton that owns the CUDA device context and arbitrates access.

    Problem this solves:
      pycuda.autoinit only pushes the context on the importing thread. Any other
      thread (FastAPI worker, scheduler thread) that touches CUDA will trigger
      cuCtxGetCurrent() -> NULL -> driver implicitly creates a NEW primary
      context. Those implicit contexts leak ~64MB of device memory each and are
      never garbage-collected because the CUDA driver holds internal references.

    Solution:
      - One and only one CUcontext per process (created once, owned by singleton).
      - Every thread that needs CUDA must call push()/pop() via the `context()`
        context manager. We track per-thread push depth so nested calls do not
        corrupt the stack.
      - All device memory / streams / TensorRT engines are allocated inside this
        single context, so cross-thread sharing works as long as the caller holds
        the context active.
    """

    _instance: Optional["CudaContextManager"] = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, device_id: int = 0):
        if self._initialized:
            return

        self._cuda = _try_import_pycuda()
        self._device_id = device_id
        self._ctx: Any = None
        self._device: Any = None
        self._tls = _ThreadLocalState()
        self._global_lock = threading.RLock()
        self._allocations: Dict[int, int] = {}
        self._initialized = True

    @classmethod
    def instance(cls) -> "CudaContextManager":
        return cls()

    def initialize(self):
        """Initialize CUDA driver + device + context (idempotent)."""
        if self._cuda is None:
            logger.warning("PyCUDA not available; CudaContextManager operating in no-op mode")
            return

        with self._global_lock:
            if self._ctx is not None:
                return

            self._cuda.init()
            num_devices = self._cuda.Device.count()
            if num_devices == 0:
                raise RuntimeError("No CUDA-capable devices detected")
            if self._device_id >= num_devices:
                raise RuntimeError(
                    f"Device id {self._device_id} out of range (found {num_devices})"
                )

            self._device = self._cuda.Device(self._device_id)
            self._ctx = self._device.make_context()
            info = self._get_memory_info_unlocked()
            logger.info(
                "CUDA context created on device %d [%s], total=%.1f MB, free=%.1f MB",
                self._device_id,
                self._device.name(),
                info.total_mb,
                info.free_mb,
            )
            self._ctx.pop()

    @property
    def is_available(self) -> bool:
        return self._cuda is not None

    @property
    def is_initialized(self) -> bool:
        return self._ctx is not None

    @property
    def device(self):
        return self._device

    @property
    def context(self):
        return self._ctx

    @contextmanager
    def context_scope(self):
        """RAII guard: ensure the global CUDA context is current on this thread.

        Nested calls are reference-counted per thread; only the outermost
        __exit__ actually pops the context from the CPU thread stack.
        """
        if self._cuda is None or self._ctx is None:
            yield self
            return

        if self._tls.stack_depth == 0:
            self._ctx.push()
        self._tls.stack_depth += 1
        try:
            yield self
        finally:
            self._tls.stack_depth -= 1
            if self._tls.stack_depth == 0:
                self._ctx.pop()

    def execute(self, fn: Callable, *args, **kwargs):
        """Run a callable with the CUDA context active on this thread."""
        with self.context_scope():
            return fn(*args, **kwargs)

    def register_allocation(self, ptr: int, size_bytes: int):
        with self._global_lock:
            self._allocations[ptr] = size_bytes

    def unregister_allocation(self, ptr: int):
        with self._global_lock:
            self._allocations.pop(ptr, None)

    def get_memory_info(self) -> GpuMemoryInfo:
        if self._cuda is None:
            return GpuMemoryInfo(0.0, 0.0, 0.0)
        with self.context_scope():
            return self._get_memory_info_unlocked()

    def _get_memory_info_unlocked(self) -> GpuMemoryInfo:
        free, total = self._cuda.mem_get_info()
        total_mb = total / (1024 * 1024)
        free_mb = free / (1024 * 1024)
        return GpuMemoryInfo(
            total_mb=total_mb,
            used_mb=total_mb - free_mb,
            free_mb=free_mb,
        )

    def get_context_stack_depth(self) -> int:
        return self._tls.stack_depth

    def diagnostic_dump(self) -> Dict:
        info = self.get_memory_info()
        with self._global_lock:
            tracked_count = len(self._allocations)
            tracked_bytes = sum(self._allocations.values())
        return {
            "cuda_available": self.is_available,
            "context_initialized": self.is_initialized,
            "device_id": self._device_id,
            "device_name": self._device.name() if self._device else None,
            "thread_stack_depth": self._tls.stack_depth,
            "tracked_allocations_count": tracked_count,
            "tracked_allocations_mb": tracked_bytes / (1024 * 1024),
            "gpu_total_mb": info.total_mb,
            "gpu_used_mb": info.used_mb,
            "gpu_free_mb": info.free_mb,
        }

    def shutdown(self):
        if self._cuda is None or self._ctx is None:
            return

        with self._global_lock:
            if self._ctx is None:
                return

            with self.context_scope():
                try:
                    self._cuda.Context.synchronize()
                except Exception:
                    pass

            try:
                self._ctx.detach()
            except Exception as e:
                logger.debug("CUDA context detach: %s", e)
            self._ctx = None
            self._device = None
            self._allocations.clear()
            logger.info("CUDA context shut down")


def get_cuda_context() -> CudaContextManager:
    """Module-level accessor; returns the singleton."""
    return CudaContextManager.instance()
