from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from wafer_aoi.utils import setup_logger

logger = setup_logger(__name__)


def _try_import_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


class FaissFalsePositiveIndex:
    """Lightweight FAISS IndexFlatIP for cosine-similarity retrieval against
    known false-positive feature vectors.

    Because all vectors are L2-normalized before insertion, inner-product
    (IndexFlatIP) is mathematically equivalent to cosine similarity.
    Thread safety is provided via an internal RLock.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        index_path: Optional[str] = None,
    ):
        self.feature_dim = feature_dim
        self.index_path = index_path
        self._faiss = _try_import_faiss()
        self._index = None
        self._lock = threading.RLock()
        self._metadata: List[Dict] = []
        self._initialized = False

    def initialize(self):
        """Build a new index or load from disk."""
        with self._lock:
            if self._initialized:
                return

            if self._faiss is None:
                logger.warning(
                    "faiss-cpu not available; FaissFalsePositiveIndex running in no-op mode"
                )
                self._initialized = True
                return

            if self.index_path and Path(self.index_path).exists():
                self._load_from_disk()
            else:
                self._index = self._faiss.IndexFlatIP(self.feature_dim)
                self._metadata = []
                logger.info(
                    "Created new FAISS IndexFlatIP (dim=%d)", self.feature_dim
                )

            self._initialized = True

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: Optional[List[Dict]] = None,
    ):
        """Add L2-normalized feature vectors to the index.

        Args:
            vectors: (N, feature_dim) float32, must be L2-normalized.
            metadata: Optional per-vector metadata dicts.
        """
        if self._faiss is None or self._index is None:
            return

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        with self._lock:
            self._index.add(vectors)
            n = vectors.shape[0]
            if metadata is not None:
                self._metadata.extend(metadata)
            else:
                self._metadata.extend([{}] * n)
            logger.debug("Added %d vectors to FAISS index (total=%d)", n, self._index.ntotal)

    def search(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for top-K nearest neighbors by cosine similarity.

        Args:
            query: (N, feature_dim) or (feature_dim,) float32, L2-normalized.
            top_k: Number of neighbors to retrieve.

        Returns:
            (similarities, indices) each of shape (N, top_k).
            If the index is empty, returns zero arrays.
        """
        if self._faiss is None or self._index is None:
            return (
                np.zeros((1, top_k), dtype=np.float32),
                np.full((1, top_k), -1, dtype=np.int64),
            )

        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        with self._lock:
            ntotal = self._index.ntotal
            if ntotal == 0:
                return (
                    np.zeros((query.shape[0], top_k), dtype=np.float32),
                    np.full((query.shape[0], top_k), -1, dtype=np.int64),
                )
            actual_k = min(top_k, ntotal)
            similarities, indices = self._index.search(query, actual_k)

        if actual_k < top_k:
            pad_sim = np.zeros((query.shape[0], top_k), dtype=np.float32)
            pad_idx = np.full((query.shape[0], top_k), -1, dtype=np.int64)
            pad_sim[:, :actual_k] = similarities
            pad_idx[:, :actual_k] = indices
            return pad_sim, pad_idx

        return similarities, indices

    def search_single(
        self,
        query: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience: search for a single feature vector.

        Returns:
            (similarities, indices) each of shape (top_k,).
        """
        sims, idxs = self.search(query.reshape(1, -1), top_k)
        return sims[0], idxs[0]

    @property
    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        with self._lock:
            return self._index.ntotal

    def save(self, path: Optional[str] = None):
        """Persist the index and metadata to disk."""
        save_path = path or self.index_path
        if save_path is None or self._faiss is None or self._index is None:
            return

        with self._lock:
            index_file = save_path
            meta_file = save_path + ".meta.json"

            dir_path = os.path.dirname(index_file)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            self._faiss.write_index(self._index, index_file)

            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)

            logger.info(
                "Saved FAISS index (%d vectors) to %s",
                self._index.ntotal,
                index_file,
            )

    def _load_from_disk(self):
        index_file = self.index_path
        meta_file = self.index_path + ".meta.json"

        if not os.path.exists(index_file):
            logger.warning("FAISS index file not found: %s", index_file)
            self._index = self._faiss.IndexFlatIP(self.feature_dim)
            self._metadata = []
            return

        self._index = self._faiss.read_index(index_file)

        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = [{}] * self._index.ntotal

        logger.info(
            "Loaded FAISS index: %d vectors (dim=%d) from %s",
            self._index.ntotal,
            self._index.d,
            index_file,
        )

    def stats(self) -> Dict:
        with self._lock:
            return {
                "total_vectors": self._index.ntotal if self._index else 0,
                "feature_dim": self.feature_dim,
                "faiss_available": self._faiss is not None,
                "index_type": "IndexFlatIP",
            }

    def shutdown(self):
        with self._lock:
            if self.index_path:
                self.save()
            self._index = None
            self._metadata = []
            self._initialized = False
        logger.info("FAISS false-positive index shut down")
