from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from wafer_aoi.utils import DetectedDefect, setup_logger

logger = setup_logger(__name__)


class FeatureExtractor:
    """Extract compact feature vectors from bounding-box crop regions.

    The pipeline is:
      1. Crop the detection region from the original frame (with context padding).
      2. Resize to a fixed patch size.
      3. Compute a multi-scale LBP + histogram feature vector.
      4. L2-normalize for cosine-similarity compatibility with FAISS IndexFlatIP.

    This is deliberately lightweight (CPU-only, <0.5 ms per crop) so it does
    not add latency to the post-processing hot path.  If a GPU feature
    backbone is available downstream, the feature_dim should match that
    network's embedding size.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (64, 64),
        feature_dim: int = 256,
        context_padding_ratio: float = 0.15,
    ):
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.context_padding_ratio = context_padding_ratio

    def extract_from_frame(
        self,
        frame: np.ndarray,
        defect: DetectedDefect,
    ) -> Optional[np.ndarray]:
        """Extract a single feature vector for one detection on one frame."""
        crop = self._crop_with_context(frame, defect.bbox)
        if crop is None or crop.size == 0:
            return None

        patch = cv2.resize(crop, self.patch_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        feat = self._compute_features(gray)
        feat = self._l2_normalize(feat)
        return feat

    def extract_batch(
        self,
        frame: np.ndarray,
        defects: List[DetectedDefect],
    ) -> List[Optional[np.ndarray]]:
        """Extract feature vectors for multiple detections on the same frame."""
        return [self.extract_from_frame(frame, d) for d in defects]

    def _crop_with_context(
        self, frame: np.ndarray, bbox: tuple
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        bw, bh = x2 - x1, y2 - y1
        pad_w = int(bw * self.context_padding_ratio)
        pad_h = int(bh * self.context_padding_ratio)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _compute_features(self, gray: np.ndarray) -> np.ndarray:
        parts: List[np.ndarray] = []

        hist_1 = self._lbp_histogram(gray, radius=1, n_points=8)
        parts.append(hist_1)

        hist_2 = self._lbp_histogram(gray, radius=2, n_points=12)
        parts.append(hist_2)

        h, w = gray.shape
        if h >= 4 and w >= 4:
            quadrants = [
                gray[: h // 2, : w // 2],
                gray[: h // 2, w // 2 :],
                gray[h // 2 :, : w // 2],
                gray[h // 2 :, w // 2 :],
            ]
            for q in quadrants:
                parts.append(self._lbp_histogram(q, radius=1, n_points=8))

        concatenated = np.concatenate(parts).astype(np.float32)

        if len(concatenated) >= self.feature_dim:
            feature = concatenated[: self.feature_dim]
        else:
            feature = np.zeros(self.feature_dim, dtype=np.float32)
            feature[: len(concatenated)] = concatenated

        return feature

    @staticmethod
    def _lbp_histogram(
        gray: np.ndarray, radius: int = 1, n_points: int = 8
    ) -> np.ndarray:
        h, w = gray.shape
        padded = cv2.copyMakeBorder(
            gray, radius, radius, radius, radius, cv2.BORDER_REFLECT
        )
        center = padded[radius : radius + h, radius : radius + w]
        lbp = np.zeros((h, w), dtype=np.uint8)

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dx = int(round(radius * np.cos(angle)))
            dy = int(round(radius * np.sin(angle)))
            neighbor = padded[
                radius + dy : radius + dy + h,
                radius + dx : radius + dx + w,
            ]
            lbp |= ((neighbor >= center).astype(np.uint8) << i)

        n_bins = 2 ** min(n_points, 8)
        hist = np.zeros(n_bins, dtype=np.float32)
        for b in range(n_bins):
            hist[b] = np.sum(lbp == b)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec
