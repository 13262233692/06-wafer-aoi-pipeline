from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wafer_aoi.ohem.feature_extractor import FeatureExtractor
from wafer_aoi.ohem.faiss_index import FaissFalsePositiveIndex
from wafer_aoi.utils import DetectedDefect, setup_logger

logger = setup_logger(__name__)


@dataclass
class OhemVerdict:
    defect: DetectedDefect
    suppressed: bool = False
    max_similarity: float = 0.0
    matched_fp_ids: List[int] = field(default_factory=list)
    feature_vector: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "defect": self.defect.to_dict(),
            "suppressed": self.suppressed,
            "max_similarity": float(self.max_similarity),
            "matched_fp_count": len(self.matched_fp_ids),
        }


class OhemFilter:
    """Online Hard Example Mining filter.

    For each detection whose confidence falls in the ambiguous zone
    [ohem_low, ohem_high], extract a feature vector from the crop region
    and query the FAISS false-positive index.  If the top-1 cosine
    similarity exceeds ohem_similarity_threshold, the detection is
    **suppressed** (downgraded from a true defect to a false positive).

    All ambiguous detections — whether suppressed or not — are returned as
    HardExample objects so the archiver can persist them for future model
    retraining.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        fp_index: FaissFalsePositiveIndex,
        ohem_low: float = 0.4,
        ohem_high: float = 0.7,
        similarity_threshold: float = 0.85,
        top_k: int = 3,
    ):
        self.feature_extractor = feature_extractor
        self.fp_index = fp_index
        self.ohem_low = ohem_low
        self.ohem_high = ohem_high
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        self._total_filtered = 0
        self._total_ambiguous = 0

    @property
    def is_available(self) -> bool:
        return self.fp_index._faiss is not None

    def filter_defects(
        self,
        frame: np.ndarray,
        defects: List[DetectedDefect],
    ) -> Tuple[List[DetectedDefect], List[OhemVerdict]]:
        """Apply OHEM filtering to a list of detections on a single frame.

        Returns:
            kept:      List of defects that survived filtering.
            hard_examples: List of OhemVerdict for all ambiguous detections
                           (both suppressed and retained).
        """
        if not defects or not self.is_available:
            return defects, []

        kept: List[DetectedDefect] = []
        hard_examples: List[OhemVerdict] = []

        for defect in defects:
            if not self._is_ambiguous(defect):
                kept.append(defect)
                continue

            self._total_ambiguous += 1

            feat = self.feature_extractor.extract_from_frame(frame, defect)
            if feat is None:
                kept.append(defect)
                continue

            similarities, indices = self.fp_index.search_single(feat, self.top_k)
            max_sim = float(similarities[0]) if len(similarities) > 0 else 0.0
            matched_ids = [int(idx) for idx in indices if idx >= 0]

            verdict = OhemVerdict(
                defect=defect,
                suppressed=False,
                max_similarity=max_sim,
                matched_fp_ids=matched_ids,
                feature_vector=feat,
            )

            if max_sim >= self.similarity_threshold:
                verdict.suppressed = True
                self._total_filtered += 1
                logger.debug(
                    "OHEM suppressed %s conf=%.3f sim=%.3f",
                    defect.class_name,
                    defect.confidence,
                    max_sim,
                )
            else:
                kept.append(defect)

            hard_examples.append(verdict)

        return kept, hard_examples

    def filter_defects_batch(
        self,
        frames: List[np.ndarray],
        batch_defects: List[List[DetectedDefect]],
    ) -> Tuple[List[List[DetectedDefect]], List[List[OhemVerdict]]]:
        """Apply OHEM filtering across a batch of frames."""
        kept_batch: List[List[DetectedDefect]] = []
        hard_batch: List[List[OhemVerdict]] = []

        for frame, defects in zip(frames, batch_defects):
            kept, hard = self.filter_defects(frame, defects)
            kept_batch.append(kept)
            hard_batch.append(hard)

        return kept_batch, hard_batch

    def _is_ambiguous(self, defect: DetectedDefect) -> bool:
        return self.ohem_low <= defect.confidence <= self.ohem_high

    def stats(self) -> Dict:
        return {
            "ohem_low": self.ohem_low,
            "ohem_high": self.ohem_high,
            "similarity_threshold": self.similarity_threshold,
            "total_ambiguous": self._total_ambiguous,
            "total_filtered": self._total_filtered,
            "filter_rate": (
                self._total_filtered / self._total_ambiguous
                if self._total_ambiguous > 0
                else 0.0
            ),
            "fp_index_stats": self.fp_index.stats(),
        }
