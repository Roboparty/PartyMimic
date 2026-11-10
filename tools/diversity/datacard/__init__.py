"""Dataset-diversity data cards for whole-body tracking datasets.

Pipeline: ``features`` (heading-invariant 1 s window features per robot) -> ``embed`` (frozen z-score + PCA32 on a
declared reference pool) -> ``codebook`` (k-means occupancy) -> ``coverage`` (target-centred novelty / Naeem coverage)
-> ``card`` (JSON + markdown report), plus ``dedup`` (duplicated segments in raw joint space) and ``splits``
(source- / cluster-held-out splits). Command line: ``python -m datacard --help``.
"""

from .card import Reference, build_card, fit_reference, write_card
from .datasets import DatasetSpec, GroupSpec
from .features import FEATURE_VERSION, FeatureSet, build_features

__all__ = ["DatasetSpec", "FEATURE_VERSION", "FeatureSet", "GroupSpec", "Reference", "build_card", "build_features",
           "fit_reference", "write_card"]
