"""Feature assembly: build the fused feature matrix and remember block boundaries.

Block boundaries (descriptors / esm2 / ablang2) are tracked so explainability can attribute
importance per representation, and so per-block scaling can be applied in-fold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data.dataset import GDPa1
from .features.descriptors import build_descriptor_matrix
from .features.plm import build_plm_matrix


@dataclass
class FeatureMatrix:
    X: pd.DataFrame
    blocks: Dict[str, List[str]] = field(default_factory=dict)  # block -> column names

    @property
    def values(self) -> np.ndarray:
        return self.X.to_numpy(dtype=float)

    @property
    def names(self) -> List[str]:
        return list(self.X.columns)


def build_features(ds: GDPa1, cfg: dict) -> FeatureMatrix:
    f = cfg["features"]
    parts: List[pd.DataFrame] = []
    blocks: Dict[str, List[str]] = {}

    if f.get("use_descriptors", True):
        desc = build_descriptor_matrix(ds.heavy, ds.light, ph=f.get("ph", 7.4))
        blocks["descriptors"] = list(desc.columns)
        parts.append(desc)

    if f.get("use_esm2", False) or f.get("use_ablang2", False):
        plm = build_plm_matrix(ds.heavy, ds.light, cfg)
        if plm.shape[1]:
            for prefix in ("esm2", "ablang2"):
                cols = [c for c in plm.columns if c.startswith(prefix)]
                if cols:
                    blocks[prefix] = cols
            parts.append(plm)

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=ds.df.index)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return FeatureMatrix(X=X, blocks=blocks)
