"""Explainability: SHAP attribution + per-block importance + per-antibody liability cards.

Three complementary views:
  1. Global SHAP over interpretable descriptors (which physicochemical features drive a property).
  2. Per-representation-block importance (descriptors vs ESM-2 vs AbLang2 contribution).
  3. Per-antibody explanation cards: top contributing features + flagged sequence liabilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .features.descriptors import _liabilities
from .features.numbering import all_cdrs


def shap_global(model, X: np.ndarray, feature_names: List[str], out_dir: str,
                prop: str, max_display: int = 20) -> Optional[pd.DataFrame]:
    """Compute mean |SHAP| per feature for a fitted tree/linear model; save a summary CSV+plot."""
    try:
        import shap
    except ImportError:
        print("[warn] shap not installed; skipping SHAP. `pip install shap`.")
        return None

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    inner = getattr(model, "model_", model)  # unwrap LGBMHead
    try:
        explainer = shap.Explainer(inner, X)
        sv = explainer(X)
        vals = sv.values
    except Exception:
        explainer = shap.KernelExplainer(model.predict, shap.sample(X, 50))
        vals = explainer.shap_values(X, nsamples=100)

    imp = np.abs(vals).mean(0)
    df = (pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
          .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    df.to_csv(Path(out_dir) / f"shap_{prop}.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top = df.head(max_display)[::-1]
        plt.figure(figsize=(7, 6))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.title(f"Top SHAP features - {prop}")
        plt.tight_layout()
        plt.savefig(Path(out_dir) / f"shap_{prop}.png", dpi=130)
        plt.close()
    except Exception:
        pass
    return df


def block_importance(shap_df: pd.DataFrame, blocks: Dict[str, List[str]]) -> pd.DataFrame:
    """Aggregate mean|SHAP| by representation block."""
    name_to_block = {n: b for b, names in blocks.items() for n in names}
    shap_df = shap_df.copy()
    shap_df["block"] = shap_df["feature"].map(name_to_block).fillna("other")
    agg = shap_df.groupby("block")["mean_abs_shap"].sum().reset_index()
    agg["fraction"] = agg["mean_abs_shap"] / agg["mean_abs_shap"].sum()
    return agg.sort_values("fraction", ascending=False).reset_index(drop=True)


def liability_card(heavy: str, light: str, name: str = "antibody") -> Dict[str, object]:
    """Per-antibody interpretable developability flags (sequence liabilities in CDRs)."""
    regions = all_cdrs(heavy, light)
    cdr_liab: Dict[str, Dict[str, int]] = {}
    for rname, frag in regions.items():
        if "CDR" in rname and frag:
            lia = {k: v for k, v in _liabilities(frag).items() if v}
            if lia:
                cdr_liab[rname] = lia
    return {"antibody": name, "cdr_liabilities": cdr_liab,
            "n_flags": int(sum(sum(v.values()) for v in cdr_liab.values()))}
