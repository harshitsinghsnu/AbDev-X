"""Interpretable biophysical / sequence descriptors for antibody developability.

Every feature here has a direct physicochemical meaning, which is what makes the model's
SHAP explanations actionable (e.g. "high CDR-H3 net positive charge -> high self-association").
Pure NumPy/pandas; no ML deps, no network, no GPU. Runs on CPU in milliseconds per antibody.

Feature groups (computed for whole H, whole L, and each CDR region):
  * length, net charge @ pH, isoelectric point (approx), GRAVY hydrophobicity
  * fraction aromatic / aliphatic / positive / negative / polar
  * Eisenberg hydrophobic moment (amphipathicity proxy)
  * developability liability motif counts (deamidation, isomerization, N-glyc, oxidation, free Cys)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .numbering import all_cdrs

# Kyte-Doolittle hydrophobicity.
_KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
       "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
       "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
# Eisenberg consensus hydrophobicity (for hydrophobic moment).
_EIS = {"A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29, "Q": -0.85, "E": -0.74,
        "G": 0.48, "H": -0.40, "I": 1.38, "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19,
        "P": 0.12, "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08}
# pKa values for net-charge (Henderson-Hasselbalch).
_PKA_POS = {"K": 10.5, "R": 12.5, "H": 6.0}  # + N-term 9.0
_PKA_NEG = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}  # + C-term 2.0

_AROMATIC = set("FWY")
_ALIPHATIC = set("AILVM")
_POSITIVE = set("KR")
_NEGATIVE = set("DE")
_POLAR = set("STNQ")


def _net_charge(seq: str, ph: float) -> float:
    if not seq:
        return 0.0
    pos = 1.0 / (1.0 + 10 ** (ph - 9.0))  # N-term
    for aa, pk in _PKA_POS.items():
        pos += seq.count(aa) * (1.0 / (1.0 + 10 ** (ph - pk)))
    neg = 1.0 / (1.0 + 10 ** (2.0 - ph))  # C-term
    for aa, pk in _PKA_NEG.items():
        neg += seq.count(aa) * (1.0 / (1.0 + 10 ** (pk - ph)))
    return pos - neg


def _approx_pi(seq: str) -> float:
    if not seq:
        return 7.0
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _net_charge(seq, mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _gravy(seq: str) -> float:
    vals = [_KD[a] for a in seq if a in _KD]
    return float(np.mean(vals)) if vals else 0.0


def _hydrophobic_moment(seq: str, period_deg: float = 100.0) -> float:
    """Eisenberg hydrophobic moment (amphipathicity); period 100deg ~ alpha-helix."""
    if not seq:
        return 0.0
    delta = np.deg2rad(period_deg)
    sx = sum(_EIS.get(a, 0.0) * np.cos(i * delta) for i, a in enumerate(seq))
    sy = sum(_EIS.get(a, 0.0) * np.sin(i * delta) for i, a in enumerate(seq))
    return float(np.sqrt(sx * sx + sy * sy) / max(len(seq), 1))


def _frac(seq: str, members: set) -> float:
    return sum(1 for a in seq if a in members) / len(seq) if seq else 0.0


def _liabilities(seq: str) -> Dict[str, int]:
    """Counts of common chemical-/sequence-liability motifs."""
    return {
        "deamidation_NG_NS": seq.count("NG") + seq.count("NS"),
        "isomerization_DG_DS": seq.count("DG") + seq.count("DS"),
        "nglyc_NxST": sum(1 for i in range(len(seq) - 2)
                          if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST"),
        "oxidation_MW": seq.count("M") + seq.count("W"),
        "free_cys": seq.count("C") % 2,  # odd Cys -> potential unpaired
        "tryptophan": seq.count("W"),
    }


def region_descriptors(seq: str, ph: float, prefix: str) -> Dict[str, float]:
    seq = (seq or "").upper()
    d: Dict[str, float] = {
        f"{prefix}_len": float(len(seq)),
        f"{prefix}_charge_pH": _net_charge(seq, ph),
        f"{prefix}_pI": _approx_pi(seq),
        f"{prefix}_gravy": _gravy(seq),
        f"{prefix}_hmoment": _hydrophobic_moment(seq),
        f"{prefix}_frac_aromatic": _frac(seq, _AROMATIC),
        f"{prefix}_frac_aliphatic": _frac(seq, _ALIPHATIC),
        f"{prefix}_frac_pos": _frac(seq, _POSITIVE),
        f"{prefix}_frac_neg": _frac(seq, _NEGATIVE),
        f"{prefix}_frac_polar": _frac(seq, _POLAR),
    }
    for k, v in _liabilities(seq).items():
        d[f"{prefix}_{k}"] = float(v)
    return d


def descriptors_for_pair(heavy: str, light: str, ph: float = 7.4) -> Dict[str, float]:
    """Full interpretable descriptor vector for one paired antibody."""
    feats: Dict[str, float] = {}
    feats.update(region_descriptors(heavy, ph, "H"))
    feats.update(region_descriptors(light, ph, "L"))

    regions = all_cdrs(heavy, light)
    for name, frag in regions.items():
        feats.update(region_descriptors(frag, ph, name))

    # Whole-Fv aggregate signals.
    fv = (heavy or "") + (light or "")
    feats["Fv_charge_pH"] = _net_charge(fv.upper(), ph)
    feats["Fv_pI"] = _approx_pi(fv.upper())
    cdr3 = regions.get("H_CDR3", "") + regions.get("L_CDR3", "")
    feats["CDR3_total_len"] = float(len(cdr3))
    feats["CDR3_net_charge"] = _net_charge(cdr3.upper(), ph)
    feats["CDR3_hydrophobicity"] = _gravy(cdr3.upper())
    return feats


def build_descriptor_matrix(heavy: pd.Series, light: pd.Series, ph: float = 7.4) -> pd.DataFrame:
    """Vectorize descriptors over all antibodies -> dense, named feature frame."""
    rows: List[Dict[str, float]] = [
        descriptors_for_pair(h, l, ph) for h, l in zip(heavy.tolist(), light.tolist())
    ]
    df = pd.DataFrame(rows).fillna(0.0)
    df.index = heavy.index
    return df


if __name__ == "__main__":
    # Smoke test with a toy VH/VL pair.
    vh = ("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKG"
          "RFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGYYGMDVWGQGTTVTVSS")
    vl = ("DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGQGTKVEIK")
    feats = descriptors_for_pair(vh, vl)
    print(f"{len(feats)} descriptors computed. Sample:")
    for k in list(feats)[:12]:
        print(f"  {k}: {feats[k]:.3f}")
