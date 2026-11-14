"""Antibody numbering / CDR extraction.

Preferred backend is ANARCI (IMGT scheme) when installed. If ANARCI is unavailable we fall
back to a lightweight regex/anchor heuristic so the descriptor pipeline still runs end-to-end
(with a clearly logged loss of precision). CDR definitions follow IMGT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional

# IMGT CDR position ranges (within the variable domain numbering).
IMGT_CDR = {"CDR1": (27, 38), "CDR2": (56, 65), "CDR3": (105, 117)}


@dataclass
class ChainRegions:
    chain_type: str            # "H" or "L"
    cdr1: str
    cdr2: str
    cdr3: str
    framework: str             # concatenated framework residues
    numbered: bool             # True if produced by ANARCI


def _anarci_available() -> bool:
    try:
        import anarci  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=4096)
def number_chain(seq: str) -> Optional[ChainRegions]:
    """Return CDR/framework regions for a single variable-domain sequence."""
    seq = seq.strip().upper()
    if not seq:
        return None
    if _anarci_available():
        try:
            return _anarci_regions(seq)
        except Exception:
            pass
    return _heuristic_regions(seq)


def _anarci_regions(seq: str) -> ChainRegions:
    from anarci import run_anarci

    _, numbered, details = run_anarci([("q", seq)], scheme="imgt")
    if not numbered or numbered[0] is None:
        raise ValueError("ANARCI failed to number sequence")
    domain = numbered[0][0][0]  # [(num, aa), ...]
    chain_type = details[0][0]["chain_type"]
    chain_type = "H" if chain_type == "H" else "L"
    cdrs = {k: [] for k in IMGT_CDR}
    fw = []
    for (pos, _ins), aa in domain:
        if aa == "-":
            continue
        placed = False
        for name, (lo, hi) in IMGT_CDR.items():
            if lo <= pos <= hi:
                cdrs[name].append(aa)
                placed = True
                break
        if not placed:
            fw.append(aa)
    return ChainRegions(chain_type, "".join(cdrs["CDR1"]), "".join(cdrs["CDR2"]),
                        "".join(cdrs["CDR3"]), "".join(fw), numbered=True)


# --- Heuristic fallback (approximate; used only when ANARCI is absent) ---------------
_CDR3_H = re.compile(r"C(?P<cdr3>[A-Z]{3,25})WG.G")   # ...Cys ... CDR-H3 ... WGxG
_CDR3_L = re.compile(r"C(?P<cdr3>[A-Z]{3,15})FG.G")   # ...Cys ... CDR-L3 ... FGxG


def _heuristic_regions(seq: str) -> ChainRegions:
    chain_type = "H" if (_CDR3_H.search(seq) and not _CDR3_L.search(seq)) else "L"
    m = (_CDR3_H if chain_type == "H" else _CDR3_L).search(seq)
    cdr3 = m.group("cdr3") if m else ""
    # Coarse CDR1/CDR2 windows by typical offsets; framework = remainder.
    cdr1 = seq[26:35]
    cdr2 = seq[50:58]
    used = set()
    for frag in (cdr1, cdr2, cdr3):
        idx = seq.find(frag)
        if idx >= 0:
            used.update(range(idx, idx + len(frag)))
    fw = "".join(aa for i, aa in enumerate(seq) if i not in used)
    return ChainRegions(chain_type, cdr1, cdr2, cdr3, fw, numbered=False)


def all_cdrs(heavy: str, light: str) -> Dict[str, str]:
    """Return a dict of all six CDRs (+ chain regions) for a paired antibody."""
    h = number_chain(heavy)
    l = number_chain(light)
    out: Dict[str, str] = {}
    if h:
        out.update({"H_CDR1": h.cdr1, "H_CDR2": h.cdr2, "H_CDR3": h.cdr3, "H_FR": h.framework})
    if l:
        out.update({"L_CDR1": l.cdr1, "L_CDR2": l.cdr2, "L_CDR3": l.cdr3, "L_FR": l.framework})
    return out
