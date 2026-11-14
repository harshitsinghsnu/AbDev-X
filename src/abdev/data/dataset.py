"""Load and normalize the GDPa1 table into a typed in-memory dataset.

The public CSV header is not assumed: every logical column is resolved against a list of
candidate aliases (case-insensitive, whitespace/underscore tolerant). This keeps the loader
robust to minor schema/version drift while still failing loudly when a required column is
genuinely absent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_column(df: pd.DataFrame, aliases: List[str], required: bool = True,
                   what: str = "column") -> Optional[str]:
    """Return the real df column matching any alias (normalized), else None/raise."""
    norm_map = {_norm(c): c for c in df.columns}
    for a in aliases:
        if _norm(a) in norm_map:
            return norm_map[_norm(a)]
    if required:
        raise KeyError(
            f"Could not resolve {what}. Tried aliases {aliases}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


@dataclass
class GDPa1:
    df: pd.DataFrame
    id_col: str
    heavy_col: str
    light_col: str
    fold_cols: Dict[str, str]            # logical name -> real column
    target_cols: Dict[str, str]          # property -> real column
    aux_target_cols: Dict[str, str] = field(default_factory=dict)
    directions: Dict[str, str] = field(default_factory=dict)

    @property
    def heavy(self) -> pd.Series:
        return self.df[self.heavy_col].astype(str).str.strip()

    @property
    def light(self) -> pd.Series:
        return self.df[self.light_col].astype(str).str.strip()

    def y(self, prop: str) -> np.ndarray:
        return self.df[self.target_cols[prop]].to_numpy(dtype=float)

    def folds(self, logical_name: str) -> np.ndarray:
        return self.df[self.fold_cols[logical_name]].to_numpy()

    def labeled_mask(self, prop: str) -> np.ndarray:
        """Boolean mask of rows with a non-NaN measurement for `prop`."""
        return ~self.df[self.target_cols[prop]].isna().to_numpy()

    def summary(self) -> pd.DataFrame:
        rows = []
        for p, col in self.target_cols.items():
            v = self.df[col]
            rows.append({"property": p, "column": col, "n_labeled": int(v.notna().sum()),
                         "mean": float(v.mean()), "std": float(v.std())})
        return pd.DataFrame(rows)


def _resolve_csv_path(d: dict, csv_path: Optional[str]) -> Path:
    if csv_path:
        return Path(csv_path)
    active = Path(d["raw_dir"]) / "ACTIVE_CSV.txt"
    if active.exists():
        return Path(d["raw_dir"]) / active.read_text(encoding="utf-8").strip()
    return Path(d["raw_dir"]) / d["csv_filename"]


def _aggregate_replicates(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """If the table is tidy/long (multiple rows per antibody), collapse to one row each:
    numeric assay columns -> mean, sequence/metadata columns -> first."""
    if df[id_col].is_unique:
        return df
    num_cols = df.select_dtypes("number").columns
    agg = {c: ("mean" if c in num_cols else "first") for c in df.columns if c != id_col}
    out = df.groupby(id_col, as_index=False).agg(agg)
    print(f"[info] tidy table detected: aggregated {len(df)} rows -> {len(out)} antibodies.")
    return out


def load_gdpa1(config_path: str = "configs/default.yaml",
               csv_path: Optional[str] = None) -> GDPa1:
    cfg = yaml.safe_load(open(config_path))
    d = cfg["data"]
    path = _resolve_csv_path(d, csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m abdev.data.download --config {config_path}` first."
        )
    df = pd.read_csv(path)
    print(f"[info] loaded {path.name}: {len(df)} rows x {df.shape[1]} cols")

    id_col = resolve_column(df, d["id_aliases"], what="id column")
    if d.get("aggregate_replicates", True):
        df = _aggregate_replicates(df, id_col)
    heavy_col = resolve_column(df, d["heavy_aliases"], what="heavy chain column")
    light_col = resolve_column(df, d["light_aliases"], what="light chain column")

    fold_cols = {}
    for logical, aliases in d["fold_aliases"].items():
        col = resolve_column(df, aliases, required=False, what=f"{logical} fold")
        if col:
            fold_cols[logical] = col

    target_cols, directions = {}, {}
    for prop, spec in cfg["targets"].items():
        col = resolve_column(df, spec["column"], required=False, what=f"target {prop}")
        if col:
            target_cols[prop] = col
            directions[prop] = spec.get("direction", "higher")
        else:
            print(f"[warn] target '{prop}' not found; skipping. (aliases: {spec['column']})")

    aux_cols = {}
    for aliases in cfg.get("aux_targets", {}).get("column_aliases", []):
        col = resolve_column(df, aliases, required=False, what="aux target")
        if col:
            aux_cols[aliases[0]] = col

    ds = GDPa1(df=df, id_col=id_col, heavy_col=heavy_col, light_col=light_col,
               fold_cols=fold_cols, target_cols=target_cols,
               aux_target_cols=aux_cols, directions=directions)
    return ds


if __name__ == "__main__":
    ds = load_gdpa1()
    print(f"Loaded {len(ds.df)} antibodies.")
    print("Folds:", ds.fold_cols)
    print(ds.summary().to_string(index=False))
