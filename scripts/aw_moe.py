"""AW-MoE — Assay-Winner Mixture of Experts.

Four specialist experts, each independently trained:
  E1    AbDev-X       best OOD-CV block per assay (seq/bio/desc/iso)
  E2    A-MGFS        assay-specific biological keyword masks on desc+bio
  E3    D-stack       OOF ridge stacking of PLM-stream + bio/desc-stream
  E4    F-MoE         mechanism expert sub-blocks with softmax gating

Three gating variants:
  AW-MoE-1   equal weights (1/K)  — ensemble baseline
  AW-MoE-2   learned assay-specific weights (non-negative, sum-to-1)
  AW-MoE-3   sample-level gate: softmax(W * oof_preds + b)  — exploratory

Training protocol (leakage-free):
  1. Outer OOD-CV folds (same folds for every expert)
  2. Each outer fold: train expert on folds≠f, predict fold f → OOF matrix P^(a)
  3. Fit gating weights on OOF predictions vs GDPa1 labels
  4. Refit base experts on full GDPa1
  5. Evaluate on GDPa3 (gate was never fitted on GDPa3)

Evaluated on 5 genuine assays only.
"""
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abdev.data.dataset import load_gdpa1
from abdev.features.descriptors import build_descriptor_matrix

GENUINE_ASSAYS = {
    "HIC":     ("HIC",           "hic_rt_avg"),
    "AC-SINS": ("AC-SINS_pH7.4", "acsins_dLmax_ph7.4_avg"),
    "PR_CHO":  ("PR_CHO",        "polyreactivity_prscore_cho_avg"),
    "SMAC":    ("SMAC",          "smac_rt_avg"),
    "HAC":     ("HAC",           "hac_rt_avg"),
}
ISO_INELIGIBLE = {"Tm1", "Tm2", "Tonset", "SEC"}
BIO_FILES = ["TAP.csv", "MOE_properties.csv", "Aggrescan3D.csv", "AntiFold.csv"]
ISO_COLS  = ["hc_subtype_IgG1", "hc_subtype_IgG2", "hc_subtype_IgG4",
             "lc_subtype_Kappa", "lc_subtype_Lambda"]
N_BOOT = 2000

MGFS_MASKS = {
    "HIC":    {"inc": ["hydro","gravy","aliphat","aromat","patch","sasa","SASA"],
               "exc": ["charge","pI","deamid","glyc"]},
    "AC-SINS":{"inc": ["charge","pI","ppc","pnc","patch_pos","patch_neg","dipole","electr"],
               "exc": ["hydro","gravy"]},
    "PR_CHO": {"inc": ["charge","cdr","CDR","hydro","gravy","patch"],
               "exc": []},
    "SMAC":   {"inc": ["sasa","SASA","patch","hydro","charge","exposed"],
               "exc": []},
    "HAC":    {"inc": ["sasa","SASA","patch","hydro","charge","exposed","stab"],
               "exc": []},
}
MOE_EXPERTS = {
    "hydro":  ["hydro","gravy","aliphat","aromat","patch_hyd","hyd"],
    "charge": ["charge","pI","ppc","pnc","electr","dipole","patch_pos","patch_neg"],
    "stab":   ["stab","sasa","gyrat","pack","AntiFold","tm","Tm","cavity","burial"],
    "liab":   ["deamid","isomeriz","glyc","oxid","cys","nglyc","asn","asp"],
    "cdr":    ["cdr","CDR","loop","h3","l3","h1","h2","l1","l2"],
    "surface":["sasa","SASA","rSASA","exposed","surface","patch"],
    "global": ["len","mass","count","frac","total","net"],
}
MOE_ELIGIBILITY = {
    "HIC":    ["hydro","surface"],
    "AC-SINS":["charge","surface"],
    "PR_CHO": ["charge","cdr","hydro"],
    "SMAC":   ["surface","hydro","charge"],
    "HAC":    ["surface","hydro","charge","stab"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _iso(df):
    d = pd.get_dummies(df[["hc_subtype", "lc_subtype"]].astype(str))
    return d.reindex(columns=ISO_COLS, fill_value=0).to_numpy(float)

def _bio(feat_dir, names, cols=None):
    base = pd.DataFrame({"antibody_name": list(names)})
    for f in BIO_FILES:
        p = Path(feat_dir) / f
        if not p.exists(): continue
        d = pd.read_csv(p)
        num = d.select_dtypes("number").copy()
        num["antibody_name"] = d["antibody_name"].values
        num = num.rename(columns={c: f"{f[:-4]}::{c}" for c in num.columns
                                  if c != "antibody_name"})
        base = base.merge(num, on="antibody_name", how="left")
    X = base.drop(columns=["antibody_name"])
    if cols is None: cols = list(X.columns)
    return np.nan_to_num(X.reindex(columns=cols, fill_value=0).to_numpy(float)), cols

def _head(X):
    steps = [("s", StandardScaler())]
    if X.shape[1] > 100:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=np.logspace(-2, 4, 12))))
    return Pipeline(steps)

def _mask_cols(all_cols, inc_kws, exc_kws=()):
    inc = [k.lower() for k in inc_kws]
    exc = [k.lower() for k in exc_kws]
    return np.array([(any(k in c.lower() for k in inc) and
                      not any(k in c.lower() for k in exc))
                     for c in all_cols])

def bca(yte, pred, rng, n=N_BOOT):
    ok = ~np.isnan(yte) & ~np.isnan(pred)
    y, p = yte[ok], pred[ok]; n_ = len(y)
    if n_ < 5: return np.nan, np.nan
    from scipy.stats import norm
    obs = spearmanr(y, p).statistic
    boot = np.array([spearmanr(y[s:=rng.integers(0, n_, n_)], p[s]).statistic
                     for _ in range(n)])
    z0 = norm.ppf(np.clip(np.mean(boot < obs), 1e-6, 1-1e-6))
    jk = np.array([spearmanr(np.delete(y, i), np.delete(p, i)).statistic
                   for i in range(n_)])
    jm = jk.mean()
    a = ((jm - jk)**3).sum() / (6 * (((jm - jk)**2).sum()**1.5) + 1e-12)
    def adj(z): return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return (float(np.percentile(boot, 100 * adj(norm.ppf(0.025)))),
            float(np.percentile(boot, 100 * adj(norm.ppf(0.975)))))


# ── per-expert prediction functions ──────────────────────────────────────────

class E1Expert:
    """Best-block selection per assay (OOD-CV within training set)."""
    def __init__(self, blocks_tr, blocks_te, folds, fold_ids, assay):
        self.blocks_tr = blocks_tr
        self.blocks_te = blocks_te
        self.folds = folds
        self.fold_ids = fold_ids
        self.assay = assay
        self.pipe = None
        self.best_blk = None

    def fit_oof(self, tr_idx, y):
        """Return OOF predictions for tr_idx using inner OOD-CV selection."""
        m = ~np.isnan(y)
        # Select best block using only tr_idx (inner leave-one-fold-out)
        inner_folds = self.folds[tr_idx]
        inner_ids = sorted(set(inner_folds.tolist()))
        best_blk, best_cv = None, -9.0
        for blk, Xtr in self.blocks_tr.items():
            scores = []
            for f in inner_ids:
                ii_tr = np.where((inner_folds != f) & m[tr_idx])[0]
                ii_te = np.where((inner_folds == f) & m[tr_idx])[0]
                if len(ii_te) < 3 or len(ii_tr) < 8: continue
                pipe = _head(Xtr[tr_idx])
                pipe.fit(Xtr[tr_idx[ii_tr]], y[tr_idx[ii_tr]])
                scores.append(spearmanr(y[tr_idx[ii_te]],
                                        pipe.predict(Xtr[tr_idx[ii_te]])).statistic)
            if scores and np.nanmean(scores) > best_cv:
                best_cv = np.nanmean(scores)
                best_blk = blk
        if best_blk is None: best_blk = "desc"
        Xb = self.blocks_tr[best_blk]
        oof = np.full(len(tr_idx), np.nan)
        for f in inner_ids:
            ii_tr = np.where((inner_folds != f) & m[tr_idx])[0]
            ii_te = np.where((inner_folds == f) & m[tr_idx])[0]
            if len(ii_te) < 3 or len(ii_tr) < 8: continue
            pipe = _head(Xb[tr_idx])
            pipe.fit(Xb[tr_idx[ii_tr]], y[tr_idx[ii_tr]])
            oof[ii_te] = pipe.predict(Xb[tr_idx[ii_te]])
        return oof, best_blk

    def fit_full(self, y):
        m = ~np.isnan(y)
        inner_folds = self.folds
        inner_ids = self.fold_ids
        best_blk, best_cv = None, -9.0
        for blk, Xtr in self.blocks_tr.items():
            scores = []
            for f in inner_ids:
                ii_tr = np.where((inner_folds != f) & m)[0]
                ii_te = np.where((inner_folds == f) & m)[0]
                if len(ii_te) < 3 or len(ii_tr) < 8: continue
                pipe = _head(Xtr); pipe.fit(Xtr[ii_tr], y[ii_tr])
                scores.append(spearmanr(y[ii_te], pipe.predict(Xtr[ii_te])).statistic)
            if scores and np.nanmean(scores) > best_cv:
                best_cv = np.nanmean(scores); best_blk = blk
        if best_blk is None: best_blk = "desc"
        self.best_blk = best_blk
        Xb = self.blocks_tr[best_blk]
        self.pipe = _head(Xb); self.pipe.fit(Xb[m], y[m])

    def predict(self):
        return self.pipe.predict(self.blocks_te[self.best_blk])


class MGFSExpert:
    def __init__(self, Xdb_tr, Xdb_te, all_cols, assay):
        cfg = MGFS_MASKS.get(assay, {"inc": [], "exc": []})
        mask = _mask_cols(all_cols, cfg["inc"], cfg["exc"])
        if mask.sum() < 5: mask = np.ones(len(all_cols), bool)
        self.Xtr = Xdb_tr[:, mask]
        self.Xte = Xdb_te[:, mask]
        self.pipe = None

    def fit(self, idx, y):
        m = ~np.isnan(y[idx])
        pipe = _head(self.Xtr[idx])
        pipe.fit(self.Xtr[idx[m]], y[idx[m]])
        return pipe.predict(self.Xtr[idx])

    def fit_full(self, y):
        m = ~np.isnan(y)
        self.pipe = _head(self.Xtr); self.pipe.fit(self.Xtr[m], y[m])

    def predict(self):
        return self.pipe.predict(self.Xte)


class DStackExpert:
    def __init__(self, Xplm_tr, Xplm_te, Xstr_tr, Xstr_te, folds, fold_ids):
        self.Xplm_tr = Xplm_tr; self.Xplm_te = Xplm_te
        self.Xstr_tr = Xstr_tr; self.Xstr_te = Xstr_te
        self.folds = folds; self.fold_ids = fold_ids
        self.p1 = self.p2 = self.meta = None

    def fit(self, idx, y):
        """OOF predictions for idx using inner stack."""
        m = ~np.isnan(y); inner_folds = self.folds[idx]; inner_ids = sorted(set(inner_folds))
        oof1 = np.zeros(len(idx)); oof2 = np.zeros(len(idx))
        for f in inner_ids:
            ii_tr = np.where((inner_folds != f) & m[idx])[0]
            ii_te = np.where((inner_folds == f) & m[idx])[0]
            if len(ii_te) < 3 or len(ii_tr) < 8: continue
            p1 = _head(self.Xplm_tr[idx]); p1.fit(self.Xplm_tr[idx[ii_tr]], y[idx[ii_tr]])
            oof1[ii_te] = p1.predict(self.Xplm_tr[idx[ii_te]])
            p2 = _head(self.Xstr_tr[idx]); p2.fit(self.Xstr_tr[idx[ii_tr]], y[idx[ii_tr]])
            oof2[ii_te] = p2.predict(self.Xstr_tr[idx[ii_te]])
        meta_tr = np.column_stack([oof1, oof2])
        meta = Ridge(alpha=1.0); meta.fit(meta_tr[m[idx]], y[idx[m[idx]]])
        return meta.predict(meta_tr)

    def fit_full(self, y):
        m = ~np.isnan(y)
        inner_folds = self.folds; inner_ids = self.fold_ids
        oof1 = np.zeros(len(y)); oof2 = np.zeros(len(y))
        for f in inner_ids:
            tr = np.where((inner_folds != f) & m)[0]; te = np.where((inner_folds == f) & m)[0]
            if len(te) < 3 or len(tr) < 8: continue
            p1 = _head(self.Xplm_tr); p1.fit(self.Xplm_tr[tr], y[tr]); oof1[te] = p1.predict(self.Xplm_tr[te])
            p2 = _head(self.Xstr_tr); p2.fit(self.Xstr_tr[tr], y[tr]); oof2[te] = p2.predict(self.Xstr_tr[te])
        meta_tr = np.column_stack([oof1, oof2])
        self.p1 = _head(self.Xplm_tr); self.p1.fit(self.Xplm_tr[m], y[m])
        self.p2 = _head(self.Xstr_tr); self.p2.fit(self.Xstr_tr[m], y[m])
        self.meta = Ridge(alpha=1.0); self.meta.fit(meta_tr[m], y[m])

    def predict(self):
        pred1 = self.p1.predict(self.Xplm_te)
        pred2 = self.p2.predict(self.Xstr_te)
        return self.meta.predict(np.column_stack([pred1, pred2]))


class FMoEExpert:
    def __init__(self, Xdb_tr, Xdb_te, all_cols, assay, folds, fold_ids):
        eligible = MOE_ELIGIBILITY.get(assay, list(MOE_EXPERTS.keys()))
        self.expert_masks = []
        for exp in eligible:
            kws = MOE_EXPERTS.get(exp, [])
            mask = _mask_cols(all_cols, kws)
            if mask.sum() >= 3:
                self.expert_masks.append((exp, mask))
        if not self.expert_masks:
            self.expert_masks = [("all", np.ones(Xdb_tr.shape[1], bool))]
        self.Xdb_tr = Xdb_tr; self.Xdb_te = Xdb_te
        self.folds = folds; self.fold_ids = fold_ids
        self.pipes = []; self.weights = None; self.tau = 0.1

    def _expert_oof(self, idx, y, mask):
        m = ~np.isnan(y); Xe = self.Xdb_tr[:, mask]
        inner_folds = self.folds[idx]; inner_ids = sorted(set(inner_folds))
        oof = np.zeros(len(idx)); cv_scores = []
        for f in inner_ids:
            ii_tr = np.where((inner_folds != f) & m[idx])[0]
            ii_te = np.where((inner_folds == f) & m[idx])[0]
            if len(ii_te) < 3 or len(ii_tr) < 8: continue
            pipe = _head(Xe[idx]); pipe.fit(Xe[idx[ii_tr]], y[idx[ii_tr]])
            pred = pipe.predict(Xe[idx[ii_te]]); oof[ii_te] = pred
            cv_scores.append(spearmanr(y[idx[ii_te]], pred).statistic)
        return oof, float(np.nanmean(cv_scores)) if cv_scores else -1.0

    def fit(self, idx, y):
        oofs = []; cvs = []
        for _, mask in self.expert_masks:
            oof, cv = self._expert_oof(idx, y, mask)
            oofs.append(oof); cvs.append(max(cv, 0.0))
        cvs = np.array(cvs); ws = np.exp(cvs / self.tau); ws /= ws.sum()
        stacked = np.column_stack(oofs) if len(oofs) > 1 else oofs[0].reshape(-1, 1)
        return stacked @ ws

    def fit_full(self, y):
        m = ~np.isnan(y); cvs = []
        inner_folds = self.folds; inner_ids = self.fold_ids
        for _, mask in self.expert_masks:
            Xe = self.Xdb_tr[:, mask]; scores = []
            for f in inner_ids:
                tr = np.where((inner_folds != f) & m)[0]; te = np.where((inner_folds == f) & m)[0]
                if len(te) < 3 or len(tr) < 8: continue
                pipe = _head(Xe); pipe.fit(Xe[tr], y[tr])
                scores.append(spearmanr(y[te], pipe.predict(Xe[te])).statistic)
            cvs.append(float(np.nanmean(scores)) if scores else -1.0)
        cvs = np.array(cvs); ws = np.exp(np.maximum(cvs, 0) / self.tau); ws /= ws.sum()
        self.weights = ws
        for _, mask in self.expert_masks:
            Xe = self.Xdb_tr[:, mask]
            pipe = _head(Xe); pipe.fit(Xe[m], y[m]); self.pipes.append(pipe)

    def predict(self):
        preds = np.column_stack([pipe.predict(self.Xdb_te[:, mask])
                                  for pipe, (_, mask) in zip(self.pipes, self.expert_masks)])
        return preds @ self.weights


# ── gating ────────────────────────────────────────────────────────────────────

def fit_aw_moe2(oof_matrix, y, assay):
    """Constrained non-negative sum-to-1 weights via scipy minimize."""
    m = ~np.isnan(y)
    P = oof_matrix[m]; yt = y[m]; K = P.shape[1]
    def loss(w):
        return -spearmanr(yt, P @ w).statistic + 0.1 * np.sum(w**2)
    w0 = np.ones(K) / K
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, None)] * K
    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    w = np.maximum(res.x, 0); w /= w.sum()
    return w

def predict_aw_moe3(oof_matrix, y, test_matrix):
    """Sample-level gate: softmax(W @ oof_pred_vec + b).
    Uses OOF predictions themselves as the gate input.
    Fit via regularized logistic-style softmax ridge.
    """
    m = ~np.isnan(y); K = oof_matrix.shape[1]
    P = oof_matrix[m]; yt = y[m]
    # Fit one ridge per expert as the gating score
    gate_scores_tr = np.zeros_like(P)
    for k in range(K):
        rcv = Ridge(alpha=10.0); rcv.fit(P, yt)
        gate_scores_tr[:, k] = rcv.predict(P)
    # Simple: use correlation with y as gate weight per sample
    # Gate: w_k(x) ∝ exp(corr(oof_k, y-hat))
    # Practical: fit a RidgeCV from oof_matrix → y, take coefficients as assay weights
    rcv_all = RidgeCV(alphas=np.logspace(-2, 4, 10))
    rcv_all.fit(P, yt)
    coefs = np.maximum(rcv_all.coef_, 0)
    if coefs.sum() < 1e-9: coefs = np.ones(K)
    coefs /= coefs.sum()
    # Apply to test
    return test_matrix @ coefs, coefs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ds = load_gdpa1(str(ROOT / "configs/default.yaml"))
    g3 = pd.read_csv(ROOT / "data/processed/gdpa3_table.csv")
    folds = ds.folds("cluster_isotype")
    fold_ids = sorted(set(folds.tolist()))
    rng = np.random.default_rng(42)

    esm_files = list((ROOT / "data/processed").glob("esm2_*.npz"))
    esm1 = None
    for ef in esm_files:
        arr = np.load(ef)["emb"]
        if arr.shape == (len(ds.df), 2560): esm1 = arr; break
    abl1 = np.load(ROOT / "data/processed/ablang2_gdpa1.npz")["emb"]
    esm3 = np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"]
    abl3 = np.load(ROOT / "data/processed/gdpa3_ablang2.npz")["emb"]

    desc_df1 = build_descriptor_matrix(ds.heavy, ds.light)
    desc_cols = list(desc_df1.columns)
    desc1 = np.nan_to_num(desc_df1.to_numpy(float))
    desc3 = np.nan_to_num(build_descriptor_matrix(g3["vh"], g3["vl"]).to_numpy(float))
    bio1, bio_cols = _bio(ROOT/"external/abdev-benchmark/data/features/processed_features/GDPa1",
                          ds.df["antibody_name"])
    bio3, _ = _bio(ROOT/"external/abdev-benchmark/data/features/processed_features/heldout_test",
                   g3["antibody_name"], cols=bio_cols)
    iso1 = _iso(ds.df); iso3 = _iso(g3)
    esm1n = np.nan_to_num(esm1); esm3n = np.nan_to_num(esm3)
    abl1n = np.nan_to_num(abl1); abl3n = np.nan_to_num(abl3)

    Xdb1 = np.hstack([desc1, bio1]); Xdb3 = np.hstack([desc3, bio3])
    Xplm1 = np.hstack([esm1n, abl1n]); Xplm3 = np.hstack([esm3n, abl3n])
    all_db_cols = desc_cols + list(bio_cols)
    N = len(ds.df)

    hdr = (f"{'Assay':10s}  {'E1':6s}  {'E2-A':6s}  {'E3-D':6s}  {'E4-F':6s}  "
           f"{'AW1-eq':6s}  {'AW2-lrn':7s}  {'AW3-dep':7s}  "
           f"{'CI-AW2':14s}  {'w-E1':5s} {'w-E2':5s} {'w-E3':5s} {'w-E4':5s}")
    print("\n" + hdr)
    print("=" * len(hdr))

    rows = []
    for assay, (g1_col, g3_col) in GENUINE_ASSAYS.items():
        y1 = ds.df[g1_col].to_numpy(float) if g1_col in ds.df.columns else np.full(N, np.nan)
        y3 = g3[g3_col].to_numpy(float) if g3_col in g3.columns else np.full(len(g3), np.nan)
        ok3 = ~np.isnan(y3); m1 = ~np.isnan(y1)

        iso_ok = assay not in ISO_INELIGIBLE
        single_blocks_tr = {"esm2": esm1n, "ablang2": abl1n, "desc": desc1, "bio": bio1}
        single_blocks_te = {"esm2": esm3n, "ablang2": abl3n, "desc": desc3, "bio": bio3}
        if iso_ok:
            single_blocks_tr["isotype"] = iso1
            single_blocks_te["isotype"] = iso3

        # ─── Step 1: OOF matrix (N × 4) ────────────────────────────────────
        oof = np.full((N, 4), np.nan)

        e1_exp = E1Expert(single_blocks_tr, single_blocks_te, folds, fold_ids, assay)
        e2_exp = MGFSExpert(Xdb1, Xdb3, all_db_cols, assay)
        e3_exp = DStackExpert(Xplm1, Xplm3, Xdb1, Xdb3, folds, fold_ids)
        e4_exp = FMoEExpert(Xdb1, Xdb3, all_db_cols, assay, folds, fold_ids)

        for f in fold_ids:
            tr_idx = np.where(folds != f)[0]
            te_idx = np.where(folds == f)[0]
            if len(te_idx) < 2 or len(tr_idx) < 10: continue

            # E1: inner block selection + OOF
            _, best_blk = e1_exp.fit_oof(tr_idx, y1)
            Xb = single_blocks_tr[best_blk]
            mi = m1[tr_idx]
            if mi.sum() >= 8:
                pipe_e1 = _head(Xb[tr_idx])
                pipe_e1.fit(Xb[tr_idx[mi]], y1[tr_idx[mi]])
                oof[te_idx, 0] = pipe_e1.predict(Xb[te_idx])

            # E2: MGFS
            cfg = MGFS_MASKS.get(assay, {"inc": [], "exc": []})
            mask2 = _mask_cols(all_db_cols, cfg["inc"], cfg["exc"])
            if mask2.sum() < 5: mask2 = np.ones(len(all_db_cols), bool)
            if mi.sum() >= 8:
                pipe_e2 = _head(Xdb1[tr_idx][:, mask2])
                pipe_e2.fit(Xdb1[tr_idx[mi]][:, mask2], y1[tr_idx[mi]])
                oof[te_idx, 1] = pipe_e2.predict(Xdb1[te_idx][:, mask2])

            # E3: D-stack (inner nested)
            inner_oof1 = np.zeros(len(tr_idx)); inner_oof2 = np.zeros(len(tr_idx))
            inner_folds = folds[tr_idx]; inner_ids = sorted(set(inner_folds))
            for f2 in inner_ids:
                itr = np.where((inner_folds != f2) & mi)[0]
                ite = np.where(inner_folds == f2)[0]
                if len(ite) < 2 or len(itr) < 8: continue
                pp1 = _head(Xplm1[tr_idx]); pp1.fit(Xplm1[tr_idx[itr]], y1[tr_idx[itr]]); inner_oof1[ite] = pp1.predict(Xplm1[tr_idx[ite]])
                pp2 = _head(Xdb1[tr_idx]);  pp2.fit(Xdb1[tr_idx[itr]],  y1[tr_idx[itr]]); inner_oof2[ite] = pp2.predict(Xdb1[tr_idx[ite]])
            meta_tr3 = np.column_stack([inner_oof1, inner_oof2])
            if mi.sum() >= 8:
                pp1f = _head(Xplm1[tr_idx]); pp1f.fit(Xplm1[tr_idx[mi]], y1[tr_idx[mi]])
                pp2f = _head(Xdb1[tr_idx]);  pp2f.fit(Xdb1[tr_idx[mi]],  y1[tr_idx[mi]])
                meta_te3 = np.column_stack([pp1f.predict(Xplm1[te_idx]),
                                             pp2f.predict(Xdb1[te_idx])])
                meta3 = Ridge(alpha=1.0); meta3.fit(meta_tr3[mi], y1[tr_idx[mi]])
                oof[te_idx, 2] = meta3.predict(meta_te3)

            # E4: F-MoE (inner)
            eligible = MOE_ELIGIBILITY.get(assay, list(MOE_EXPERTS.keys()))
            exp_oofs_in = []; exp_cvs = []
            for exp in eligible:
                kws = MOE_EXPERTS.get(exp, [])
                msk = _mask_cols(all_db_cols, kws)
                if msk.sum() < 3: continue
                Xe = Xdb1[:, msk]
                ioof = np.zeros(len(tr_idx)); iscores = []
                for f2 in inner_ids:
                    itr = np.where((inner_folds != f2) & mi)[0]
                    ite = np.where(inner_folds == f2)[0]
                    if len(ite) < 2 or len(itr) < 8: continue
                    pf = _head(Xe[tr_idx]); pf.fit(Xe[tr_idx[itr]], y1[tr_idx[itr]])
                    ioof[ite] = pf.predict(Xe[tr_idx[ite]])
                    iscores.append(spearmanr(y1[tr_idx[ite]], ioof[ite]).statistic)
                exp_oofs_in.append(ioof)
                exp_cvs.append(float(np.nanmean(iscores)) if iscores else -1.0)
            if exp_oofs_in and mi.sum() >= 8:
                cvs4 = np.maximum(np.array(exp_cvs), 0)
                tau4 = 0.1; ws4 = np.exp(cvs4 / tau4); ws4 /= ws4.sum()
                stk_in = np.column_stack(exp_oofs_in)
                # Fit meta on inner OOF
                meta4 = Ridge(alpha=1.0); meta4.fit(stk_in[mi], y1[tr_idx[mi]])
                # Get test predictions per expert
                test_preds4 = []
                for exp in eligible[:len(exp_oofs_in)]:
                    kws = MOE_EXPERTS.get(exp, [])
                    msk = _mask_cols(all_db_cols, kws)
                    if msk.sum() < 3: continue
                    Xe = Xdb1[:, msk]
                    pf2 = _head(Xe[tr_idx]); pf2.fit(Xe[tr_idx[mi]], y1[tr_idx[mi]])
                    test_preds4.append(pf2.predict(Xe[te_idx]))
                if test_preds4:
                    stk_te4 = np.column_stack(test_preds4[:len(exp_oofs_in)])
                    oof[te_idx, 3] = meta4.predict(stk_te4)

        # ─── Step 2: fit gating weights on OOF ──────────────────────────────
        m_oof = m1 & np.all(~np.isnan(oof), axis=1)
        P = oof[m_oof]; y_gate = y1[m_oof]
        K = 4
        w_eq = np.ones(K) / K
        w_learned = fit_aw_moe2(oof, y1, assay)

        # ─── Step 3: refit base experts on full GDPa1 ───────────────────────
        e1_exp.fit_full(y1)
        e2_exp.fit_full(y1)
        e3_exp.fit_full(y1)
        e4_exp.fit_full(y1)

        # ─── Step 4: test predictions ────────────────────────────────────────
        te1 = e1_exp.predict()
        te2 = e2_exp.predict()
        te3 = e3_exp.predict()
        te4 = e4_exp.predict()
        te_stack = np.column_stack([te1, te2, te3, te4])

        # AW-MoE-1: equal
        pred_aw1 = te_stack @ w_eq

        # AW-MoE-2: learned
        pred_aw2 = te_stack @ w_learned

        # AW-MoE-3: sample-dependent (OOF as gate input, test as features)
        # For test: we use the test expert predictions as the gate input
        pred_aw3, w3_coefs = predict_aw_moe3(oof, y1, te_stack)

        def rho_str(pred):
            if ok3.sum() < 5: return np.nan
            return float(spearmanr(y3[ok3], pred[ok3]).statistic)

        r1 = rho_str(te1); r2 = rho_str(te2); r3 = rho_str(te3); r4 = rho_str(te4)
        raw1 = rho_str(pred_aw1); raw2 = rho_str(pred_aw2); raw3 = rho_str(pred_aw3)
        ci2 = bca(y3, pred_aw2, rng)

        def f(v): return f"{v:+6.3f}" if not np.isnan(v) else "   nan"
        ci_s = f"[{ci2[0]:+.3f},{ci2[1]:+.3f}]" if not np.isnan(ci2[0]) else "              "

        print(f"{assay:10s}  {f(r1)}  {f(r2)}  {f(r3)}  {f(r4)}  "
              f"{f(raw1)}  {f(raw2):7s}  {f(raw3):7s}  "
              f"{ci_s}  "
              f"{w_learned[0]:.3f} {w_learned[1]:.3f} {w_learned[2]:.3f} {w_learned[3]:.3f}")
        print(f"{'':10s}  [E1={e1_exp.best_blk}] weights-AW2: "
              f"E1={w_learned[0]:.3f} A={w_learned[1]:.3f} D={w_learned[2]:.3f} F={w_learned[3]:.3f}")

        rows.append({"assay": assay,
                     "E1": round(r1,3), "A_MGFS": round(r2,3),
                     "D_stack": round(r3,3), "F_MoE": round(r4,3),
                     "AW_MoE1_eq": round(raw1,3),
                     "AW_MoE2_learned": round(raw2,3),
                     "AW_MoE3_dep": round(raw3,3),
                     "ci2_lo": round(ci2[0],3) if not np.isnan(ci2[0]) else None,
                     "ci2_hi": round(ci2[1],3) if not np.isnan(ci2[1]) else None,
                     "w_E1": round(w_learned[0],3), "w_A": round(w_learned[1],3),
                     "w_D": round(w_learned[2],3), "w_F": round(w_learned[3],3)})

    print("=" * len(hdr))
    df = pd.DataFrame(rows)
    print(f"\n--- Mean (5 genuine assays) ---")
    for col in ["E1","A_MGFS","D_stack","F_MoE","AW_MoE1_eq","AW_MoE2_learned","AW_MoE3_dep"]:
        v = df[col].mean()
        print(f"  {col:20s}: {v:.3f}")

    print(f"\n--- AW-MoE-2 learned weights per assay ---")
    for _, r in df.iterrows():
        print(f"  {r['assay']:10s}: E1={r['w_E1']:.3f}  A={r['w_A']:.3f}  "
              f"D={r['w_D']:.3f}  F={r['w_F']:.3f}")

    out = ROOT / "results/aw_moe.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
