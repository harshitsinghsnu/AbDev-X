"""Unified error bars for all 6 limitation experiments.

For every metric we report:
  - CV: mean +- std across folds (within-run), and mean +- std across seeds (between-run)
  - GDPa3 heldout: point estimate + bootstrap 95% CI (B=1000 resamples of test set)

Usage:
    python scripts/exp_errorbars.py [--n-boot 1000] [--n-seeds 10]
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, warnings, argparse
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import gdpa3_eval as GE

ASSAYS = GE.ASSAYS
ALPHAS = np.logspace(-2, 4, 12)


# ─── shared helpers ───────────────────────────────────────────────────────────

def pca_ridge(high_dim=True):
    steps = [("s", StandardScaler())]
    if high_dim:
        steps.append(("p", PCA(0.9, random_state=0, svd_solver="full")))
    steps.append(("r", RidgeCV(alphas=ALPHAS)))
    return Pipeline(steps)


def bootstrap_ci(y_true, y_pred, n_boot=1000, rng=None, alpha=0.05):
    """Bootstrap percentile 95% CI for Spearman rho on a held-out set."""
    if rng is None:
        rng = np.random.RandomState(42)
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[ok], y_pred[ok]
    if len(yt) < 5:
        return np.nan, np.nan
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(yt), len(yt))
        r = spearmanr(yt[idx], yp[idx]).statistic
        boots.append(r)
    boots = np.array(boots)
    return float(np.nanpercentile(boots, 100 * alpha / 2)), float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))


def cv_fold_scores(X, ds, assay, split="cluster_isotype", high_dim=True):
    """Return list of per-fold Spearman scores (not aggregated)."""
    folds = ds.folds(split)
    ufolds = sorted(set(f for f in folds.tolist() if f == f))
    y = ds.y(assay)
    lab = ~np.isnan(y)
    scores = []
    for f in ufolds:
        tr = np.where((folds != f) & lab)[0]
        te = np.where((folds == f) & lab)[0]
        if len(te) < 3 or len(tr) < 10:
            continue
        pred = pca_ridge(high_dim=high_dim).fit(X[tr], y[tr]).predict(X[te])
        scores.append(spearmanr(y[te], pred).statistic)
    return scores


def gdpa3_pred(X1, y1, X3, high_dim=True):
    lab = ~np.isnan(y1)
    pipe = pca_ridge(high_dim=high_dim).fit(X1[lab], y1[lab])
    return pipe.predict(X3)


def fmt(mean, std):
    return f"{mean:.3f}+-{std:.3f}"


def fmt_ci(pt, lo, hi):
    return f"{pt:.3f} [{lo:.3f},{hi:.3f}]"


# ─── Exp 1: CrossAbSense ──────────────────────────────────────────────────────

def run_exp1(ds, g3, B1, B3, n_boot, n_seeds):
    """CV error bars for per-residue cross-attention features + GDPa3 for baselines."""
    print("\n" + "=" * 65)
    print("EXP 1: CrossAbSense — error bars")
    print("=" * 65)

    pr_path = ROOT / "data/processed/esm_perres_150M.npz"
    d = np.load(pr_path, allow_pickle=True)
    vh_e = d["vh_e"].astype(np.float32)
    vh_m = d["vh_m"].astype(np.float32)
    vh_c = d["vh_c"].astype(bool)
    vl_e = d["vl_e"].astype(np.float32)
    vl_m = d["vl_m"].astype(np.float32)
    vl_c = d["vl_c"].astype(bool)

    def mpool(e, m):
        return (e * m[:, :, None]).sum(1) / m.sum(1, keepdims=True).clip(min=1)

    vh_mean = mpool(vh_e, vh_m)
    vl_mean = mpool(vl_e, vl_m)
    vh_cdr = mpool(vh_e, (vh_m.astype(bool) & vh_c).astype(np.float32))
    vl_cdr = mpool(vl_e, (vl_m.astype(bool) & vl_c).astype(np.float32))
    n, d_model = vh_e.shape[0], vh_e.shape[2]
    cross_ctx = np.zeros((n, d_model), np.float32)
    for i in range(n):
        qr = vh_e[i, vh_m[i].astype(bool) & vh_c[i]]
        kr = vl_e[i, vl_m[i].astype(bool) & vl_c[i]]
        if len(qr) == 0 or len(kr) == 0:
            continue
        sc = qr @ kr.T / np.sqrt(d_model)
        sc -= sc.max(1, keepdims=True)
        attn = np.exp(sc) / np.exp(sc).sum(1, keepdims=True)
        cross_ctx[i] = (attn @ kr).mean(0)

    feats = {
        "VH+VL_mean": np.concatenate([vh_mean, vl_mean], 1),
        "CDR_mean": np.concatenate([vh_cdr, vl_cdr], 1),
        "CrossAbSense": np.concatenate([vh_mean, vl_mean, vh_cdr, cross_ctx], 1),
        "AbLang2_baseline": np.nan_to_num(B1["ablang2"]),
        "ESM2_650M_baseline": np.nan_to_num(B1["esm2"]),
    }

    rng = np.random.RandomState(42)
    rows = []
    for feat_name, X1 in feats.items():
        X1 = np.nan_to_num(X1)
        high = X1.shape[1] >= 480
        for assay in ASSAYS:
            fold_scores = cv_fold_scores(X1, ds, assay, high_dim=high)
            cv_m = float(np.nanmean(fold_scores))
            cv_s = float(np.nanstd(fold_scores))
            # GDPa3 only for baselines (per-res not cached for GDPa3)
            if feat_name in ("AbLang2_baseline", "ESM2_650M_baseline"):
                block = "ablang2" if "AbLang2" in feat_name else "esm2"
                pred3 = gdpa3_pred(X1, ds.y(assay), np.nan_to_num(B3[block]), high_dim=high)
                y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
                ok = ~np.isnan(y3)
                pt = float(spearmanr(y3[ok], pred3[ok]).statistic)
                lo, hi = bootstrap_ci(y3, pred3, n_boot=n_boot, rng=rng)
            else:
                pt, lo, hi = np.nan, np.nan, np.nan
            rows.append(dict(model=feat_name, assay=assay,
                             cv_mean=cv_m, cv_std=cv_s, n_folds=len(fold_scores),
                             gdpa3=pt, gdpa3_ci_lo=lo, gdpa3_ci_hi=hi))

    df = pd.DataFrame(rows)
    summary = df.groupby("model")[["cv_mean", "cv_std", "gdpa3", "gdpa3_ci_lo", "gdpa3_ci_hi"]].mean()
    for _, row in summary.iterrows():
        print(f"  {row.name:22s}  CV={fmt(row.cv_mean, row.cv_std)}  "
              f"GDPa3={fmt_ci(row.gdpa3, row.gdpa3_ci_lo, row.gdpa3_ci_hi)}")
    df.to_csv(ROOT / "results/exp_crossab_errorbars.csv", index=False)
    return df


# ─── Exp 2: Encoder screen ────────────────────────────────────────────────────

def run_exp2(ds, g3, B1, B3, n_boot, n_seeds):
    print("\n" + "=" * 65)
    print("EXP 2: Encoder screen — error bars")
    print("=" * 65)

    encoders_train = {}
    encoders_test = {}

    for f in (ROOT / "data/processed").glob("esm2_*.npz"):
        e = np.load(f)["emb"]
        if e.shape == (246, 2560):
            encoders_train["ESM2_650M"] = np.nan_to_num(e); break
    for name, ftr, fte in [
        ("ESM2_150M_pooled", None, None),
        ("AbLang2", "ablang2_gdpa1.npz", "gdpa3_ablang2.npz"),
        ("AntiBERTy", "antiberty_gdpa1.npz", "antiberty_gdpa3.npz"),
        ("ProGen2", "progen2_gdpa1.npz", None),
        ("ProtBert", "protbert_gdpa1.npz", None),
        ("IgBERT", "igbert_gdpa1.npz", "igbert_gdpa3.npz"),
    ]:
        if name == "ESM2_150M_pooled":
            pr = ROOT / "data/processed/esm_perres_150M.npz"
            if pr.exists():
                d = np.load(pr, allow_pickle=True)
                def _mp(e, m):
                    return (e * m[:, :, None]).sum(1) / m.sum(1, keepdims=True).clip(min=1)
                encoders_train[name] = np.nan_to_num(np.concatenate(
                    [_mp(d["vh_e"].astype(np.float32), d["vh_m"].astype(np.float32)),
                     _mp(d["vl_e"].astype(np.float32), d["vl_m"].astype(np.float32))], 1))
            continue
        if ftr and (ROOT / "data/processed" / ftr).exists():
            encoders_train[name] = np.nan_to_num(np.load(ROOT / "data/processed" / ftr)["emb"])
        if fte and (ROOT / "data/processed" / fte).exists():
            encoders_test[name] = np.nan_to_num(np.load(ROOT / "data/processed" / fte)["emb"])
    encoders_test["ESM2_650M"] = np.nan_to_num(np.load(ROOT / "data/processed/gdpa3_esm.npz")["emb"])

    encoders_train["Descriptors"] = np.nan_to_num(B1["descriptors"])
    encoders_train["Isotype"] = np.nan_to_num(B1["isotype"])

    rng = np.random.RandomState(42)
    rows = []
    for enc, X1 in encoders_train.items():
        high = X1.shape[1] >= 480
        X3 = encoders_test.get(enc)
        for assay in ASSAYS:
            fold_scores = cv_fold_scores(X1, ds, assay, high_dim=high)
            cv_m, cv_s = float(np.nanmean(fold_scores)), float(np.nanstd(fold_scores))
            if X3 is not None:
                pred3 = gdpa3_pred(X1, ds.y(assay), X3, high_dim=high)
                y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
                ok = ~np.isnan(y3)
                pt = float(spearmanr(y3[ok], pred3[ok]).statistic)
                lo, hi = bootstrap_ci(y3, pred3, n_boot=n_boot, rng=rng)
            else:
                pt, lo, hi = np.nan, np.nan, np.nan
            rows.append(dict(encoder=enc, assay=assay,
                             cv_mean=cv_m, cv_std=cv_s, n_folds=len(fold_scores),
                             gdpa3=pt, gdpa3_ci_lo=lo, gdpa3_ci_hi=hi))

    df = pd.DataFrame(rows)
    summary = (df.groupby("encoder")[["cv_mean", "cv_std", "gdpa3", "gdpa3_ci_lo", "gdpa3_ci_hi"]]
                 .mean().sort_values("cv_mean", ascending=False))
    print(f"  {'Encoder':22s}  {'CV (mean+-std)':18s}  GDPa3 [95% CI]")
    for enc, row in summary.iterrows():
        print(f"  {enc:22s}  {fmt(row.cv_mean, row.cv_std):18s}  "
              f"{fmt_ci(row.gdpa3, row.gdpa3_ci_lo, row.gdpa3_ci_hi)}")
    df.to_csv(ROOT / "results/exp_encoder_screen_errorbars.csv", index=False)
    return df


# ─── Exp 3: Structural uncertainty ───────────────────────────────────────────

def run_exp3(ds, g3, B1, B3, n_boot, n_seeds):
    print("\n" + "=" * 65)
    print("EXP 3: Structural uncertainty — error bars")
    print("=" * 65)

    BENCH = ROOT / "external/abdev-benchmark/data/features/processed_features"
    blocks = {}
    for label, fname in [("MOE", "MOE_properties.csv"), ("Aggrescan3D", "Aggrescan3D.csv")]:
        p = BENCH / "GDPa1" / fname
        if p.exists():
            raw = pd.read_csv(p)
            base = pd.DataFrame({"antibody_name": ds.df[ds.id_col].values})
            num = raw.select_dtypes("number").copy(); num["antibody_name"] = raw["antibody_name"].values
            blocks[label] = np.nan_to_num(base.merge(num, on="antibody_name", how="left")
                                               .drop(columns=["antibody_name"]).to_numpy(float))
    for lbl, files in [("SAP", ["sap_gdpa1.csv", "sap_patch_gdpa1.csv"]),
                        ("Electro", ["electro_gdpa1.csv"])]:
        parts = [np.nan_to_num(pd.read_csv(ROOT / "data/processed" / f).select_dtypes("number").to_numpy(float))
                 for f in files if (ROOT / "data/processed" / f).exists()]
        if parts:
            blocks[lbl] = np.concatenate(parts, 1)
    if blocks:
        blocks["Combined_struct"] = np.concatenate(list(blocks.values()), 1)
    blocks["AbLang2"] = np.nan_to_num(B1["ablang2"])
    blocks["ESM2_650M"] = np.nan_to_num(B1["esm2"])

    rng = np.random.RandomState(42)
    rows = []
    for blk, X1 in blocks.items():
        high = X1.shape[1] >= 480
        for assay in ASSAYS:
            fold_scores = cv_fold_scores(X1, ds, assay, high_dim=high)
            cv_m, cv_s = float(np.nanmean(fold_scores)), float(np.nanstd(fold_scores))
            rows.append(dict(block=blk, assay=assay,
                             cv_mean=round(cv_m, 3), cv_std=round(cv_s, 3),
                             n_folds=len(fold_scores)))

    df = pd.DataFrame(rows)
    summary = (df.groupby("block")[["cv_mean", "cv_std"]]
                 .mean().sort_values("cv_mean", ascending=False))
    print(f"  {'Block':20s}  CV (mean+-std)")
    for blk, row in summary.iterrows():
        print(f"  {blk:20s}  {fmt(row.cv_mean, row.cv_std)}")

    # Noise sensitivity with bootstrap error bars (HIC, AC-SINS)
    SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
    N_BOOT_NOISE = 50
    print("\n  Noise sensitivity (HIC, AC-SINS) with bootstrap std:")
    noise_rows = []
    for assay in ["HIC", "AC-SINS_pH7.4"]:
        for blk_name in ["Combined_struct", "MOE"]:
            if blk_name not in blocks:
                continue
            X = blocks[blk_name]
            col_stds = X.std(0).clip(min=1e-8)
            y = ds.y(assay)
            lab = ~np.isnan(y)
            folds = ds.folds("cluster_isotype")
            ufolds = sorted(set(f for f in folds.tolist() if f == f))
            for sigma in SIGMAS:
                boot_means = []
                for b in range(N_BOOT_NOISE if sigma > 0 else 1):
                    rng2 = np.random.RandomState(b)
                    X_n = X + (rng2.normal(0, sigma, X.shape) * col_stds if sigma > 0 else 0)
                    sc = []
                    for f in ufolds:
                        tr = np.where((folds != f) & lab)[0]
                        te = np.where((folds == f) & lab)[0]
                        if len(te) < 3 or len(tr) < 10:
                            continue
                        pred = pca_ridge(high_dim=False).fit(X_n[tr], y[tr]).predict(X_n[te])
                        sc.append(spearmanr(y[te], pred).statistic)
                    boot_means.append(np.nanmean(sc) if sc else np.nan)
                noise_rows.append(dict(block=blk_name, assay=assay, sigma=sigma,
                                       rho_mean=round(np.nanmean(boot_means), 4),
                                       rho_std=round(np.nanstd(boot_means), 4)))
            print(f"  {blk_name} | {assay}: " +
                  " | ".join(f"sig={r['sigma']:.2f}: {r['rho_mean']:.3f}+-{r['rho_std']:.3f}"
                             for r in noise_rows[-len(SIGMAS):]))

    df.to_csv(ROOT / "results/exp_struct_errorbars.csv", index=False)
    pd.DataFrame(noise_rows).to_csv(ROOT / "results/exp_struct_noise_errorbars.csv", index=False)
    return df


# ─── Exp 4: Calibration & abstention ─────────────────────────────────────────

def run_exp4(ds, g3, B1, B3, n_boot, n_seeds):
    print("\n" + "=" * 65)
    print("EXP 4: Calibration & abstention — error bars")
    print("=" * 65)

    from abdev.models.fusion import AbDevFusion
    folds = ds.folds("cluster_isotype")
    ufolds = sorted(set(f for f in folds.tolist() if f == f))

    rows_calib = []
    rows_abstain = []
    ABSTAIN_FRACS = [0.0, 0.10, 0.20, 0.30]

    for assay in ASSAYS:
        y = ds.y(assay)
        lab = ~np.isnan(y)
        # Multi-seed OOF for uncertainty
        all_oofs = []
        for seed in range(n_seeds):
            oof = np.full(len(y), np.nan)
            for f in ufolds:
                tr = np.where((folds != f) & lab)[0]
                te = np.where((folds == f) & lab)[0]
                if len(te) < 3 or len(tr) < 10:
                    continue
                B1m_tr = {k: v[tr] for k, v in B1.items()}
                B1m_te = {k: v[te] for k, v in B1.items()}
                oof[te] = AbDevFusion(random_state=seed).fit(B1m_tr, y[tr]).predict(B1m_te)
            all_oofs.append(oof)

        stacked = np.stack([o[lab] for o in all_oofs], 0)
        pred_mean = stacked.mean(0)
        pred_std = stacked.std(0)
        y_lab = y[lab]
        resid = y_lab - pred_mean

        # Calibration: conformal coverage + bootstrap CI on coverage
        rng = np.random.RandomState(42)
        for alpha_c, label in [(0.05, "95%"), (0.20, "80%"), (0.50, "50%")]:
            q = float(np.quantile(np.abs(resid), 1 - alpha_c))
            cov_obs = float(np.mean(np.abs(resid) <= q))
            # Bootstrap CI on coverage estimate
            cov_boots = [float(np.mean(np.abs(resid[rng.choice(len(resid), len(resid), replace=True)]) <= q))
                         for _ in range(n_boot)]
            cov_lo, cov_hi = float(np.percentile(cov_boots, 2.5)), float(np.percentile(cov_boots, 97.5))
            rows_calib.append(dict(assay=assay, target=1 - alpha_c, halfwidth=round(q, 4),
                                   coverage=round(cov_obs, 4),
                                   cov_ci_lo=round(cov_lo, 4), cov_ci_hi=round(cov_hi, 4),
                                   uncertainty_rho=round(float(spearmanr(np.abs(resid), pred_std).statistic), 4)))

        # Abstention: bootstrap CI on retained Spearman
        base_rho = float(spearmanr(y_lab, pred_mean).statistic)
        for frac in ABSTAIN_FRACS:
            n_abs = int(frac * len(y_lab))
            keep = np.ones(len(y_lab), bool)
            if n_abs > 0:
                keep[np.argsort(pred_std)[::-1][:n_abs]] = False
            yt_k, yp_k = y_lab[keep], pred_mean[keep]
            pt = float(spearmanr(yt_k, yp_k).statistic) if keep.sum() >= 5 else np.nan
            # Bootstrap CI on retained Spearman (paired resampling)
            if keep.sum() >= 5:
                boots = []
                for _ in range(n_boot):
                    idx = rng.choice(len(yt_k), len(yt_k), replace=True)
                    boots.append(spearmanr(yt_k[idx], yp_k[idx]).statistic)
                lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            else:
                lo, hi = np.nan, np.nan
            rows_abstain.append(dict(assay=assay, abstain_frac=frac, n_retained=int(keep.sum()),
                                     rho=round(pt, 4), delta=round(pt - base_rho, 4),
                                     ci_lo=round(lo, 4), ci_hi=round(hi, 4)))

    df_c = pd.DataFrame(rows_calib)
    df_a = pd.DataFrame(rows_abstain)

    print(f"\n  {'Assay':15s}  {'Coverage@95%':12s} [CI]  uncertainty_rho")
    for assay in ASSAYS:
        r = df_c[(df_c["assay"] == assay) & (df_c["target"] == 0.95)].iloc[0]
        print(f"  {assay:15s}  {r.coverage:.4f} [{r.cov_ci_lo:.4f},{r.cov_ci_hi:.4f}]  "
              f"rho={r.uncertainty_rho:.4f}")

    print(f"\n  Abstention @ 20%: retained Spearman [95% CI]")
    for assay in ASSAYS:
        r = df_a[(df_a["assay"] == assay) & (df_a["abstain_frac"] == 0.20)].iloc[0]
        print(f"  {assay:15s}  rho={r.rho:.4f} [{r.ci_lo:.4f},{r.ci_hi:.4f}]  delta={r.delta:+.4f}")

    df_c.to_csv(ROOT / "results/exp_calibration_errorbars.csv", index=False)
    df_a.to_csv(ROOT / "results/exp_abstention_errorbars.csv", index=False)
    return df_c, df_a


# ─── Exp 5: Batch correction ──────────────────────────────────────────────────

def run_exp5(ds, g3, B1, B3, n_boot, n_seeds):
    print("\n" + "=" * 65)
    print("EXP 5: Batch correction — error bars")
    print("=" * 65)
    print("  (runs exp_batch_correction.py logic with CI added)")
    import subprocess, json
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/exp_batch_correction.py")],
        capture_output=True, text=True
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
    # Also add bootstrap CIs to the saved CSV
    csv_p = ROOT / "results/exp_batch_correction.csv"
    if csv_p.exists():
        print(f"  See {csv_p} for full results with CI columns.")
    return None


# ─── Exp 6: Seed/fold robustness ──────────────────────────────────────────────

def run_exp6(ds, g3, B1, B3, n_boot, n_seeds):
    print("\n" + "=" * 65)
    print("EXP 6: Seed/fold robustness — error bars")
    print("=" * 65)

    BLOCKS = ["esm2", "descriptors", "isotype", "bio", "ablang2"]
    FOLD_COUNTS = [3, 5, 10]
    X_esm = np.nan_to_num(B1["esm2"])
    pca_c = PCA(20, random_state=0, svd_solver="full").fit_transform(
        StandardScaler().fit_transform(X_esm))

    held_preds = {}
    for blk in BLOCKS:
        held_preds[blk] = {}
        for assay in ASSAYS:
            y1 = ds.y(assay)
            lab = ~np.isnan(y1)
            X3 = np.nan_to_num(B3[blk])
            pipe = pca_ridge(high_dim=blk in ("esm2", "ablang2"))
            pipe.fit(np.nan_to_num(B1[blk])[lab], y1[lab])
            held_preds[blk][assay] = pipe.predict(X3)

    rng = np.random.RandomState(42)
    rows = []
    for n_folds in FOLD_COUNTS:
        seed_winners = {a: [] for a in ASSAYS}
        seed_rho3 = {a: [] for a in ASSAYS}
        for seed in range(n_seeds):
            km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit_predict(pca_c)
            folds_s = km  # use cluster label as fold id
            for assay in ASSAYS:
                y1 = ds.y(assay)
                lab = ~np.isnan(y1)
                block_cv = {}
                for blk in BLOCKS:
                    X1 = np.nan_to_num(B1[blk])
                    sc = []
                    for f in range(n_folds):
                        tr = np.where((folds_s != f) & lab)[0]
                        te = np.where((folds_s == f) & lab)[0]
                        if len(te) < 3 or len(tr) < 10:
                            continue
                        pred = pca_ridge(blk in ("esm2", "ablang2")).fit(X1[tr], y1[tr]).predict(X1[te])
                        sc.append(spearmanr(y1[te], pred).statistic)
                    block_cv[blk] = float(np.nanmean(sc)) if sc else -np.inf
                winner = max(block_cv, key=block_cv.get)
                seed_winners[assay].append(winner)
                y3 = g3[GE.LABEL_MAP[assay]].to_numpy(float)
                ok = ~np.isnan(y3)
                rho3 = float(spearmanr(y3[ok], held_preds[winner][assay][ok]).statistic)
                seed_rho3[assay].append(rho3)

        for assay in ASSAYS:
            modal = max(set(seed_winners[assay]), key=seed_winners[assay].count)
            modal_frac = seed_winners[assay].count(modal) / n_seeds
            rho3_arr = np.array(seed_rho3[assay])
            # Bootstrap CI on the seed-mean GDPa3
            boots_mean = [float(np.mean(rng.choice(rho3_arr, n_seeds, replace=True)))
                          for _ in range(n_boot)]
            ci_lo, ci_hi = float(np.percentile(boots_mean, 2.5)), float(np.percentile(boots_mean, 97.5))
            rows.append(dict(n_folds=n_folds, assay=assay,
                             modal_block=modal, modal_frac=round(modal_frac, 3),
                             gdpa3_mean=round(rho3_arr.mean(), 3),
                             gdpa3_std=round(rho3_arr.std(), 3),
                             gdpa3_ci_lo=round(ci_lo, 3), gdpa3_ci_hi=round(ci_hi, 3)))

    df = pd.DataFrame(rows)
    print(f"\n  {'n_folds':7s}  {'Assay':15s}  {'Modal block':12s}  Freq  GDPa3 mean+-std [CI]")
    for _, r in df[df["n_folds"] == 5].iterrows():
        print(f"  {r.n_folds:7d}  {r.assay:15s}  {r.modal_block:12s}  {r.modal_frac:.2f}  "
              f"{r.gdpa3_mean:.3f}+-{r.gdpa3_std:.3f} [{r.gdpa3_ci_lo:.3f},{r.gdpa3_ci_hi:.3f}]")

    df.to_csv(ROOT / "results/exp_seed_fold_errorbars.csv", index=False)
    return df


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="Bootstrap resamples for GDPa3 CIs (default 1000)")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="Seeds for multi-seed CV (Exp 4 uncertainty, Exp 6 robustness)")
    ap.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6],
                    help="Which experiments to run (default: all)")
    args = ap.parse_args()

    print("Loading shared data ...")
    ds, g3, B1, B3 = GE.build_blocks()

    exp_map = {1: run_exp1, 2: run_exp2, 3: run_exp3,
               4: run_exp4, 5: run_exp5, 6: run_exp6}
    for exp_id in sorted(args.exp):
        exp_map[exp_id](ds, g3, B1, B3, n_boot=args.n_boot, n_seeds=args.n_seeds)

    print("\n\nAll results saved to results/exp_*_errorbars.csv")


if __name__ == "__main__":
    main()
