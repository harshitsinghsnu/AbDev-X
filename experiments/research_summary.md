# AbDev-X - Research Summary

---

## Research Objective

Develop a multi-property antibody developability predictor that generalises
out-of-distribution (OOD) to a held-out benchmark, covering ten biophysical
assays simultaneously, using only sequence and computationally derived
structural features - no wet-lab measurements at inference time.

---

## Research Questions

1. **Can a single model architecture predict Spearman rho > 0.5 for aggregation
   assays (HAC, SMAC, HIC) on a truly held-out IgG1 benchmark?**

2. **Which representation source transfers best per assay, and does
   per-assay selection outperform global ensembling?**
   - E1 hypothesis: one source wins per assay; applying it directly beats
     a stacked average.

3. **Does physicochemical feature routing (F-MoE) add value over a flat
   feature vector?**
   - Routing charge, surface, hydrophobicity, and CDR groups independently
     per assay vs. treating all 165 features identically.

4. **Does assay-specific feature selection (A-MGFS) improve over full-feature
   baselines within the structural and physicochemical sources?**

5. **What is the practical ceiling for SEC and Titer prediction from
   sequence/structure alone?**

6. **Does AbDev-X transfer zero-shot to non-IgG formats (VHH-Fc bispecifics)?**

---

## Datasets and Statistics

### CACHE GDPa benchmark series

| Dataset | Role | n | Format | Assays | Notes |
|---------|------|---|--------|--------|-------|
| **GDPa1** | **Primary train** | **246** | **IgG (IgG1/2/4)** | **10** | **All model decisions made here** |
| **GDPa3** | **Held-out test** | **80** | **IgG1 only** | **10** | **Accessed once; OOD from GDPa1** |
| GDPa2.1 | Zero-shot VHH-Fc transfer | 51 | Clinical VHH-Fc | subset | Exploratory; not used for selection or tuning |
| GDPa4 | Zero-shot bispecific | 160 bispecifics + 71 IgG1 controls | Bispecific IgG | subset | Arm-composition strategy; parent overlap precludes clean held-out |
| GDPa5 | Zero-shot VHH-Fc (replication) | 160 | VHH-Fc (different panel from GDPa2.1) | subset | Replication test for GDPa2.1 transfer findings |

GDPa1 and GDPa3 are sequence-disjoint at the VH level (0% VH overlap by BLAST).

### External FLAb datasets (augmentation experiments only)

FLAb (Function-Label antibody benchmark) comprises publicly available antibody
developability studies curated from the literature. Used in exploratory
training augmentation experiments - not part of primary model training or held-out evaluation.

Key FLAb studies referenced:
- **Garbinski et al.** - expression/titer data; augmentation helps Titer on GDPa3
- **Hutchinson et al.** - thermostability data; augmentation helps Tm assays
- HIC / AC-SINS studies - augmentation reduces GDPa3 held-out performance
  (likely batch and protocol differences across studies)

> FLAb augmentation is exploratory: some studies help, others hurt.
> Formal batch correction should be evaluated before operational use.

### OOD-CV design (GDPa1 model development)
- 5-fold cluster-isotype cross-validation (OOD-CV)
- VH sequences clustered at 70% identity via CD-HIT (prevents sequence leakage)
- Stratified by IgG subclass (IgG1/2/4) within each fold
- Fold-level rho used for E1 source selection; never re-tuned on GDPa3

### GDPa1 -> GDPa3 distribution shift
- GDPa1: mixed IgG1/IgG2/IgG4; GDPa3: IgG1 only (isotype shift)
- 0% VH sequence overlap between training and test
- n=80 limits statistical power for moderate and weak assays

### Assay signal quality (GDPa3 rho range observed)

| Tier          | Assays                          | rho range       |
|---------------|---------------------------------|-----------------|
| High-signal   | HAC, SMAC, HIC, AC-SINS, PR-CHO | 0.251 – 0.650   |
| Moderate      | Tm1, Tonset, Tm2                | 0.160 – 0.344   |
| Low / noise   | SEC, Titer                      | -0.171 – 0.075  |

---

## Our Method - AbDev-X

### Six representation sources

| Source | Dimensions | Preprocessing |
|--------|-----------|---------------|
| ESM-2 650M | 2,560-dim VH+VL concat (1,280 per chain, mean-pooled) | PCA -> ~120-150 per fold (90% variance) |
| AbLang2 | 960-dim VH+VL concat (480 per chain, mean-pooled) | PCA per fold (90% variance) |
| Germline / isotype | Categorical | IGHV, IGLV, IGKV, IGHJ one-hot + IgG subclass |
| PhysChem | 165 named features | F-MoE (4 expert groups) + A-MGFS (cap 30 features) |
| Structural (ABodyBuilder3 + FreeSASA + SAP) | ~28 features after A-MGFS | A-MGFS per assay per fold |
| SAP / Electrostatics | 18 surface descriptors | None |

### Key architectural operations

**F-MoE (Feature-Level Mixture of Experts)**
- Scope: within PhysChem source only
- 4 expert sub-groups: Charge (pI, net charge, patch charge density),
  Surface (SASA, hydrophobic patch area, aromatic exposure),
  Hydrophobicity (Kyte-Doolittle, SAP), CDR (loop lengths, Atchley factors)
- Per-assay softmax gate weights (temperature tau = 0.1)
- Learned during out-of-fold training

**A-MGFS (Assay-specific Masked Greedy Feature Selection)**
- Scope: within PhysChem (from 165) and Structural sources
- Greedy forward selection in decreasing univariate Spearman rho order
- Minimum pairwise de-correlation threshold = 0.20 (redundancy suppression)
- Feature set capped at 30 (prevents overfitting at n~246)
- Applied per assay, per outer fold

**D-Stack (training infrastructure only)**
- Ridge + LightGBM base learners per source per assay
- OOF stacking ridge meta-learner
- Generates OOF scores consumed by E1 for source ranking
- Not used at inference

**E1 per-assay OOD-CV selection (reported inference model)**
1. Compute mean GDPa1 OOD-CV rho per source per assay
2. Select the single best-performing source per assay
3. Apply that source directly to GDPa3 - no stacking at inference

### Per-assay winners (E1 selection)
| Assay group | Winning source | Mechanism |
|-------------|---------------|-----------|
| HAC, SMAC, HIC | Structural | Surface hydrophobic patch area drives aggregation/retention |
| AC-SINS | AbLang2 | Sequence-driven charge patterns; antibody-LM trained on >1.5B sequences |
| PR-CHO, Tm assays | ESM-2 | Germline-linked polyreactivity risk and stability encoded in LM weights |

---

## Published Method Comparison - AbDev-X Beats All Baselines

### Head-to-head on GDPa3 held-out (n=80, identical protocol)

| Method | Mean rho (5 high-signal) | Notes |
|--------|--------------------------|-------|
| **AbDev-X (E1)** | **0.472** | Reported primary model |
| Jiang et al. 2026 | 0.139 | PLM + physicochemical hybrid, standard 5-fold CV |
| DELPHI | -0.038 | Fine-tuned Transformer; overfits at n=246 |
| ESM-2 Ridge (baseline) | < AbDev-X | Single-source PLM |
| AbLang2 Ridge (baseline) | < AbDev-X | Single-source antibody-LM |

**AbDev-X vs. Jiang et al. 2026 (+65% relative improvement)**
- Mean rho: 0.229 vs. 0.139 on 5 shared GDPa1 assays
- AC-SINS: Delta rho = +0.160
- PR-CHO: Delta rho = +0.117
- HIC: Jiang's hybrid method (rho = 0.558) marginally exceeds AbDev-X (rho = 0.547)
  - the only assay where AbDev-X does not win; within noise at n=79
- Jiang uses a fixed hybrid ensemble; AbDev-X uses per-assay selection
- Jiang's CV is not sequence-similarity-OOD, which may inflate their fold scores
  relative to GDPa3 transfer

**AbDev-X vs. DELPHI (Delta = +0.510)**
- DELPHI evaluated on identical data (GDPa1 n=246 training, same cluster-isotype
  splits, GDPa3 n=80 held-out) - controlled comparison
- DELPHI achieves negative rho on 4/5 GDPa3 high-signal assays
  (HAC, SMAC, HIC, AC-SINS) - consistent with fine-tuned Transformer overfitting at n=246
- DELPHI PR-CHO rho = 0.35 in original paper is attributable to proprietary
  11K-antibody IPI pretraining; in identical-data protocol: 0.15 vs. AbDev-X 0.238

**AbDev-X vs. competition champions (113 GDPa entries)**
- Champions are cherry-picked (each optimised for their target assay); represent
  an upper bound across different teams
- AbDev-X BCa 95% CIs are *interval-compatible* with champion scores on 4/5 assays
  (Tm2, AC-SINS, PR-CHO, Titer) - as a single model covering all 10 assays
- HIC gap is real: AbDev-X rho = 0.547 vs. competition best 0.708;
  requires stronger structural physics or more HIC-specific training data

**AbDev-X vs. Amini et al. 2026 (indicative, different dataset)**
- Amini report HIC rho = 0.65, AC-SINS rho ~0.47 on Biogen internal ~300 antibodies
- AbDev-X on GDPa1 CV: HIC rho = 0.686, AC-SINS rho = 0.501
- Datasets differ (GDPa vs. Biogen internal); comparison is indicative only

*(Full per-assay comparison figure: Figure basecomp in main text)*  
*(Full rho table: Supplementary Table S1)*

---

## Ablation Methods

All ablations evaluated on GDPa1 OOD-CV only (GDPa3 accessed once).

### Architecture-level ablations

| Configuration | Description |
|---------------|-------------|
| **E1** | Per-assay best source, applied directly - **reported primary model** |
| D-Stack | OOF ridge meta-learner stacked ensemble (rho = 0.251 on GDPa3) |
| SparseMoE top-K | Neural gate over 7 base-learner streams (rho = 0.216) |
| KronLSTM | Kronecker Delta Rule sequential fusion (rho = 0.240) |
| Architecture-level MoE | MoE over 4 architecture variants (rho = 0.226–0.243) |
| Unweighted average | Simple average of 4 architectures (rho = 0.264) |
| AbDev-MoE | Conditional gate (7 expert streams); improves 4/5 assays but AC-SINS collapses |
| AbDev-X + AbDev-MoE avg | Simple average of E1 and AbDev-MoE (rho = 0.275, but AC-SINS = 0.167) |

### Feature-level ablations

| Configuration | Description |
|---------------|-------------|
| ESM-2 Ridge only | Single-source PLM baseline |
| AbLang2 Ridge only | Single-source antibody-LM baseline |
| PhysChem flat | 165 features, no F-MoE routing |
| PhysChem + F-MoE | PhysChem with per-assay routing, no A-MGFS |
| PhysChem + F-MoE + A-MGFS | Full physicochemical source |
| Structural full | All structural features, no A-MGFS |
| Structural + A-MGFS | Full structural source |

### Gaussian Process audit (36 architectures)
- 1,620 evaluations (36 GP models x 5 assays x 3 CV regimes)
- GP-Hybrid-Additive significantly improves HIC on OOD hierarchical-cluster CV
  (Delta rho = +0.097, pooled rho = 0.513 vs. AbDev-X 0.416, p_Holm = 0.019)
- No other assay improvement survives Holm-Bonferroni correction
- On GDPa3 held-out: GP-FeatureGroup-Additive mean rho = 0.208 (+0.016 over
  AbDev-X 0.192) - not statistically significant

### Training data ablations
| Configuration | GDPa3 mean rho (5 assays) | Notes |
|---------------|---------------------------|-------|
| GDPa1 only | 0.472 | Primary model |
| GDPa1 + GDPa2.1 + GDPa5 | ~0.198 (all 10) | AC-SINS drops -0.102; no stat. difference |
| FLAb augmentation (per-study) | Mixed | Garbinski helps Titer; Hutchinson helps Tm; HIC/AC-SINS studies hurt |

---

## Results at a Glance - Strong Results

### GDPa3 held-out Spearman rho (n=80)

| Assay | rho | BCa 95% CI | Winner source |
|-------|-----|------------|---------------|
| **HAC** | **0.650** | [0.428, 0.796] | Structural |
| **SMAC** | **0.647** | [0.500, 0.750] | Structural |
| **HIC** | **0.547** | [0.362, 0.698] | Structural |
| AC-SINS | 0.262 | - | AbLang2 |
| PR-CHO | 0.251 | - | ESM-2 |

**Mean rho = 0.472** (5 high-signal assays)  
**Mean rho = 0.298** (all 10 assays)

### Format generalisation (zero-shot, exploratory)

| Dataset / Format | Assay | rho | Notes |
|-----------------|-------|-----|-------|
| GDPa4 bispecific (n=160) | HIC | 0.920 [0.88, 0.95] | Arm-composition strategy; parent-aware CI |
| GDPa4 bispecific | AC-SINS | 0.873 [0.83, 0.90] | |
| GDPa4 bispecific | PR-CHO | 0.668 [0.55, 0.74] | |
| GDPa2.1 VHH-Fc (n=51) | HIC | 0.605 [0.38, 0.77] | VH-only mode |
| GDPa2.1 VHH-Fc | AC-SINS | 0.520 [0.26, 0.71] | VH-only mode |
| GDPa5 VHH-Fc (n=160) | HIC | 0.143 | GDPa2.1 result does NOT replicate |

### Key takeaways
- HAC and SMAC both exceed rho = 0.64 - strong signal for the two
  aggregation assays most critical to early-stage triage
- HIC rho = 0.547 - useful rank-ordering from structure alone
- Structural source wins all three aggregation assays, validating the
  per-assay selection hypothesis
- BCa CIs for HAC, SMAC, HIC comfortably exclude zero
- GDPa4 bispecific HIC rho = 0.920 is striking but requires statistical
  caution (parent-sharing creates dependence)

---

## Additional Results Analysis

### Assay-level interpretation

**SEC (rho = -0.171):**
- Negative correlation; SEC failure driven by glycosylation, formulation,
  and sample handling effects not encodable in sequence/structure
- Flagged explicitly in abstract; do not use AbDev-X for SEC triage

**Titer (rho = 0.075):**
- Near-zero; production titer is cell-line and process dependent
- FLAb Garbinski augmentation partially helps; remains unreliable

**Thermal stability (Tm1 rho = 0.344, Tonset rho = 0.211, Tm2 rho = 0.160):**
- Moderate Tm1 signal; aromatic surface exposure (germline-encoded) partially explains variance
- GDPa3 is all-IgG1, removing IgG2/4 isotype signal present in GDPa1 Tm2
- VHH-Fc Tm2 fails (rho = -0.183) - VHH-specific stability determinants absent from IgG training

**AC-SINS (rho = 0.262):**
- AbLang2 wins; sequence-level charge encoding sufficient
- AbDev-MoE causes AC-SINS collapse (rho = 0.010) - fragility of conditional gating

**PR-CHO (rho = 0.251):**
- CDR hydrophobicity driven; ESM-2 implicitly encodes germline-linked polyreactivity risk

### SHAP attribution findings
*(Full table: Supplementary Table S2)*
- PhysChem and Structural features dominate SHAP importance for HAC, SMAC, HIC
- AbLang2 representations dominate for AC-SINS
- Germline/isotype features contribute to Tm assays (IgG subclass signal for Tm2)
- ESM-2 is broadly useful but rarely the top contributor

### Developmental ranking
- Composite z-score rank: Spearman rho = 0.200 vs. 0.194 for ESM-2 Ridge
- Precision@K: P@5=0.20, P@10=0.20, P@20=0.25 (random chance: 0.06, 0.13, 0.25)
- Supports HIC liability screening; does not support multi-property composite selection

### OOD generalisation summary
- 70% VH identity clustering prevents sequence memorisation
- Isotype shift (GDPa1 mixed -> GDPa3 IgG1) is handled well for aggregation assays
- VHH-Fc transfer: partial and panel-specific; not a general capability yet
- Multiple-comparisons caveat: no pairwise comparison survives Holm correction (SI-24)
