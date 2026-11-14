# AbDev-X — Revision Plan (Major Revision)

Priority order: **Critical fix → Major methodological → Major writing → Moderate**

---

## STATUS LEGEND
- [x] DONE
- [ ] TODO
- [~] PARTIAL

---

## CRITICAL — Fix before any resubmission

### [x] Issue 6: Arithmetic error in competition comparison (§3.3)
**Error:** Paper claimed "four of five" competition champion scores fall inside AbDev-X CIs.
**Actual count using Table S1 values:**

| Assay | Champion | AbDev-X CI | Inside? |
|-------|----------|-----------|---------|
| HIC | 0.708 | [0.362, 0.698] | NO |
| Tm2 | 0.392 | [-0.063, 0.371] | NO (0.392 > 0.371) |
| AC-SINS | 0.356 | [0.031, 0.476] | YES |
| PR-CHO | 0.337 | [0.017, 0.443] | YES |
| Titer | 0.310 | [-0.147, 0.293] | NO (0.310 > 0.293) |

**Fix applied:** Changed to "two of five assays (AC-SINS, PR-CHO)" and added
explicit sentences noting that Tm2 and Titer champion scores fall outside CIs. ✓

---

## MAJOR — Methodological/scientific accuracy

### [ ] Issue 1: Residual model-naming inconsistency (manuscript-wide)
Check for and remove remaining instances of:
- "Full stacked AbDev-X" — replace with "D-Stack configuration"
- "AbDev-X stacking" — replace with "D-Stack"
- "Four-component AW-MoE pipeline" — remove; AW-MoE was already deleted
- "Single unified model trained on all assays" — rephrase as "single model
  evaluated on all assays simultaneously" (does not imply shared training objective)
- Any equation or figure caption still implying equal-weight ensemble

**Action:** `grep -n "stack\|MoE\|unified" bmc_paper.tex` and review each hit.

### [ ] Issue 3: Overlap between structural source (5) and SAP/electrostatics (6)
Sources 5 and 6 both derive from ABodyBuilder3 structures.
The paper must clarify that SAP/electrostatics (source 6) is a curated subset
of the same structural calculations used in source 5 — not independently derived.
Their contributions are not independent biological signals.

**Action required:**
- Add a sentence in §2.2 stating that both sources derive from the same
  ABodyBuilder3 structures and that source 6 is a curated, lightweight subset
- Qualify any "independent biological contribution" language in the Discussion

### [ ] Issue 4: D-Stack OOF matrix — 7-column inconsistency
The paper says there are 6 sources × 2 base learners, then describes an
OOF matrix in R^{n×7}. The extra column is unexplained.
**Likely cause:** the physicochemical source (4) contributes two columns
(Ridge head and LightGBM head), while other sources contribute one each.
The structural source (5) may also have a separate Ridge and LGBM column.

**Action required:**
- Add a table or equation in §2.3 explicitly defining all 7 columns:
  e.g. [ESM2_R, AbLang2_R, Iso_R, PhysChem_R, PhysChem_LGBM, Struct_R, SAP_R]
  (or whatever the actual column mapping is)
- Reconcile "six sources", "six base-learner streams", "seven base-learner
  predictions" into one consistent description

### [ ] Issue 5: OOD-CV terminology precision
Current: "OOD-CV" used to describe cluster-stratified CV.
Problem: "OOD" implies leave-out of a class; IgG subclass stratification
distributes all subclasses across folds (not leave-one-isotype-out).

**Action required:**
- Replace "OOD-CV" with "cluster-stratified CV" in the Methods
  (keep OOD-CV as shorthand but define it precisely on first use)
- Add one sentence: "Stratification ensures each fold contains IgG1, IgG2
  and IgG4 antibodies; the out-of-distribution aspect refers to VH cluster
  separation, not isotype held-out."
- Explicitly state that the GDPa1→GDPa3 isotype shift (mixed→IgG1 only)
  is not captured by this CV design

### [ ] Issue 7: Jiang comparison — mixed evaluation settings
The comparison of AbDev-X rho=0.547 (HIC) vs Jiang 0.558 mixes:
- 0.547 = GDPa3 held-out result
- Jiang values are from their GDPa1 5-fold CV paper table

**Action required:**
- Separate the comparison into two clearly labelled parts:
  (a) GDPa1 OOD-CV AbDev-X vs. Jiang's reported GDPa1 CV values
  (b) GDPa3 held-out AbDev-X results (cannot be directly compared to Jiang,
      who does not report GDPa3 results)
- Distinguish "five shared assays (same as Jiang)" from "five high-signal assays"
  — these are not necessarily the same set

### [ ] Issue 8: DELPHI comparison — PR-CHO value inconsistency
Paper reports AbDev-X PR-CHO = 0.251 (primary result) but DELPHI paragraph
compares against 0.238. The 0.238 value is unexplained.

**Action required:**
- Identify which configuration produced 0.238 (likely D-Stack or reconstructed
  feature variant); state it explicitly
- Soften "DELPHI's published PR-CHO is fully attributable to its proprietary
  pretraining corpus" → "the performance difference is consistent with a
  contribution from additional pretraining data, although training protocols differ"

### [ ] Issue 9: Add configuration reference table (§3, beginning)
Readers cannot track which results come from which model variant.

**Add this table near the start of Results:**

| Configuration | Description | Where reported |
|---------------|-------------|----------------|
| AbDev-X (primary / E1) | Per-assay source selection, direct inference | All GDPa3 results |
| D-Stack | OOF dual-stream ridge meta-learner | §3.5 ablation |
| F-MoE | Feature-level expert gating (within PhysChem) | §3.x ablation |
| A-MGFS | Assay-specific greedy feature selection | §3.x ablation |
| Reconstructed feature model | ESM-2 excluded, 480 features | §3.8 GP comparison |

### [ ] Issue 10: GDPa4 bispecific — correct the "zero-shot" framing
GDPa4 bispecifics are assembled from GDPa1 parental antibodies.
The parental sequences were in the AbDev-X training set.
This is NOT zero-shot generalization to unseen sequences.

**Action required:**
- Remove "zero-shot" from GDPa4 bispecific discussion
- Replace with: "exploratory arm-composition test on bispecific constructs
  with parental antibodies present in the training set"
- Acknowledge that the high HIC correlation (0.920) may reflect the
  approximately additive nature of HIC on this specific panel, not
  general bispecific generalization capability

### [ ] Issue 11: VHH-Fc "VH-only mode" — add method detail
The paper says "VH-only mode" for GDPa2.1/GDPa5 transfer but does not
explain what happens to VL-dependent features.

**Action required in Methods §2.6 or §3.7:**
- State explicitly: which of the 6 sources are affected by VH-only input
  (ESM-2, AbLang2, germline: use VH input only; structural: VH-only structure
  prediction; PhysChem, SAP/Elec: VH chain features only)
- State whether a zero/placeholder VL representation was used or VL was dropped
- Clarify which features become undefined and how they are handled (zeroed? imputed?)

---

## MAJOR — Writing clarity

### [ ] Issue 2: Abstract method description — add three explicit points
The Abstract should state:
1. Six representation sources are constructed
2. One source is selected per assay using GDPa1 OOD-CV
3. The selected source is applied directly to GDPa3 without stacking

Also: define "five high-signal assays" before presenting results, not
as an apparent post hoc grouping. Add in §2.1 or at start of §3.1.

### [ ] Issue 12: SHAP terminology — soften "validates"
Replace:
- "validates the biological mechanism" → "provides biologically plausible
  associations consistent with known mechanisms"
- "confirms the mechanism" → "is consistent with known assay drivers"
- Use "predictive attribution" (SHAP on LightGBM proxy model)
- Use "residue-level proxy attribution" (ESM-2 proxy model)
- Use "structure-mapped feature visualisation" (surface maps)

### [ ] Issue 13: Mechanism interpretation — add evidence-level distinctions
In Discussion, distinguish:
- Mechanistic hypotheses from literature (with citation)
- Features selected by the model
- Features associated with predictions
- Experimentally validated mechanisms

Specifically cautious areas:
- Exposed aromatic residues → thermal stability: present as "consistent with"
  not "explains"
- SEC failure: frame as "consistent with the hypothesis that..." not causal
- Titer: same caution

---

## MODERATE — Polish and structural issues

### [ ] Background §1, page 3: ridge stacking claim
Background says AbDev-X "combines representations through a Ridge stacking
meta-learner" — this describes D-Stack, not the primary E1 model.
Fix: describe AbDev-X as selecting representations per assay;
mention stacking only as a training-time component.

### [ ] Figure 2 label "AbDev-X as stacking"
If Figure 2 labels AbDev-X with a stacking architecture diagram,
relabel to match E1 as the primary.

### [ ] "Fully attributable" language (DELPHI, §3.3)
See Issue 8. Soften.

### [ ] Holm correction claim (SI-24)
Paper states: "No individual pairwise comparison survives Holm correction."
Ensure this is stated consistently in Results, not only in Limitations.

---

## Summary — estimated effort

| Priority | Issues | Effort |
|----------|--------|--------|
| Critical (fix now) | 1 (✓ done) | Complete |
| Major methodological | 7 issues | ~2–3 days of revisions |
| Major writing | 3 issues | ~1 day |
| Moderate | 5 issues | ~0.5 day |

---

## First action items for next working session

1. Run `grep -n "stack\|MoE\|unified\|mixture" bmc_paper.tex` — audit every hit
2. Add configuration table to §3 (Issue 9)
3. Restructure §2.2 with six short per-source subsections (Issue 3)
4. Fix Jiang comparison to clearly separate GDPa1 CV vs. GDPa3 held-out (Issue 7)
5. Add precise D-Stack OOF matrix definition with column labels (Issue 4)
6. Fix GDPa4 bispecific framing (Issue 10)
