# AbDev-X — Paper Structure and Story

---

## Positioning

### What gap we fill

Most published developability predictors address **one assay at a time**
(typically HIC or SEC) using a single representation (sequence or structure)
and are evaluated on their own training cohorts without a truly held-out
benchmark. When multi-property models exist they either:

- use wet-lab measurements as inputs (not useful early-stage), or
- are evaluated on random splits (sequence leakage from CDR similarity), or
- report mean-across-assays performance that averages strong and noise signals

**Our gap:** a multi-property predictor that (a) uses only sequence +
computed structure, (b) is evaluated OOD (cluster-level held-out IgG1
benchmark GDPa3), and (c) is honest about which assays are and are not
predictable from sequence/structure.

### How we differ from prior work

| Axis | Prior work | AbDev-X |
|------|-----------|---------|
| Assay coverage | 1–4 assays typical | 10 assays |
| Evaluation | Random / IID split | OOD cluster-isotype split |
| Representation | Single source | 6 sources, per-assay selected |
| Transparency | Not all assays reported | Negative results (SEC, Titer) explicit |
| Feature routing | None | F-MoE within physicochemical |
| Feature selection | None or global | A-MGFS per assay |

### Audience

- Antibody engineers and developability scientists needing early triage tools
- ML/computational biologists benchmarking multi-property predictors
- Reviewers at BMC Bioinformatics (computational biology audience; methods
  rigour expected)

---

## The Story Arc

### Opening tension
Antibody drug development fails late because developability issues are caught
in Phase I/II, not at discovery. Early in silico triage could flag high-risk
molecules, but existing predictors cover only one or two assays, are evaluated
on memorisable train/test splits, and rarely disclose when prediction is
impossible from sequence alone.

### Our answer
AbDev-X introduces per-assay representation selection: instead of building one
universal feature vector and stacking it across all assays, we ask — *which
source of information transfers best for each specific assay?* The answer is
assay-specific. Aggregation (HAC, SMAC) is best predicted from structural
surface patches; polyreactivity (AC-SINS) from antibody language model
embeddings; hydrophobic retention (HIC) from structural hydrophobicity.
Selecting sources per assay, and applying the winner directly without
additional stacking at inference, is both simpler and better.

### The honest result
Not every developability property is predictable from sequence alone. SEC and
Titer show near-zero or negative ρ on the held-out benchmark. We report these
explicitly rather than omitting them — a more credible and practically useful
contribution than inflated mean-ρ numbers over hand-picked assays.

---

## Section-by-Section Structure

### Title
*AbDev-X: Per-Assay Representation Selection for Multi-Property Antibody
Developability Prediction Across Ten Biophysical Assays*

### Abstract (structure)
1. Background sentence — why developability prediction matters
2. Gap sentence — existing models are single-assay or evaluated IID
3. Method sentence — what AbDev-X does (E1 + six sources + F-MoE + A-MGFS)
4. Results sentence — ρ values for 5 high-signal assays on GDPa3
5. Caveat sentence — SEC and Titer are not predictable from sequence alone
6. Conclusion sentence — per-assay selection is a practical approach for
   early-stage triage

### 1. Background
- Antibody developability: what it is, why it matters, when in the pipeline
- What biophysical assays measure: HAC/SMAC (aggregation), HIC (hydrophobicity),
  AC-SINS (polyreactivity), SEC (size), Titer (expression)
- Prior computational approaches: structure-based, PLM-based, hybrid;
  limitations of single-assay and IID evaluation
- Motivation for a multi-source, multi-assay OOD approach

**Story beat:** *"We know the problem matters; the tools don't yet cover it.*"

### 2. Methods
Sequential subsections that mirror the architecture diagram:

**2.1 Datasets — GDPa1 and GDPa3**
- Cohort descriptions, assay definitions, IgG subclass distributions
- Why IgG1-only GDPa3 represents OOD generalisation

**2.2 Representation sources**
- ESM-2: extraction, concatenation, PCA compression
- AbLang2: analogous pipeline
- Germline/isotype: encoding
- PhysChem 165: feature engineering, F-MoE routing (4 expert groups,
  softmax gate τ=0.1), A-MGFS
- Structural: ABodyBuilder3, FreeSASA, SAP; A-MGFS
- SAP/Electrostatics: descriptor list

**2.3 Training infrastructure (D-Stack)**
- Ridge + LightGBM base learners per source
- OOF stacking meta-learner
- Role: generates OOF scores for E1 to consume; not an inference component

**2.4 E1 per-assay selection**
- OOD-CV protocol
- Scoring and selection logic
- Why direct application beats stacking at inference

**2.5 Statistical evaluation**
- Two primary endpoints vs ESM-2 Ridge and AbLang2 Ridge
- BCa bootstrap CIs
- SHAP attribution

**Story beat:** *"Methods are comprehensive but each component has a clear
purpose; the inference model is deliberately simple (best source, applied
directly)."*

### 3. Results
**3.1 GDPa3 held-out performance**
- Lead with the strong results: HAC ρ=0.650, SMAC ρ=0.647, HIC ρ=0.547
- Report the full 10-assay picture with BCa CIs
- Be explicit: SEC ρ=−0.171, Titer ρ=0.075 — these are not artefacts
- Table S1 in Supplementary; prose with key numbers in main text

**3.2 Comparison with baselines**
- ESM-2 Ridge and AbLang2 Ridge as primary comparators
- Per-assay deltas; where AbDev-X adds value and where a single PLM suffices
- Acknowledge: for some assays the single-source baseline is competitive

**3.3 Ablation — what each component contributes**
- E1 vs D-Stack vs equal-weight ensemble
- PhysChem flat vs + F-MoE vs + F-MoE + A-MGFS
- Structural full vs + A-MGFS
- Key message: per-assay selection > global stacking for OOD generalisation

**Story beat:** *"The results are real, appropriately bounded, and we understand
why each component helps."*

### 4. Discussion
**4.1 Why structural features win aggregation assays**
- Surface hydrophobic patches mechanistically drive HAC/SMAC/HIC
- ABodyBuilder3 + FreeSASA + SAP captures this; sequence-only PLMs don't

**4.2 Why PLMs win some assays**
- AbLang2 for AC-SINS: trained on antibody sequences, implicit charge encoding
- ESM-2 for PR-CHO: germline-linked polyreactivity risk embedded in LM weights

**4.3 SEC and Titer — the prediction ceiling**
- What sequence/structure cannot encode: glycosylation, formulation effects,
  cell-line dependent expression
- Practical implication: these assays need experimental data; do not rely on
  in silico triage

**4.4 Limitations**
- GDPa3 n=80 — power limited for weak assays
- IgG1-only test set — generalisation to IgG2/4 untested
- Computationally derived structures (ABodyBuilder3) vs experimental PDBs
- No therapeutic-sequence diversity outside IgG

**4.5 Future directions**
- Larger OOD benchmarks across IgG subclasses
- Incorporation of developability-optimised PLM pretraining
- Extension to bispecifics and non-IgG formats

**Story beat:** *"We know what works, why it works, and where the boundaries
are — a more useful contribution than an over-claimed result."*

### 5. Conclusions
- AbDev-X predicts aggregation and hydrophobicity developability OOD with
  practically useful accuracy
- Per-assay source selection is the key design choice
- SEC and Titer cannot currently be predicted reliably from sequence/structure
- Open benchmark (GDPa3) and code are available for community use

---

## Narrative Dos and Don'ts

| Do | Don't |
|----|-------|
| Lead every result with ρ values and CIs | Report only mean across all 10 |
| Explicitly discuss SEC and Titer failure | Bury or omit weak assays |
| Name the inference model precisely (E1) | Say "our model" ambiguously |
| Show D-Stack as training infrastructure | Imply stacking is used at test time |
| Position F-MoE as within PhysChem only | Describe it as a global router |
| Acknowledge n=80 power limitation | Overstate statistical confidence |

---

## Key Claims (ranked by strength)

1. **Strong, well-supported:**
   AbDev-X achieves ρ > 0.64 for HAC and SMAC on the held-out GDPa3 benchmark —
   the two most clinically relevant aggregation assays.

2. **Strong, well-supported:**
   Per-assay source selection (E1) outperforms both single-source baselines
   and global stacking on OOD evaluation.

3. **Moderate, directional:**
   F-MoE routing within PhysChem improves CV performance over flat feature vectors,
   particularly for surface-charge-dominated assays.

4. **Honest negative:**
   SEC and Titer are not reliably predictable from sequence/structure in this dataset;
   models reporting high SEC performance on IID splits should be interpreted with caution.
