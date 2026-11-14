"""Published reference numbers for the research-article comparison.

We are NOT submitting to the leaderboard; instead the paper reports our cross-validated
Spearman ρ against (a) our own re-run baseline on identical official folds — the fair,
verifiable comparison — and (b) the published competition results as an external reference.

IMPORTANT CAVEAT (must be stated in the article): the competition's top scores were computed
on a PRIVATE held-out 80-antibody test set, NOT on the public CV folds. They are therefore an
*indicative* reference, not an identical-split comparison. The rigorous, like-for-like claim is
vs. our in-repo ESM-2 ridge baseline on the same `hierarchical_cluster_IgG_isotype_stratified_fold`.

Source: 2025 Ginkgo Datapoints Antibody Developability Competition outcomes
(mAbs 2026; doi:10.1080/19420862.2026.2634216).
"""
from __future__ import annotations

import pandas as pd

# Top reported Spearman rho on the PRIVATE held-out test set (best team per property).
COMPETITION_TOP_TEST = {
    "HIC": 0.708,
    "Tm2": 0.392,
    "PR_CHO": 0.356,
    "AC-SINS_pH7.4": 0.337,
    "Titer": 0.310,
}

REFERENCE_NOTE = (
    "Competition top scores are on the PRIVATE 80-antibody held-out test set (different split); "
    "shown as an external reference only. The like-for-like comparison is vs. the in-repo "
    "ESM-2 ridge baseline on identical official CV folds."
)


def add_reference(comparison: pd.DataFrame) -> pd.DataFrame:
    """Append the published competition top-test-set column to a comparison table."""
    out = comparison.copy()
    out["competition_top_test_set"] = out["property"].map(COMPETITION_TOP_TEST)
    return out
