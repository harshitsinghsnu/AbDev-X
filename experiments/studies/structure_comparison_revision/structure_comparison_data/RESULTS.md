# Matched structure-predictor comparison

This is a retrospective AbDev-X/common-structure20 experiment. It is not a replacement of the original primary model results. The other four representation blocks and four-expert training code are held fixed. All methods use the same common-complete-case antibodies and the same outer fold assignments.

Common cases: 244 GDPa1 training antibodies and 80 GDPa3 antibodies before assay-specific missing-label exclusions.

CV errors are fold SD; held-out errors are antibody-resampling BCa 95% intervals (20,000 replicates). Paired differences include a Bonferroni interval sensitivity across all predictor-assay contrasts. Bootstrap intervals do not account for training-set uncertainty or antibody family dependence.

| Predictor | Assay | CV mean +/- SD | GDPa3 rho [95% CI] | Held-out n |
|---|---|---|---|---|
| ABodyBuilder3 | HIC | 0.621 +/- 0.142 | 0.518 [0.320, 0.665] | 79 |
| ABodyBuilder3 | AC-SINS | 0.502 +/- 0.136 | 0.260 [0.017, 0.470] | 80 |
| ABodyBuilder3 | PR_CHO | 0.401 +/- 0.160 | 0.251 [0.024, 0.452] | 80 |
| ABodyBuilder3 | SMAC | 0.485 +/- 0.083 | 0.580 [0.400, 0.717] | 80 |
| ABodyBuilder3 | HAC | 0.774 +/- 0.119 | 0.606 [0.372, 0.769] | 61 |
| ABodyBuilder3 | Tm1 | 0.111 +/- 0.218 | 0.176 [-0.074, 0.409] | 80 |
| ABodyBuilder3 | Tm2 | -0.112 +/- 0.071 | -0.082 [-0.322, 0.164] | 74 |
| ABodyBuilder3 | Tonset | -0.047 +/- 0.136 | 0.092 [-0.133, 0.303] | 80 |
| ABodyBuilder3 | SEC | -0.074 +/- 0.190 | -0.135 [-0.345, 0.095] | 80 |
| ABodyBuilder3 | Titer | 0.302 +/- 0.082 | 0.048 [-0.178, 0.261] | 80 |
| IgFold | HIC | 0.629 +/- 0.167 | 0.475 [0.275, 0.635] | 79 |
| IgFold | AC-SINS | 0.498 +/- 0.148 | 0.256 [0.023, 0.468] | 80 |
| IgFold | PR_CHO | 0.415 +/- 0.143 | 0.245 [0.017, 0.447] | 80 |
| IgFold | SMAC | 0.446 +/- 0.069 | 0.589 [0.409, 0.726] | 80 |
| IgFold | HAC | 0.805 +/- 0.096 | 0.615 [0.387, 0.774] | 61 |
| IgFold | Tm1 | 0.132 +/- 0.209 | 0.161 [-0.093, 0.392] | 80 |
| IgFold | Tm2 | -0.031 +/- 0.172 | -0.061 [-0.301, 0.185] | 74 |
| IgFold | Tonset | -0.070 +/- 0.161 | 0.064 [-0.157, 0.277] | 80 |
| IgFold | SEC | -0.047 +/- 0.171 | -0.027 [-0.231, 0.182] | 80 |
| IgFold | Titer | 0.271 +/- 0.111 | 0.129 [-0.103, 0.342] | 80 |
| ESMFold | HIC | 0.627 +/- 0.116 | 0.512 [0.304, 0.666] | 79 |
| ESMFold | AC-SINS | 0.500 +/- 0.144 | 0.243 [0.006, 0.459] | 80 |
| ESMFold | PR_CHO | 0.412 +/- 0.161 | 0.239 [0.009, 0.441] | 80 |
| ESMFold | SMAC | 0.515 +/- 0.051 | 0.592 [0.406, 0.731] | 80 |
| ESMFold | HAC | 0.782 +/- 0.127 | 0.625 [0.403, 0.780] | 61 |
| ESMFold | Tm1 | 0.110 +/- 0.200 | 0.178 [-0.075, 0.412] | 80 |
| ESMFold | Tm2 | -0.106 +/- 0.093 | -0.068 [-0.313, 0.180] | 74 |
| ESMFold | Tonset | -0.065 +/- 0.122 | 0.100 [-0.121, 0.305] | 80 |
| ESMFold | SEC | -0.081 +/- 0.200 | -0.117 [-0.328, 0.096] | 80 |
| ESMFold | Titer | 0.269 +/- 0.121 | 0.090 [-0.134, 0.301] | 80 |
| AlphaFold3 | HIC | 0.598 +/- 0.109 | 0.455 [0.249, 0.617] | 79 |
| AlphaFold3 | AC-SINS | 0.495 +/- 0.147 | 0.250 [0.013, 0.463] | 80 |
| AlphaFold3 | PR_CHO | 0.393 +/- 0.171 | 0.253 [0.023, 0.452] | 80 |
| AlphaFold3 | SMAC | 0.490 +/- 0.113 | 0.518 [0.326, 0.672] | 80 |
| AlphaFold3 | HAC | 0.791 +/- 0.112 | 0.613 [0.389, 0.772] | 61 |
| AlphaFold3 | Tm1 | 0.101 +/- 0.215 | 0.184 [-0.065, 0.415] | 80 |
| AlphaFold3 | Tm2 | 0.011 +/- 0.185 | -0.063 [-0.303, 0.183] | 74 |
| AlphaFold3 | Tonset | -0.048 +/- 0.118 | 0.068 [-0.151, 0.280] | 80 |
| AlphaFold3 | SEC | -0.060 +/- 0.180 | -0.134 [-0.340, 0.079] | 80 |
| AlphaFold3 | Titer | 0.273 +/- 0.109 | 0.074 [-0.150, 0.282] | 80 |

## Interpretation limits

AF3 uses seed 42, 10 recycles, five diffusion samples, and official top-ranking-score selection, with empty MSAs and no templates. This is an AF3 single-sequence condition, not the full database-assisted AF3 pipeline. ESMFold uses VH + 25 glycines + VL and the documented chain index offset, with linker atoms excluded. IgFold uses the four-model ensemble without optional refinement. Cached ABodyBuilder3 structures are the baseline; their original inference runtime is unavailable. The same exact-chain check, heavy-atom completion policy, and feature definitions are applied to all sources.

The primary 57-feature block combines TAP, MOE, Aggrescan3D, and AntiFold caches. Its complete regeneration pipeline is unavailable here. Therefore these results assess a common open 20-feature structural panel, not an exact structure-predictor swap within the original bio57 model. No structural RMSD or optimal-predictor claim is supported without experimental reference structures or independent confirmatory evaluation.