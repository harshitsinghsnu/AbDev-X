"""Prepare GDPa3 sequences in DELPHI input format (BARCODE, HSEQ, LSEQ)."""
import pandas as pd

df = pd.read_csv(r"C:\Users\hs494\ab-dev\data\processed\gdpa3_table.csv")
out = pd.DataFrame({
    "BARCODE": df["antibody_id"],
    "HSEQ": df["vhn"],
    "LSEQ": df["vl"],
    # targets for post-hoc evaluation
    "PR_CHO_measured": df["polyreactivity_prscore_cho_avg"],
    "HIC_measured": df["hic_rt_avg"],
    "Tm2_measured": df["tm2_nanodsf_avg"],
    "ACSINS_measured": df["acsins_dLmax_ph7.4_avg"],
    "Titer_measured": df["titer_avg"],
})
print(f"Shape: {out.shape}")
print(f"Missing HSEQ: {out['HSEQ'].isna().sum()}  Missing LSEQ: {out['LSEQ'].isna().sum()}")
print(out.head(3))

# Save in both formats DELPHI expects
out[["BARCODE","HSEQ","LSEQ"]].to_csv(
    r"C:\Users\hs494\ab-dev\data\processed\gdpa3_delphi_input.csv", index=False)
out.to_csv(
    r"C:\Users\hs494\ab-dev\data\processed\gdpa3_delphi_input_with_targets.csv", index=False)
print("Saved gdpa3_delphi_input.csv")
