"""
analyze_material_params.py
==========================
Read all 220 M0040 npz samples, extract per-material CTE and E,
and report the distribution to help decide the extrapolation test split.
"""

import os
import numpy as np
import pandas as pd

NPZ_DIR = "./02_abaqus_npz_m0040"

records = []
for i in range(1, 221):
    sid = f"S{i:04d}"
    path = os.path.join(NPZ_DIR, f"{sid}.npz")
    if not os.path.exists(path):
        print(f"MISSING: {sid}")
        continue
    d = np.load(path)
    em = d["elem_mat"]
    ec = d["elem_CTE"]
    ee = d["elem_E"]

    def mat_val(mat_id):
        mask = (em == mat_id)
        if not mask.any():
            return np.nan, np.nan
        return float(ec[mask][0]), float(ee[mask][0])

    # mat ids: 0,1=Si, 2=Solder, 3=UF
    si_cte,  si_e  = mat_val(0)
    sl_cte,  sl_e  = mat_val(2)
    uf_cte,  uf_e  = mat_val(3)

    records.append(dict(
        sid=sid, idx=i,
        si_cte=si_cte,   si_e=si_e,
        sl_cte=sl_cte,   sl_e=sl_e,
        uf_cte=uf_cte,   uf_e=uf_e,
    ))

df = pd.DataFrame(records).set_index("sid")
print("=" * 60)
print(f"Total samples loaded: {len(df)}")
print("\n--- CTE (ppm/K) range per material ---")
for col in ["si_cte", "sl_cte", "uf_cte"]:
    print(f"  {col}: min={df[col].min():.4g}  max={df[col].max():.4g}  std={df[col].std():.4g}")

print("\n--- E (MPa) range per material ---")
for col in ["si_e", "sl_e", "uf_e"]:
    print(f"  {col}: min={df[col].min():.4g}  max={df[col].max():.4g}  std={df[col].std():.4g}")

# Identify which parameters actually vary (std > 0)
print("\n--- Parameters that vary across samples ---")
for col in df.columns:
    if col == "idx": continue
    if df[col].std() > 0:
        print(f"  {col}  (std={df[col].std():.4g})")
    else:
        print(f"  {col}  [FIXED]")

# Show top/bottom 5 for each varying parameter
varying = [c for c in ["si_cte","sl_cte","uf_cte","si_e","sl_e","uf_e"]
           if df[c].std() > 0]

print("\n" + "=" * 60)
print("Top/Bottom 5 samples for each varying parameter:")
for col in varying:
    bot = df.nsmallest(5, col)[col]
    top = df.nlargest(5,  col)[col]
    print(f"\n  [{col}]")
    print(f"  Bottom 5: {list(zip(bot.index, bot.values.round(4)))}")
    print(f"  Top    5: {list(zip(top.index, top.values.round(4)))}")

df.to_csv("./material_params_m0040.csv")
print("\nSaved -> material_params_m0040.csv")
