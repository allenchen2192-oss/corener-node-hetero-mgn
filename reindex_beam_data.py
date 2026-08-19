"""
reindex_beam_data.py
====================
將 beam data 從「方向分組」排列改成「交錯方向」排列。

原始排列：
  sample_00000 - sample_00274: Direction 0 (Y+)
  sample_00275 - sample_00549: Direction 1 (Y-)
  sample_00550 - sample_00824: Direction 2 (X+)
  sample_00825 - sample_01099: Direction 3 (X-)

新交錯排列：
  new 0 = old 0    (Dir 0, j=0)
  new 1 = old 275  (Dir 1, j=0)
  new 2 = old 550  (Dir 2, j=0)
  new 3 = old 825  (Dir 3, j=0)
  new 4 = old 1    (Dir 0, j=1)
  new 5 = old 276  (Dir 1, j=1)
  ...

效果：
  Train (0-999)  → 250 samples × 4 directions（平衡）
  Test (1000-1099) → 25 samples × 4 directions（平衡）
"""

import os

SAMPLES_PER_DIR = 275
NUM_DIRS = 4
TOTAL = SAMPLES_PER_DIR * NUM_DIRS  # 1100

PT_DIR  = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/04_preprocessed_pt_beam_prevvel"
NPZ_DIR = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/02_beam_npz"

DRY_RUN = True  # 改成 False 才會真正重新命名


def build_mapping():
    """old_idx → new_idx"""
    mapping = {}
    new_idx = 0
    for j in range(SAMPLES_PER_DIR):
        for d in range(NUM_DIRS):
            old_idx = d * SAMPLES_PER_DIR + j
            mapping[old_idx] = new_idx
            new_idx += 1
    return mapping


def reindex_directory(data_dir, ext, mapping):
    files = sorted(f for f in os.listdir(data_dir)
                   if f.startswith("sample_") and f.endswith(ext))
    print(f"  Found {len(files)} {ext} files")

    # Step 1: old name → temp name
    for fname in files:
        old_idx = int(fname.replace("sample_", "").replace(ext, ""))
        if old_idx not in mapping:
            print(f"  [warn] {fname} not in mapping, skip")
            continue
        new_idx = mapping[old_idx]
        src = os.path.join(data_dir, fname)
        dst = os.path.join(data_dir, f"_tmp_{new_idx:05d}{ext}")
        if DRY_RUN:
            print(f"  sample_{old_idx:05d}{ext}  →  _tmp_{new_idx:05d}{ext}")
        else:
            os.rename(src, dst)

    if DRY_RUN:
        return

    # Step 2: temp name → final name
    tmp_files = sorted(f for f in os.listdir(data_dir)
                       if f.startswith("_tmp_") and f.endswith(ext))
    for fname in tmp_files:
        new_idx = int(fname.replace("_tmp_", "").replace(ext, ""))
        src = os.path.join(data_dir, fname)
        dst = os.path.join(data_dir, f"sample_{new_idx:05d}{ext}")
        os.rename(src, dst)
    print(f"  Done: {len(tmp_files)} files renamed")


if __name__ == "__main__":
    mapping = build_mapping()

    # 驗證映射
    assert len(mapping) == TOTAL
    assert set(mapping.values()) == set(range(TOTAL))
    print(f"Mapping verified: {TOTAL} samples, 4 directions")
    print(f"DRY_RUN = {DRY_RUN}")
    print()

    print("=== PT Directory ===")
    reindex_directory(PT_DIR, ".pt", mapping)

    print("\n=== NPZ Directory ===")
    reindex_directory(NPZ_DIR, ".npz", mapping)

    if DRY_RUN:
        print("\nDry run 完成。確認映射正確後，將 DRY_RUN = False 再執行一次。")
    else:
        print("\nReindex 完成！Train (0-999) 和 Test (1000-1099) 現在各包含 4 個方向。")
