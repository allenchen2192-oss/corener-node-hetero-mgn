import numpy as np
import os

NPZ_DIR = "D:/Allen/Allen_Workspace/beam_data/Abaqus_Beam_Data/02_beam_npz"
all_files = sorted(f for f in os.listdir(NPZ_DIR) if f.endswith(".npz"))

BAL_TRAIN = list(range(0,25))   + list(range(275,300)) + list(range(550,575)) + list(range(825,850))
BAL_TEST  = list(range(250,275))+ list(range(525,550)) + list(range(800,825)) + list(range(1075,1100))

def get_max_disps(indices):
    disps = []
    for i in indices:
        d = np.load(os.path.join(NPZ_DIR, all_files[i]))
        disp = np.linalg.norm(d["world_pos"][-1] - d["mesh_pos"], axis=1).max()
        disps.append(disp * 1000)  # mm
    return np.array(disps)

tr = get_max_disps(BAL_TRAIN)
te = get_max_disps(BAL_TEST)

print(f"Train (25/dir x4=100): min={tr.min():.1f}  max={tr.max():.1f}  mean={tr.mean():.1f}  median={np.median(tr):.1f} mm")
print(f"Test  (25/dir x4=100): min={te.min():.1f}  max={te.max():.1f}  mean={te.mean():.1f}  median={np.median(te):.1f} mm")
print(f"Train <5mm:  {(tr<5).sum()}/100   Test <5mm:  {(te<5).sum()}/100")
print(f"Train >20mm: {(tr>20).sum()}/100  Test >20mm: {(te>20).sum()}/100")
