"""
Generate bi-material cantilever beam dataset (1100 samples).

Layout:
  Material 1 (Aluminum, E1=70 GPa): bottom J < SPLIT_J (y < 0)
  Material 2 (random E2):            top    J >= SPLIT_J (y >= 0)
  Interface at y ~ -0.1 m (horizontal split, visible for Y-bending)

1100 samples = 250 train + 25 test per direction x 4 directions
  Round-robin order: +Y, -Y, +X, -X, +Y, -Y, ...
  Train indices: 0-999   Test indices: 1000-1099

Usage:
  python generate_bimaterial_beam.py

Output: ./02_abaqus_npz_bimaterial/sample_XXXXX.npz
"""

import csv
import os
import subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Config ────────────────────────────────────────────────────────────────────
ABAQUS_CMD  = "abaqus"
OUTPUT_DIR  = "./02_abaqus_npz_bimaterial"
WORK_DIR    = "./abaqus_work_bimaterial"
EXTRACT_SCR = os.path.abspath("extract_odb_bimaterial.py")

N_WORKERS = 8   # parallel Abaqus jobs (server has 16 cores, 9999 licenses)

# Geometry
L, W, H = 5.0, 1.0, 1.0
NX, NY, NZ = 25, 5, 5          # 26x6x6 = 936 nodes
SPLIT_J    = NY // 2            # bottom J<2 = Mat1, top J>=2 = Mat2

E1, NU = 70e9, 0.33
E2_MIN, E2_MAX = 50e9, 200e9    # log-uniform: E2/E1 = 0.7 ~ 2.9
P_MIN,  P_MAX  = 200e3, 1000e3  # log-uniform: 200 ~ 1000 kN  (5x range)

DIRECTIONS      = ['+Y', '-Y', '+X', '-X']
N_TRAIN_PER_DIR = 250
N_TEST_PER_DIR  = 25
N_PER_DIR       = N_TRAIN_PER_DIR + N_TEST_PER_DIR   # 275 per direction

# ── Mesh (computed once at import; workers inherit these) ──────────────────────
def nid(i, j, k):
    return i * (NY+1) * (NZ+1) + j * (NZ+1) + k + 1

def build_nodes():
    nodes = {}
    for i in range(NX+1):
        x = L * i / NX
        for j in range(NY+1):
            y = W * j / NY - W/2
            for k in range(NZ+1):
                z = H * k / NZ - H/2
                nodes[nid(i,j,k)] = (x, y, z)
    return nodes

def elem_conn_8(I, J, K):
    return (
        nid(I,   J,   K  ), nid(I+1, J,   K  ),
        nid(I+1, J+1, K  ), nid(I,   J+1, K  ),
        nid(I,   J,   K+1), nid(I+1, J,   K+1),
        nid(I+1, J+1, K+1), nid(I,   J+1, K+1),
    )

def build_elements():
    mat1, mat2 = {}, {}
    eid = 1
    for I in range(NX):
        for J in range(NY):
            for K in range(NZ):
                conn = elem_conn_8(I, J, K)
                if J < SPLIT_J:
                    mat1[eid] = conn
                else:
                    mat2[eid] = conn
                eid += 1
    return mat1, mat2

NODES = build_nodes()
MAT1_ELEMS, MAT2_ELEMS = build_elements()

def clamped_nodes():
    return [nid(0, j, k) for j in range(NY+1) for k in range(NZ+1)]

def tip_face_elem_ids():
    ids = []
    for J in range(NY):
        for K in range(NZ):
            ids.append((NX-1)*NY*NZ + J*NZ + K + 1)
    return ids

# ── INP writer ────────────────────────────────────────────────────────────────
def write_inp(sid, E2, P, direction, path):
    sign = 1 if '+' in direction else -1
    trvec = f"0., {sign*1.}, 0." if 'Y' in direction else f"0., 0., {sign*1.}"
    pressure = P / (W * H)
    dt = 1.0 / 19
    tip_elems = tip_face_elem_ids()

    lines = [
        "*Heading",
        f"Bi-material beam sample {sid:05d}  dir={direction}  E2={E2/1e9:.2f}GPa  P={P/1e3:.1f}kN",
        "", "*Node",
    ]
    for label, (x, y, z) in sorted(NODES.items()):
        lines.append(f"{label}, {x:.6f}, {y:.6f}, {z:.6f}")

    lines.append("*Element, type=C3D8, elset=MAT1_ELEMS")
    for eid, conn in sorted(MAT1_ELEMS.items()):
        lines.append(f"{eid}, " + ", ".join(map(str, conn)))

    lines.append("*Element, type=C3D8, elset=MAT2_ELEMS")
    for eid, conn in sorted(MAT2_ELEMS.items()):
        lines.append(f"{eid}, " + ", ".join(map(str, conn)))

    lines.append("*Elset, elset=TIP_FACE_ELEMS")
    for i in range(0, len(tip_elems), 16):
        lines.append(", ".join(map(str, tip_elems[i:i+16])))

    lines.append("*Surface, type=ELEMENT, name=TIP_SURF")
    lines.append("TIP_FACE_ELEMS, S4")

    clamp = clamped_nodes()
    lines.append("*Nset, nset=CLAMPED")
    for i in range(0, len(clamp), 16):
        lines.append(", ".join(map(str, clamp[i:i+16])))

    lines += [
        "*Solid Section, elset=MAT1_ELEMS, material=MATERIAL1", "1.0",
        "*Solid Section, elset=MAT2_ELEMS, material=MATERIAL2", "1.0",
        "*Material, name=MATERIAL1", "*Elastic", f"{E1:.6e}, {NU}",
        "*Material, name=MATERIAL2", "*Elastic", f"{E2:.6e}, {NU}",
        "*Boundary", "CLAMPED, 1, 3, 0.0",
        "*Step, name=LOADING, nlgeom=NO", "*Static",
        f"{dt:.8f}, 1.0, {dt:.8f}, {dt:.8f}",
        "*Amplitude, name=RAMP", "0.0, 0.0, 1.0, 1.0",
        "*Dsload, constant resultant=YES, amplitude=RAMP",
        f"TIP_SURF, TRVEC, {pressure:.6e}, {trvec}",
        "*Output, field, frequency=1", "*Node Output", "U",
        "*Element Output, position=AVERAGED AT NODES", "S",
        "*End Step",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

# ── Worker ────────────────────────────────────────────────────────────────────
_CLEANUP_EXTS = ['.odb', '.dat', '.msg', '.sta', '.prt', '.com', '.sim', '.stt', '.mdl']

def run_sample(args):
    sid, s, n_total = args
    direction, E2, P = s["direction"], s["E2"], s["P"]
    job = f"bimaterial_{sid:05d}"
    inp = os.path.join(WORK_DIR, f"{job}.inp")
    odb = os.path.join(WORK_DIR, f"{job}.odb")
    npz = os.path.join(OUTPUT_DIR, f"sample_{sid:05d}.npz")

    if os.path.exists(npz):
        return f"[{sid+1:4d}/{n_total}] SKIP"

    write_inp(sid, E2, P, direction, inp)

    cmd_solve = f'{ABAQUS_CMD} job={job} input="{os.path.abspath(inp)}" cpus=1 interactive'
    r = subprocess.run(cmd_solve, cwd=WORK_DIR, capture_output=True, text=True, shell=True)
    if r.returncode != 0:
        return f"[{sid+1:4d}/{n_total}] ERROR solver  dir={direction} sid={sid}"

    cmd_ext = (f'{ABAQUS_CMD} python "{EXTRACT_SCR}"'
               f' "{os.path.abspath(odb)}" "{os.path.abspath(npz)}"'
               f' {E1} {E2} {NU} {SPLIT_J} {NX} {NY} {NZ}')
    r = subprocess.run(cmd_ext, capture_output=True, text=True, shell=True)
    if r.returncode != 0:
        return f"[{sid+1:4d}/{n_total}] ERROR extract dir={direction} sid={sid}"

    for ext in _CLEANUP_EXTS:
        tmp = os.path.join(WORK_DIR, job + ext)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    return f"[{sid+1:4d}/{n_total}] OK  dir={direction}  E2={E2/1e9:.0f}GPa  P={P/1e3:.0f}kN"

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(WORK_DIR,   exist_ok=True)

    RNG = np.random.default_rng(seed=42)
    # Log-uniform sampling: equal density per decade
    E2_per_dir = {d: np.exp(RNG.uniform(np.log(E2_MIN), np.log(E2_MAX), N_PER_DIR))
                  for d in DIRECTIONS}
    P_per_dir  = {d: np.exp(RNG.uniform(np.log(P_MIN),  np.log(P_MAX),  N_PER_DIR))
                  for d in DIRECTIONS}

    samples = []
    for i in range(N_TRAIN_PER_DIR):
        for d in DIRECTIONS:
            samples.append({"direction": d, "E2": float(E2_per_dir[d][i]),
                            "P": float(P_per_dir[d][i])})
    for i in range(N_TEST_PER_DIR):
        for d in DIRECTIONS:
            samples.append({"direction": d, "E2": float(E2_per_dir[d][N_TRAIN_PER_DIR + i]),
                            "P": float(P_per_dir[d][N_TRAIN_PER_DIR + i])})

    N_TOTAL = len(samples)
    print(f"Generating {N_TOTAL} samples with {N_WORKERS} parallel workers...")

    args_list = [(sid, s, N_TOTAL) for sid, s in enumerate(samples)]

    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_sample, a): a[0] for a in args_list}
        for future in as_completed(futures):
            msg = future.result()
            done += 1
            print(f"({done:4d}/{N_TOTAL}) {msg}", flush=True)

    print(f"\nAll done. {N_TOTAL} samples attempted.")

    # Save metadata CSV: sample_id → direction, E2, P
    metadata_path = os.path.join(OUTPUT_DIR, "sample_metadata.csv")
    with open(metadata_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "direction", "E2_Pa", "P_N"])
        writer.writeheader()
        for sid, s in enumerate(samples):
            writer.writerow({"sample_id": sid, "direction": s["direction"],
                             "E2_Pa": s["E2"], "P_N": s["P"]})
    print(f"Metadata saved: {metadata_path}")
