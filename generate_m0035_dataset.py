"""
Driver: batch-extract all M0035 ODB files -> NPZ dataset.
Usage (on the server with Abaqus installed):
    python generate_m0035_dataset.py [--workers N] [--redo]

Reads  : 0035/full_validation.csv
ODB dir: 0035/ODB/
Out dir: 02_abaqus_npz_m0035/
"""

import os
import csv
import subprocess
import argparse

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ODB_DIR    = os.path.join(BASE_DIR, "0035", "ODB")
OUT_DIR    = os.path.join(BASE_DIR, "02_abaqus_npz_m0035")
CSV_PATH   = os.path.join(BASE_DIR, "0035", "full_validation.csv")
EXTRACT_PY = os.path.join(BASE_DIR, "extract_odb_m0035.py")

ABAQUS_CMD = "abaqus"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo",    action="store_true",
                        help="Re-extract even if NPZ already exists")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="Only process these sample IDs, e.g. S0001 S0002")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Read CSV
    samples = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "PASS") != "PASS":
                continue
            if args.samples and row["sample_id"] not in args.samples:
                continue
            samples.append({
                "sample_id": row["sample_id"],
                "filename":  row["filename"],
                "E_MPa":     float(row["E_MPa_actual"]),
                "CTE":       float(row["CTE_per_C_actual"]),
                "split":     row["split"],
            })

    print("Total samples to process: {}".format(len(samples)))
    ok = 0; skip = 0; fail = 0

    for s in samples:
        sid      = s["sample_id"]
        odb_name = s["filename"].replace(".inp", ".odb")
        odb_path = os.path.join(ODB_DIR, odb_name)
        npz_path = os.path.join(OUT_DIR, "{}.npz".format(sid))

        if not os.path.exists(odb_path):
            print("[MISSING] {}".format(odb_path))
            fail += 1
            continue

        if os.path.exists(npz_path) and not args.redo:
            skip += 1
            continue

        cmd = ["cmd", "/c", ABAQUS_CMD, "python", EXTRACT_PY,
               odb_path, npz_path,
               str(s["E_MPa"]), str(s["CTE"]),
               ]
        print("[RUN] {}  E={} MPa  CTE={}".format(sid, s["E_MPa"], s["CTE"]))
        print("  CMD: {}".format(" ".join(cmd)))
        ret = subprocess.call(cmd)
        if ret == 0 and os.path.exists(npz_path):
            ok += 1
        else:
            print("  [FAIL] exit code {}".format(ret))
            fail += 1

    print("\nDone: {} ok  {} skipped  {} failed".format(ok, skip, fail))


if __name__ == "__main__":
    main()