import numpy as np

path = '02_abaqus_npz'

for fname, npe, n_corner in [
    ('sample_00000.npz', 10, 4),
    ('sample_00002.npz', 20, 8),
    ('sample_00004.npz',  8, 8),
]:
    d  = np.load(f'{path}/{fname}')
    ec = d['elem_conn']
    N  = d['mesh_pos'].shape[0]
    npe_actual = int(d['n_nodes_per_elem'])

    if npe_actual == 8:
        n_corner_nodes = N
        n_midside_nodes = 0
    else:
        corner_ids = np.unique(ec[:, :n_corner])
        n_corner_nodes  = len(corner_ids)
        n_midside_nodes = N - n_corner_nodes

    print(f"{fname}  npe={npe_actual}  total={N}  "
          f"corner={n_corner_nodes}  mid-side={n_midside_nodes}  "
          f"ratio={n_corner_nodes/N:.2f}")
