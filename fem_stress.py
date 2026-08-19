"""
fem_stress.py
=============
FEM stress recovery from nodal displacement using element shape functions.
Supports C3D8 (8-node linear hex), C3D20 (20-node quadratic hex),
and C3D10 (10-node quadratic tet).

Algorithm
---------
For each Gauss point g in each element e:
  1.  J[e] = xe^T @ dN/dxi_g           (3x3 Jacobian)
  2.  dN/dx = dN/dxi_g @ J[e]^{-1}     (physical derivatives, npe x 3)
  3.  epsilon = B(dN/dx) @ u_e          (engineering strains, 6)
  4.  sigma   = D @ epsilon             (stress via Hooke's law, 6)

For C3D8: element-NODAL stress is computed by Lagrange extrapolation from
8 Gauss-point stresses to 8 corner nodes (matches Abaqus ELEMENT_NODAL).
Nodal stress = unweighted average over contributing elements.

For C3D20/C3D10: Gauss-point average per element, then scatter to nodes.
"""

import math
import torch

# -- Elastic D matrix (isotropic, Voigt: [e11,e22,e33,g12,g13,g23]) -----------

def _D(E: float = 70e9, nu: float = 0.33) -> torch.Tensor:
    lam = E * nu / ((1 + nu) * (1 - 2*nu))
    mu  = E / (2 * (1 + nu))
    return torch.tensor([
        [lam+2*mu, lam,      lam,      0,  0,  0],
        [lam,      lam+2*mu, lam,      0,  0,  0],
        [lam,      lam,      lam+2*mu, 0,  0,  0],
        [0,        0,        0,        mu, 0,  0],
        [0,        0,        0,        0,  mu, 0],
        [0,        0,        0,        0,  0,  mu],
    ], dtype=torch.float64)


def _D_batch(E_arr: torch.Tensor, nu_arr: torch.Tensor) -> torch.Tensor:
    """Per-element D matrices. E_arr, nu_arr: (E_elem,) -> returns (E_elem, 6, 6)."""
    lam = E_arr * nu_arr / ((1 + nu_arr) * (1 - 2 * nu_arr))  # (E_elem,)
    mu  = E_arr / (2 * (1 + nu_arr))                           # (E_elem,)
    z   = torch.zeros_like(lam)
    D   = torch.stack([
        torch.stack([lam+2*mu, lam,      lam,      z,  z,  z], dim=1),
        torch.stack([lam,      lam+2*mu, lam,      z,  z,  z], dim=1),
        torch.stack([lam,      lam,      lam+2*mu, z,  z,  z], dim=1),
        torch.stack([z,        z,        z,        mu, z,  z], dim=1),
        torch.stack([z,        z,        z,        z,  mu, z], dim=1),
        torch.stack([z,        z,        z,        z,  z,  mu], dim=1),
    ], dim=1)  # (E_elem, 6, 6)
    return D


# -- C3D8: 8-node linear hex ---------------------------------------------------
# Abaqus node ordering (0-indexed): corners at (+-1,+-1,+-1)
_C3D8_XI = torch.tensor([
    [-1,-1,-1], [ 1,-1,-1], [ 1, 1,-1], [-1, 1,-1],
    [-1,-1, 1], [ 1,-1, 1], [ 1, 1, 1], [-1, 1, 1],
], dtype=torch.float64)


def _c3d8_dNdxi(xi: float, eta: float, zeta: float) -> torch.Tensor:
    """Shape fn derivatives at (xi,eta,zeta). Returns (8,3)."""
    xn, en, zn = _C3D8_XI[:,0], _C3D8_XI[:,1], _C3D8_XI[:,2]
    dxi  = xn  * (1 + en*eta)  * (1 + zn*zeta) / 8
    deta = en  * (1 + xn*xi)   * (1 + zn*zeta) / 8
    dzet = zn  * (1 + xn*xi)   * (1 + en*eta)  / 8
    return torch.stack([dxi, deta, dzet], dim=1)  # (8,3)


def _gauss_c3d8():
    """2x2x2 Gauss rule. Returns list of (dNdxi (8,3), weight float).
    Ordering: outer loop zeta, middle eta, inner xi (each +-a)."""
    a = 1.0 / math.sqrt(3)
    gps = []
    for zeta in [-a, a]:
        for eta in [-a, a]:
            for xi in [-a, a]:
                gps.append((_c3d8_dNdxi(xi, eta, zeta), 1.0))
    return gps  # 8 entries


# -- C3D20: 20-node quadratic serendipity hex ----------------------------------
_C3D20_XI = torch.tensor([
    [-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],  # corners bottom
    [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1],  # corners top
    [ 0,-1,-1],[ 1, 0,-1],[ 0, 1,-1],[-1, 0,-1],  # bottom midsides
    [ 0,-1, 1],[ 1, 0, 1],[ 0, 1, 1],[-1, 0, 1],  # top midsides
    [-1,-1, 0],[ 1,-1, 0],[ 1, 1, 0],[-1, 1, 0],  # vertical midsides
], dtype=torch.float64)


def _c3d20_dNdxi(xi: float, eta: float, zeta: float) -> torch.Tensor:
    """Shape fn derivatives at (xi,eta,zeta). Returns (20,3)."""
    xn = _C3D20_XI[:,0]
    en = _C3D20_XI[:,1]
    zn = _C3D20_XI[:,2]
    dxi  = torch.zeros(20, dtype=torch.float64)
    deta = torch.zeros(20, dtype=torch.float64)
    dzet = torch.zeros(20, dtype=torch.float64)

    for i in range(8):
        x, e, z = float(xn[i]), float(en[i]), float(zn[i])
        A = 1 + e*eta;  B = 1 + z*zeta;  C = 1 + x*xi
        dxi[i]  = x * A * B * (2*x*xi + e*eta + z*zeta - 1) / 8
        deta[i] = e * C * B * (x*xi + 2*e*eta + z*zeta - 1) / 8
        dzet[i] = z * C * A * (x*xi + e*eta + 2*z*zeta - 1) / 8

    for i in range(8, 20):
        x, e, z = float(xn[i]), float(en[i]), float(zn[i])
        if x == 0:
            dxi[i]  = -xi * (1 + e*eta) * (1 + z*zeta) / 2
            deta[i] =  e  * (1 - xi*xi) * (1 + z*zeta) / 4
            dzet[i] =  z  * (1 - xi*xi) * (1 + e*eta)  / 4
        elif e == 0:
            dxi[i]  =  x   * (1 - eta*eta) * (1 + z*zeta) / 4
            deta[i] = -eta * (1 + x*xi)    * (1 + z*zeta) / 2
            dzet[i] =  z   * (1 + x*xi)    * (1 - eta*eta) / 4
        else:
            dxi[i]  =  x    * (1 + e*eta) * (1 - zeta*zeta) / 4
            deta[i] =  e    * (1 + x*xi)  * (1 - zeta*zeta) / 4
            dzet[i] = -zeta * (1 + x*xi)  * (1 + e*eta)     / 2

    return torch.stack([dxi, deta, dzet], dim=1)  # (20,3)


def _gauss_c3d20():
    """3x3x3 Gauss rule (27 pts). Returns list of (dNdxi (20,3), weight float)."""
    a = math.sqrt(3.0 / 5.0)   # approx 0.7746
    pts = [-a, 0.0, a]
    wts = [5.0/9, 8.0/9, 5.0/9]
    gps = []
    for zi, zeta in enumerate(pts):
        for ei, eta in enumerate(pts):
            for xi_i, xi in enumerate(pts):
                w = wts[xi_i] * wts[ei] * wts[zi]
                gps.append((_c3d20_dNdxi(xi, eta, zeta), w))
    return gps  # 27 entries


# -- C3D10: 10-node quadratic tet ----------------------------------------------
def _c3d10_dNdxi(r: float, s: float, t: float) -> torch.Tensor:
    """Shape fn derivatives at (r,s,t). Returns (10,3)."""
    u = 1.0 - r - s - t
    dN = torch.zeros(10, 3, dtype=torch.float64)

    dN[0, 0] = 4*r - 1
    dN[1, 1] = 4*s - 1
    dN[2, 2] = 4*t - 1
    c3 = -(4*u - 1)
    dN[3, 0] = c3;  dN[3, 1] = c3;  dN[3, 2] = c3

    dN[4, 0] = 4*s;   dN[4, 1] = 4*r
    dN[5, 1] = 4*t;   dN[5, 2] = 4*s
    dN[6, 0] = 4*t;   dN[6, 2] = 4*r
    dN[7, 0] = 4*(u-r); dN[7, 1] = -4*r;  dN[7, 2] = -4*r
    dN[8, 0] = -4*s;  dN[8, 1] = 4*(u-s); dN[8, 2] = -4*s
    dN[9, 0] = -4*t;  dN[9, 1] = -4*t;   dN[9, 2] = 4*(u-t)

    return dN  # (10,3)


def _gauss_c3d10():
    """4-point symmetric Gauss rule for quadratic tet."""
    a = (5 + 3*math.sqrt(5)) / 20
    b = (5 -   math.sqrt(5)) / 20
    w = 1.0 / 24.0
    coords = [(a,b,b), (b,a,b), (b,b,a), (b,b,b)]
    return [(_c3d10_dNdxi(*c), w) for c in coords]


# -- Core computation ----------------------------------------------------------

def _gauss_stress_at(xe, ue, dNdxi_cpu, D_mat):
    """
    Stress at a single Gauss point for all elements.

    xe:       (E, npe, 3)
    ue:       (E, npe, 3)
    dNdxi_cpu: (npe, 3) CPU tensor
    D_mat:    (6,6) or (E,6,6)
    Returns:  (E, 6)
    """
    dev, dt = xe.device, xe.dtype
    dNdxi   = dNdxi_cpu.to(device=dev, dtype=dt)
    J       = xe.transpose(-2,-1) @ dNdxi          # (E,3,3)
    J_inv   = torch.linalg.inv(J)
    dNdx    = dNdxi.unsqueeze(0) @ J_inv            # (E,npe,3)
    eps = torch.stack([
        (dNdx[:,:,0] * ue[:,:,0]).sum(1),
        (dNdx[:,:,1] * ue[:,:,1]).sum(1),
        (dNdx[:,:,2] * ue[:,:,2]).sum(1),
        (dNdx[:,:,0]*ue[:,:,1] + dNdx[:,:,1]*ue[:,:,0]).sum(1),
        (dNdx[:,:,0]*ue[:,:,2] + dNdx[:,:,2]*ue[:,:,0]).sum(1),
        (dNdx[:,:,1]*ue[:,:,2] + dNdx[:,:,2]*ue[:,:,1]).sum(1),
    ], dim=1)  # (E, 6)
    return (D_mat @ eps.unsqueeze(-1)).squeeze(-1)  # (E, 6)


def _elem_stress(xe, ue, gauss_pts, D_mat):
    """
    Gauss-point-averaged stress per element.
    Returns (E, 6).  Used for C3D20/C3D10.
    """
    dev, dt = xe.device, xe.dtype
    D_mat   = D_mat.to(device=dev, dtype=dt)
    stress_sum = torch.zeros(xe.shape[0], 6, device=dev, dtype=dt)
    w_total    = 0.0
    for dNdxi_cpu, w_g in gauss_pts:
        stress_sum += w_g * _gauss_stress_at(xe, ue, dNdxi_cpu, D_mat)
        w_total    += w_g
    return stress_sum / w_total   # (E, 6)


def _c3d8_gauss_to_node_weights() -> torch.Tensor:
    """
    8x8 Lagrange extrapolation matrix W[n,g] for C3D8.

    Maps 8 Gauss-point stresses to 8 corner-node stresses, reproducing
    Abaqus ELEMENT_NODAL extrapolation.

    Gauss points at xi=+-a (a=1/sqrt(3)); corner nodes at xi=+-1.
    1-D Lagrange weights:
      r = (1+a)/(2a) approx 1.366   (same sign: extrapolate outward)
      s = (a-1)/(2a) approx -0.366  (opposite sign: suppressed)
    W[n,g] = w(n_xi,g_xi) * w(n_eta,g_eta) * w(n_zeta,g_zeta)

    Node ordering: _C3D8_XI (corners +-1).
    Gauss ordering: _gauss_c3d8() (outer zeta, mid eta, inner xi, each +-a).
    """
    a = 1.0 / math.sqrt(3)
    r = (1.0 + a) / (2.0 * a)   # approx  1.3660
    s = (a - 1.0) / (2.0 * a)   # approx -0.3660

    node_xi = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
               (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)]
    gauss_xi = [(xi, eta, zeta)
                for zeta in [-a, a] for eta in [-a, a] for xi in [-a, a]]

    W = torch.zeros(8, 8, dtype=torch.float64)
    for n, (nx, ny, nz) in enumerate(node_xi):
        for g, (gx, gy, gz) in enumerate(gauss_xi):
            W[n, g] = (r if nx*gx > 0 else s) * \
                      (r if ny*gy > 0 else s) * \
                      (r if nz*gz > 0 else s)
    return W   # (8, 8)


_C3D8_G2N = _c3d8_gauss_to_node_weights()   # precomputed once at import


def _elem_nodal_stress_c3d8(xe, ue, gauss_pts, D_mat):
    """
    Element-NODAL stress for C3D8: Lagrange-extrapolate from 8 Gauss points
    to 8 corner nodes — matches Abaqus ELEMENT_NODAL method.

    Returns (E_elem, 8, 6).
    """
    dev, dt = xe.device, xe.dtype
    D_mat   = D_mat.to(device=dev, dtype=dt)

    # Stack per-Gauss stresses: (E, 8, 6)
    sig_g = torch.stack(
        [_gauss_stress_at(xe, ue, dNdxi_cpu, D_mat) for dNdxi_cpu, _ in gauss_pts],
        dim=1
    )

    W = _C3D8_G2N.to(device=dev, dtype=dt)  # (8, 8)
    # sig_n[e,n,:] = sum_g W[n,g] * sig_g[e,g,:]
    return torch.einsum('ng,egc->enc', W, sig_g)   # (E, 8, 6)


# -- Public API ----------------------------------------------------------------

def _scatter_elem_to_nodes(elem_sig, elem_conn, N, npe, dev):
    """Average element stresses (E_elem, 6) to nodes (N, 6)."""
    E_elem     = elem_conn.shape[0]
    node_sig   = torch.zeros(N, 6, device=dev, dtype=torch.float64)
    node_count = torch.zeros(N,    device=dev, dtype=torch.float64)
    for k in range(npe):
        idx = elem_conn[:, k]
        node_sig.scatter_add_(0, idx.unsqueeze(1).expand(-1, 6), elem_sig)
        node_count.scatter_add_(0, idx, torch.ones(E_elem, device=dev, dtype=torch.float64))
    return node_sig / node_count.clamp(min=1).unsqueeze(1)  # (N, 6)


def _scatter_elem_nodal_to_nodes(elem_nodal_sig, elem_conn, N, npe, dev):
    """
    Average element-NODAL stresses (E_elem, npe, 6) to global nodes (N, 6).
    Each (element, local-node) pair contributes to the global node.
    """
    E_elem     = elem_conn.shape[0]
    node_sig   = torch.zeros(N, 6, device=dev, dtype=torch.float64)
    node_count = torch.zeros(N,    device=dev, dtype=torch.float64)
    for k in range(npe):
        idx = elem_conn[:, k]  # (E_elem,)
        node_sig.scatter_add_(0, idx.unsqueeze(1).expand(-1, 6), elem_nodal_sig[:, k, :])
        node_count.scatter_add_(0, idx, torch.ones(E_elem, device=dev, dtype=torch.float64))
    return node_sig / node_count.clamp(min=1).unsqueeze(1)  # (N, 6)


def _vm_from_nodal(node_sig):
    """von Mises from (N, 6) Voigt stress."""
    s = node_sig
    return torch.sqrt(
        0.5 * ((s[:,0]-s[:,1])**2 + (s[:,1]-s[:,2])**2 + (s[:,2]-s[:,0])**2
               + 6*(s[:,3]**2 + s[:,4]**2 + s[:,5]**2)) + 1e-30
    )


def _get_D(dev, E_mod, nu, elem_E, elem_nu, n_elem):
    """Return D matrix: (6,6) for uniform, (E_elem,6,6) for per-element."""
    if elem_E is not None:
        E_arr  = elem_E.to(device=dev, dtype=torch.float64)
        nu_arr = (elem_nu.to(device=dev, dtype=torch.float64)
                  if elem_nu is not None
                  else torch.full((n_elem,), nu, device=dev, dtype=torch.float64))
        return _D_batch(E_arr, nu_arr)   # (E_elem, 6, 6)
    return _D(E_mod, nu)                 # (6, 6)


def compute_nodal_stress(
    disp:      torch.Tensor,              # (N, 3)
    ref_pos:   torch.Tensor,              # (N, 3)
    elem_conn: torch.Tensor,              # (E_elem, npe)  int64
    npe:       int,                       # 8, 10, or 20
    E_mod:     float = 70e9,
    nu:        float = 0.33,
    elem_E:    torch.Tensor = None,       # (E_elem,) per-element E; overrides E_mod
    elem_nu:   torch.Tensor = None,       # (E_elem,) per-element nu; overrides nu
) -> torch.Tensor:                        # (N, 6)  full stress tensor
    """
    Nodal stress tensor (N, 6) from displacement via FEM shape functions.

    For C3D8: uses Gauss-to-node Lagrange extrapolation (matches Abaqus ELEMENT_NODAL).
    For C3D20/C3D10: Gauss-point averaging per element then scatter to nodes.
    Runs in float64; result cast back to disp.dtype.
    """
    if npe == 8:
        gauss_pts = _gauss_c3d8()
    elif npe == 20:
        gauss_pts = _gauss_c3d20()
    elif npe == 10:
        gauss_pts = _gauss_c3d10()
    else:
        raise ValueError(f"Unsupported element type with {npe} nodes per element")

    orig_dtype = disp.dtype
    dev        = disp.device
    N          = disp.shape[0]
    n_elem     = elem_conn.shape[0]

    D_mat  = _get_D(dev, E_mod, nu, elem_E, elem_nu, n_elem)
    disp_f = disp.to(torch.float64)
    ref_f  = ref_pos.to(device=dev, dtype=torch.float64)

    xe = ref_f[elem_conn]    # (E, npe, 3)
    ue = disp_f[elem_conn]   # (E, npe, 3)

    if npe == 8:
        # Lagrange extrapolation: (E, 8, 6) -> (N, 6)
        elem_nodal_sig = _elem_nodal_stress_c3d8(xe, ue, gauss_pts, D_mat)
        node_sig = _scatter_elem_nodal_to_nodes(elem_nodal_sig, elem_conn, N, npe, dev)
    else:
        # Gauss-point average per element, then scatter
        elem_sig = _elem_stress(xe, ue, gauss_pts, D_mat)
        node_sig = _scatter_elem_to_nodes(elem_sig, elem_conn, N, npe, dev)

    return node_sig.to(dtype=orig_dtype)


def compute_nodal_vm_stress(
    disp:      torch.Tensor,              # (N, 3)
    ref_pos:   torch.Tensor,              # (N, 3)
    elem_conn: torch.Tensor,              # (E_elem, npe)  int64
    npe:       int,                       # 8, 10, or 20
    E_mod:     float = 70e9,
    nu:        float = 0.33,
    elem_E:    torch.Tensor = None,       # (E_elem,) per-element E; overrides E_mod
    elem_nu:   torch.Tensor = None,       # (E_elem,) per-element nu; overrides nu
) -> torch.Tensor:                        # (N,)  same dtype as disp
    """
    Nodal von Mises stress from displacement via FEM shape functions.

    Supports both uniform material (scalar E_mod/nu) and per-element material
    properties (elem_E / elem_nu tensors). Per-element takes priority when given.
    For C3D8: uses Gauss-to-node extrapolation matching Abaqus ELEMENT_NODAL.
    Runs in float64, cast back on return.
    """
    node_sig = compute_nodal_stress(
        disp, ref_pos, elem_conn, npe,
        E_mod=E_mod, nu=nu, elem_E=elem_E, elem_nu=elem_nu,
    )
    return _vm_from_nodal(node_sig.to(torch.float64)).to(dtype=disp.dtype)