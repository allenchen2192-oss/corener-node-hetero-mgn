"""
visualize_displacement.py
=========================
顯示目前 generate_synthetic_data.py 的每個節點位移與速度隨時間的變化。
使用相同的 seed=42，重現完全相同的係數 a。

輸出：
  1. 終端機：各節點在幾個關鍵時刻的位移
  2. 終端機：速度 target 隨時間的統計（確認是否有時間變化）
  3. displacement_vtu/step_*.vtu：可在 ParaView 開啟
"""

import math
import os

import numpy as np
import torch

# ── 重現 generate_synthetic_data.py 的設定（seed/函數需完全一致）────────────
torch.manual_seed(42)

L, n, T = 0.1, 3, 100
c = torch.linspace(0, L, n)
gx, gy, gz = torch.meshgrid(c, c, c, indexing='ij')
ref_pos = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=1)
X, Y, Z = ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2]

a = (torch.rand(3, 4) - 0.5) * 0.01

# 時間函數（與 generate_synthetic_data.py 完全相同）——互質頻率
def Sx(t): return math.sin(2 * math.pi * t / T)          # 頻率 1
def Sy(t): return math.sin(2 * math.pi * 1.5 * t / T)   # 頻率 1.5
def Sz(t): return math.sin(2 * math.pi * 2.3 * t / T)   # 頻率 2.3

def world_pos_at(t):
    ux = Sx(t) * (a[0,0] + a[0,1]*X + a[0,2]*Y + a[0,3]*Z)
    uy = Sy(t) * (a[1,0] + a[1,1]*X + a[1,2]*Y + a[1,3]*Z)
    uz = Sz(t) * (a[2,0] + a[2,1]*X + a[2,2]*Y + a[2,3]*Z)
    return ref_pos + torch.stack([ux, uy, uz], dim=1)

wp = [world_pos_at(t) for t in range(T)]

# ── 1. 時間函數的曲線（確認唯一性）─────────────────────────────────────────
print("=" * 60)
print("時間函數值（確認每步唯一）:")
print(f"{'t':>4} | {'Sx=sin(2πt/T)':>13} {'Sy=sin(3πt/T)':>14} {'Sz=sin(4.6πt/T)':>16}")
print("-" * 60)
for t in [0, 10, 25, 50, 75, 90, 99]:
    print(f"{t:>4} | {Sx(t):>13.4f} {Sy(t):>14.4f} {Sz(t):>14.4f}")

# 檢查所有 (Sx,Sy,Sz) 組合是否唯一
triplets = [(round(Sx(t),8), round(Sy(t),8), round(Sz(t),8)) for t in range(T)]
unique_count = len(set(triplets))
print(f"\n唯一 (Sx,Sy,Sz) 組合數：{unique_count} / {T}")
if unique_count < T:
    print("  ⚠️  警告：有重複的 (Sx,Sy,Sz)！模型輸入有歧義")
else:
    print("  ✓  所有時間步的 (Sx,Sy,Sz) 均不同，無歧義")

# ── 2. 速度 target 隨時間的變化 ────────────────────────────────────────────
print("\n" + "=" * 60)
print("中心節點 (idx=13, X=0.05,Y=0.05,Z=0.05) 的速度隨時間：")
print(f"{'t':>4} | {'vel_x':>10} {'vel_y':>10} {'vel_z':>10}")
print("-" * 45)
for t in range(T - 1):
    vel = (wp[t+1] - wp[t])[13]
    if t in [0, 10, 24, 25, 26, 49, 50, 51, 74, 75, 98]:
        print(f"{t:>4} | {vel[0].item():>10.5f} {vel[1].item():>10.5f} {vel[2].item():>10.5f}")

# 速度的統計（確認是否隨時間變化）
velocities = torch.stack([wp[t+1] - wp[t] for t in range(T-1)])  # (99, 27, 3)
vel_std_over_time = velocities.std(dim=0).mean(dim=0)  # 每個方向跨時間的 std
print(f"\n速度跨時間的標準差（越大代表時間變化越明顯）:")
print(f"  vel_x std: {vel_std_over_time[0].item():.6f}")
print(f"  vel_y std: {vel_std_over_time[1].item():.6f}")
print(f"  vel_z std: {vel_std_over_time[2].item():.6f}")
if vel_std_over_time.max() < 1e-8:
    print("  ⚠️  警告：速度幾乎不隨時間變化！")
else:
    print("  ✓  速度隨時間有明顯變化")

# ── 3. 匯出 VTU ─────────────────────────────────────────────────────────────
try:
    import meshio
    cells_list = []
    for ci in range(n-1):
        for cj in range(n-1):
            for ck in range(n-1):
                def nd(di,dj,dk): return (ci+di)*n*n+(cj+dj)*n+(ck+dk)
                cells_list.append([nd(0,0,0),nd(1,0,0),nd(1,1,0),nd(0,1,0),
                                    nd(0,0,1),nd(1,0,1),nd(1,1,1),nd(0,1,1)])
    hex_cells = [("hexahedron", np.array(cells_list, dtype=np.int64))]
    ref_np = ref_pos.numpy()
    os.makedirs("displacement_vtu", exist_ok=True)
    for t in range(T):
        pts  = wp[t].numpy()
        disp = pts - ref_np
        mesh = meshio.Mesh(
            points=pts, cells=hex_cells,
            point_data={
                "Displacement"    : disp,
                "Displacement_Mag": np.linalg.norm(disp, axis=1),
            },
        )
        mesh.write(f"displacement_vtu/step_{t:03d}.vtu")
    print(f"\n[saved] displacement_vtu/step_000 ~ step_{T-1:03d}.vtu")
except ImportError:
    print("\n[提示] pip install meshio 後可輸出 VTU")
