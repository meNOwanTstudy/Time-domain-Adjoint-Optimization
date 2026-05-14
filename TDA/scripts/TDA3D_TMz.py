import meep as mp
import numpy as np
import meep.adjoint as mpa
import scipy.interpolate as spi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import gc

# 1. Situation Set
# ======================================
resolution = 20
nx_g, ny_g, nz_g = 4, 4, 4
group_size = 2
nx, ny, nz = nx_g * group_size, ny_g * group_size, nz_g * group_size
T_f = 1000

mat_res = resolution * group_size
cell_size = mp.Vector3(6, 6, 6)
pml_layers = [mp.PML(1.0)]

air = mp.Medium(index=1.0)
silicon = mp.Medium(index=3.48)

T_f = 1000
dx = 1.0 / mat_res
dy = 1.0 / mat_res
dz = 1.0 / mat_res
dt = 0.5 / resolution

sim_dx = 1.0 / resolution
sim_dy = 1.0 / resolution
sim_dz = 1.0 / resolution

shift_x = ((nx % 4) - 1) * 0.25 * sim_dx
shift_y = ((ny % 4) - 1) * 0.25 * sim_dy
shift_z = ((nz % 4) - 1) * 0.25 * sim_dz

design_center = mp.Vector3(shift_x, shift_y, shift_z)
design_region_size = mp.Vector3(nx * dx, ny * dy, nz * dz)

design_variables = mp.MaterialGrid(mp.Vector3(nx, ny, nz), air, silicon)

design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=design_center, size=design_region_size)
)

geometry = [
    mp.Block(
        center=design_center,
        size=design_region_size,
        material=design_variables
    )
]

def snap_to_E_field(val, res):
    return (np.round(val * res) + 0.5) / res

mon_pos = mp.Vector3(snap_to_E_field(1.5 + shift_x, resolution),
                     snap_to_E_field(shift_y, resolution),
                     snap_to_E_field(shift_z, resolution))

src_pos = mp.Vector3(snap_to_E_field(-1.5 + shift_x, resolution),
                     snap_to_E_field(shift_y, resolution),
                     snap_to_E_field(shift_z, resolution))

fcen = 1.0 / 1.55
fwd_source = [mp.Source(
    mp.GaussianSource(frequency=fcen, fwidth=0.2 * fcen),
    component=mp.Ez,
    center=src_pos
    )
]

sim_coords_x = [design_center.x + (i - (nx - 1) / 2) * dx for i in range(nx)]
sim_coords_y = [design_center.y + (j - (ny - 1) / 2) * dy for j in range(ny)]
sim_coords_z = [design_center.z + (k - (nz - 1) / 2) * dz for k in range(nz)]

# Random Initialization
np.random.seed(42)
x_grouped = np.random.rand(nx_g * ny_g * nz_g)
x0_voxel = x_grouped.reshape(nx_g, ny_g, nz_g).repeat(group_size, axis=0)\
    .repeat(group_size, axis=1).repeat(group_size, axis=2).flatten()
design_variables.update_weights(x0_voxel)


# 2. Forward Run
# ======================================
sim_fwd = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=fwd_source,
    resolution=resolution
)

num_steps = int(T_f / dt) + 1
Ex_fwd = np.zeros((num_steps, nx, ny, nz), dtype=complex)
Ey_fwd = np.zeros((num_steps, nx, ny, nz), dtype=complex)
Ez_fwd = np.zeros((num_steps, nx, ny, nz), dtype=complex)
Ez_mon = np.zeros(num_steps, dtype=complex)

def record_fwd(sim):
    step = int(np.round(sim.round_time() / dt))
    if step >= num_steps: return

    for i, x in enumerate(sim_coords_x):
        for j, y in enumerate(sim_coords_y):
            for k, z in enumerate(sim_coords_z):
                if i % 2 != 0 and j % 2 == 0 and k % 2 == 0:
                    Ex_fwd[step, i, j, k] = sim.get_field_point(mp.Ex, mp.Vector3(x, y, z))
                elif i % 2 == 0 and j % 2 != 0 and k % 2 == 0:
                    Ey_fwd[step, i, j, k] = sim.get_field_point(mp.Ey, mp.Vector3(x, y, z))
                elif i % 2 == 0 and j % 2 == 0 and k % 2 != 0:
                    Ez_fwd[step, i, j, k] = sim.get_field_point(mp.Ez, mp.Vector3(x, y, z))

    Ez_mon[step] = sim.get_field_point(mp.Ez, mon_pos)

print("\n--- Forward Run Starting ---")
start_fwd = time.time()
sim_fwd.run(mp.at_every(dt, record_fwd), until=T_f)
print(f"Forward Run Done ({time.time() - start_fwd:.2f} s)")

sim_fwd.reset_meep()
del sim_fwd
gc.collect()


# 3. Adjoint Run
# ======================================
adj_sig = np.conj(Ez_mon[::-1])
t_arr = np.linspace(0, T_f, len(Ez_mon))
adj_intp = spi.interp1d(t_arr, adj_sig, kind = 'linear', fill_value=0j, bounds_error=False)
del adj_sig, t_arr
gc.collect()

adj_src = [mp.Source(
    mp.CustomSource(src_func=lambda t: complex(adj_intp(t))),
    component=mp.Ez,
    center=mon_pos
)]

sim_adj = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=adj_src,
    resolution=resolution
)

Ex_adj_rev = np.zeros((num_steps, nx, ny, nz), dtype=complex)
Ey_adj_rev = np.zeros((num_steps, nx, ny, nz), dtype=complex)
Ez_adj_rev = np.zeros((num_steps, nx, ny, nz), dtype=complex)

def record_adj(sim):
    step = int(np.round(sim.round_time() / dt))
    if step >= num_steps: return

    for i, x in enumerate(sim_coords_x):
        for j, y in enumerate(sim_coords_y):
            for k, z in enumerate(sim_coords_z):
                if i % 2 != 0 and j % 2 == 0 and k % 2 == 0:
                    Ex_adj_rev[step, i, j, k] = sim.get_field_point(mp.Ex, mp.Vector3(x, y, z))
                elif i % 2 == 0 and j % 2 != 0 and k % 2 == 0:
                    Ey_adj_rev[step, i, j, k] = sim.get_field_point(mp.Ey, mp.Vector3(x, y, z))
                elif i % 2 == 0 and j % 2 == 0 and k % 2 != 0:
                    Ez_adj_rev[step, i, j, k] = sim.get_field_point(mp.Ez, mp.Vector3(x, y, z))

print("\n--- Adjoint Run Starting ---")
start_adj = time.time()
sim_adj.run(mp.at_every(dt, record_adj), until=T_f)
print(f"Adjoint Run Done ({time.time() - start_adj:.2f} s)")

sim_adj.reset_meep()
del sim_adj
del adj_src
del adj_intp
gc.collect()

Ex_adj_rev = Ex_adj_rev[::-1]
Ey_adj_rev = Ey_adj_rev[::-1]
Ez_adj_rev = Ez_adj_rev[::-1]


# 4. Gradient Calculation
# ======================================
print("\n--- Gradient Calculation Starting ---")
voxel_volume = dx * dy * dz
deps = (3.48**2) - (1.0**2)
grad_vox = np.zeros((nx, ny, nz), dtype=complex)

for tidx in range(num_steps):
    if tidx == 0:
        dE_dt_t = (Ex_fwd[1] - Ex_fwd[0]) / dt
    elif tidx == num_steps - 1:
        dE_dt_t = (Ex_fwd[-1] - Ex_fwd[-2]) / dt
    else:
        dE_dt_t = (Ex_fwd[tidx + 1] - Ex_fwd[tidx - 1]) / (2 * dt)
    grad_vox += dE_dt_t * Ex_adj_rev[tidx] * deps * dt * voxel_volume
del Ex_fwd, Ex_adj_rev, dE_dt_t
gc.collect()

for tidx in range(num_steps):
    if tidx == 0:
        dE_dt_t = (Ey_fwd[1] - Ey_fwd[0]) / dt
    elif tidx == num_steps - 1:
        dE_dt_t = (Ey_fwd[-1] - Ey_fwd[-2]) / dt
    else:
        dE_dt_t = (Ey_fwd[tidx + 1] - Ey_fwd[tidx - 1]) / (2 * dt)
    grad_vox += dE_dt_t * Ey_adj_rev[tidx] * deps * dt * voxel_volume
del Ey_fwd, Ey_adj_rev, dE_dt_t
gc.collect()

for tidx in range(num_steps):
    if tidx == 0:
        dE_dt_t = (Ez_fwd[1] - Ez_fwd[0]) / dt
    elif tidx == num_steps - 1:
        dE_dt_t = (Ez_fwd[-1] - Ez_fwd[-2]) / dt
    else:
        dE_dt_t = (Ez_fwd[tidx + 1] - Ez_fwd[tidx - 1]) / (2 * dt)
    grad_vox += dE_dt_t * Ez_adj_rev[tidx] * deps * dt * voxel_volume
del Ez_fwd, Ez_adj_rev, dE_dt_t
gc.collect()

grad_exact = np.zeros((nx_g, ny_g, nz_g))
for i in range(nx_g):
    for j in range(ny_g):
        for k in range(nz_g):
            sub_grad = grad_vox[i*group_size:(i+1)*group_size,
                                j*group_size:(j+1)*group_size,
                                k*group_size:(k+1)*group_size]
            grad_exact[i, j, k] = np.sum(sub_grad).real


# 5. FDM Verification
# ======================================
print(f"\n--- FDM Verification Starting ({nx_g}x{ny_g}x{nz_g} Groups) ---")
start_fdm = time.time()
J0 = 0.5 * np.sum(np.abs(Ez_mon)**2) * dt
del Ez_mon
fdm_grad = np.zeros((nx_g, ny_g, nz_g))

sim_fdm = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=fwd_source,
    resolution=resolution
)

for ix in range(nx_g):
    for iy in range(ny_g):
        for iz in range(nz_g):
            x_p_g = x_grouped.copy().reshape(nx_g, ny_g, nz_g)
            x_p_g[ix, iy, iz] += 1e-4
            x_p_px = x_p_g.repeat(group_size, axis=0).repeat(group_size, axis=1).repeat(group_size, axis=2).flatten()
            design_variables.update_weights(x_p_px)

            sim_fdm.reset_meep()
            J1_accum = [0.0]

            def record_ez_mon(sim):
                st = int(np.round(sim.round_time() / dt))
                if st < num_steps:
                    ez_mon = sim.get_field_point(mp.Ez, mon_pos)
                    J1_accum[0] += np.abs(ez_mon) ** 2

            sim_fdm.run(mp.at_every(dt, record_ez_mon), until=T_f)
            J1 = 0.5 * J1_accum[0] * dt
            fdm_grad[ix, iy, iz] = (J1 - J0) / 1e-4

            progress = ix * (ny_g * nz_g) + iy * nz_g + iz + 1
            print(f"[{progress}/{nx_g*ny_g*nz_g}] Group ({ix},{iy},{iz}) Done")
            del record_ez_mon
            del J1_accum
            del x_p_g
            del x_p_px
            gc.collect()

sim_fdm.reset_meep()
del sim_fdm
gc.collect()
print(f"FDM Verification Done ({time.time() - start_fdm:.2f} s)")


# 6. Visualization & Statistics
# ======================================
slice_idx = nz_g // 2
a_slice = grad_exact[:, :, slice_idx]
f_slice = fdm_grad[:, :, slice_idx]

a_n = grad_exact / (np.max(np.abs(grad_exact)) + 1e-30)
f_n = fdm_grad / (np.max(np.abs(fdm_grad)) + 1e-30)

a_slice_n = a_n[:, :, slice_idx]
f_slice_n = f_n[:, :, slice_idx]

c_all = np.corrcoef(a_n.flatten(), f_n.flatten())[0, 1]
rmse_all = np.sqrt(np.mean((a_n - f_n) ** 2))

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
cmap = "RdBu_r"

titles = [f"(a) Adj (Slice z={slice_idx})", f"(b) FD (Slice z={slice_idx})", "(c) Diff (Adj-FD)"]
datas = [a_slice_n, f_slice_n, a_slice_n - f_slice_n]

for ax, data, title in zip(axes.flat[:3], datas, titles):
    vmin, vmax = (-1, 1) if "Diff" not in title else (-0.1, 0.1)
    im = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(nx_g)); ax.set_yticks(range(ny_g))
    plt.colorbar(im, ax=ax, shrink=0.85)

    for i in range(nx_g):
        for j in range(ny_g):
            ax.text(i, j, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(data[i,j]) > 0.5 else "black")

# Scatter plot
ax = axes[1, 1]
ax.scatter(f_n.flatten(), a_n.flatten(), s=40, alpha=0.7, c="steelblue", edgecolors="k", linewidths=0.5)
lim = 1.15
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("FD (norm)"); ax.set_ylabel("Adj (norm)")
ax.set_title("(d) Scatter: Adj vs FD (Full 3D)", fontweight="bold")
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

stats_text = (f"All 3D Groups:\nCorr={c_all:.4f}\nRMSE={rmse_all:.4f}")
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

fig.suptitle(f"3D Gradient Comparison ({nx_g}x{ny_g}x{nz_g}) | FOM(J0)={J0:.4e}", fontsize=14, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_path = f"TDA3D_PJH_ver0c_comparison_{nx_g}x{ny_g}x{nz_g}.png"
plt.savefig(out_path, dpi=150)

print(f"\n3D Correlation: {c_all:.4f}, RMSE: {rmse_all:.4f}")
print(f"Image saved: {out_path}")
