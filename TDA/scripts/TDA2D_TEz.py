import gc
import meep as mp
import meep.adjoint as mpa
import numpy as np
import scipy.interpolate as spi

# 1. Situation Set
# ======================================
resolution = 20
nx, ny = 5, 5
T_f = 1000

res_factor = 4
sim_res = resolution * res_factor
cell_size = mp.Vector3(6, 6)
pml_layers = [mp.PML(1.0)]

air = mp.Medium(index=1.0)
silicon = mp.Medium(index=3.48)

dx = 1.0 / resolution
dy = 1.0 / resolution

sim_dt = 0.5 / sim_res
sim_dx = 1.0 / sim_res
sim_dy = 1.0 / sim_res

sim_shift_x = 0.5 * sim_dx
sim_shift_y = 0.5 * sim_dy

shift_x = 0.0 if nx % 2 == 0 else 0.5 * dx
shift_y = 0.0 if ny % 2 == 0 else 0.5 * dy

design_center = mp.Vector3(shift_x, shift_y)
sim_design_center = mp.Vector3(sim_shift_x, sim_shift_y)

design_region_size = mp.Vector3(nx * dx, ny * dy)

design_variables = mp.MaterialGrid(mp.Vector3(nx, ny), air, silicon)

design_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=sim_design_center, size=design_region_size)
)

geometry = [
    mp.Block(
        center=design_center,
        size=design_region_size,
        material=design_variables
    )
]

sim_geometry = [
    mp.Block(
        center=sim_design_center,
        size=design_region_size,
        material=design_variables
    )
]


fcen = 1.0 / 1.55
fwd_source = [mp.Source(
    mp.GaussianSource(frequency=fcen, fwidth=0.2 * fcen),
    component=mp.Hz,
    center=mp.Vector3(-1.5 + sim_shift_x, sim_shift_y),
    size=mp.Vector3(0, 0)
)]

monitor_position = mp.Vector3(1.5 + sim_shift_x, sim_shift_y)

coords_x = [sim_design_center.x + (i - (nx - 1) / 2) * dx for i in range(nx)]
coords_y = [sim_design_center.y + (j - (ny - 1) / 2) * dy for j in range(ny)]

# Random initial design
np.random.seed(42)
x0 = np.random.rand(nx * ny)
design_variables.update_weights(x0)


# 2. Forward Run
# ======================================
sim_fwd = mp.Simulation(cell_size=cell_size, boundary_layers=pml_layers, geometry=sim_geometry,
                        sources=fwd_source, resolution=sim_res)

ex_fields_fwd_time = []
ey_fields_fwd_time = []
h_monitor_time = []

def record_fwd_efield(sim):
    ex_data = np.array([
        [sim.get_field_point(mp.Ex, mp.Vector3(x, y)) for y in coords_y]
        for x in coords_x
    ])

    ey_data = np.array([
        [sim.get_field_point(mp.Ey, mp.Vector3(x, y)) for y in coords_y]
        for x in coords_x
    ])

    ex_fields_fwd_time.append(ex_data)
    ey_fields_fwd_time.append(ey_data)

    hz_mon = sim.get_field_point(mp.Hz, monitor_position)
    h_monitor_time.append(hz_mon)

sim_fwd.run(record_fwd_efield, until=T_f)

Ex_fwd_history = np.array(ex_fields_fwd_time)
Ey_fwd_history = np.array(ey_fields_fwd_time)
H_monitor_history = np.array(h_monitor_time)
ex_fields_fwd_time.clear()
ey_fields_fwd_time.clear()
h_monitor_time.clear()
sim_fwd.reset_meep()
del sim_fwd
gc.collect()


# 3. Adjoint Run
# ======================================
adj_signal = - H_monitor_history[::-1]

t_array = np.linspace(0, T_f, len(H_monitor_history))
adj_interp = spi.interp1d(t_array, adj_signal, kind = 'cubic', fill_value=0j, bounds_error=False)

def adj_src_func(t):
    try:
        return complex(adj_interp(t))
    except Exception:
        return 0j

# adjoint source
adj_source = [mp.Source(mp.CustomSource(src_func=adj_src_func), component=mp.Hz, center=monitor_position)]

sim_adj = mp.Simulation(cell_size=cell_size, boundary_layers=pml_layers,
                        geometry=sim_geometry, sources=adj_source, resolution=sim_res)

ex_fields_adj_time = []
ey_fields_adj_time = []

def record_adj_efield(sim):
    ex_data = np.array([
        [sim.get_field_point(mp.Ex, mp.Vector3(x, y)) for y in coords_y]
        for x in coords_x
    ])

    ey_data = np.array([
        [sim.get_field_point(mp.Ey, mp.Vector3(x, y)) for y in coords_y]
        for x in coords_x
    ])

    ex_fields_adj_time.append(ex_data)
    ey_fields_adj_time.append(ey_data)

sim_adj.run(record_adj_efield, until=T_f)

# adjoint field
Ex_adj_history = np.array(ex_fields_adj_time)
Ey_adj_history = np.array(ey_fields_adj_time)
ex_fields_adj_time.clear()
ey_fields_adj_time.clear()
sim_adj.reset_meep()
del sim_adj
del adj_source
del adj_signal
del t_array
del adj_interp
gc.collect()


# 4. Gradient Calculation
# ======================================
pixel_area = dx * dy

eps_silicon = 3.48 ** 2
eps_air = 1.0 ** 2
deps_dp = eps_silicon - eps_air

gradient_exact = np.zeros_like(Ex_fwd_history[0], dtype=np.complex128)
for tidx in range(Ex_fwd_history.shape[0]):
    if tidx == 0:
        dEx_dt_t = (Ex_fwd_history[1] - Ex_fwd_history[0]) / sim_dt
        dEy_dt_t = (Ey_fwd_history[1] - Ey_fwd_history[0]) / sim_dt
    elif tidx == Ex_fwd_history.shape[0] - 1:
        dEx_dt_t = (Ex_fwd_history[-1] - Ex_fwd_history[-2]) / sim_dt
        dEy_dt_t = (Ey_fwd_history[-1] - Ey_fwd_history[-2]) / sim_dt
    else:
        dEx_dt_t = (Ex_fwd_history[tidx + 1] - Ex_fwd_history[tidx - 1]) / (2 * sim_dt)
        dEy_dt_t = (Ey_fwd_history[tidx + 1] - Ey_fwd_history[tidx - 1]) / (2 * sim_dt)

    gradient_exact += (
        Ex_adj_history[-1 - tidx] * dEx_dt_t
        + Ey_adj_history[-1 - tidx] * dEy_dt_t
    ) * deps_dp

gradient_exact *= sim_dt * pixel_area
del dEx_dt_t
del dEy_dt_t
del Ex_fwd_history
del Ey_fwd_history
del Ex_adj_history
del Ey_adj_history
gc.collect()

print("======================================")
print("Gradient Matrix:")
print(gradient_exact.real)


# 5. Gradient Verification via Finite Difference Method
# ======================================
print("\n--- FDM verification started ---")

J0 = 0.5 * np.sum(np.abs(H_monitor_history)**2) * sim_dt
del H_monitor_history

fdm_gradientss = []

for idx_x in range(nx):
    fdm_gradients = []

    for idx_y in range(ny):
        delta_eps = 1e-4

        x_pert = x0.copy()
        flat_idx = idx_x * ny + idx_y
        x_pert[flat_idx] += delta_eps

        design_variables.update_weights(x_pert)

        sim_fdm = mp.Simulation(cell_size=cell_size, boundary_layers=pml_layers, geometry=sim_geometry,
                                sources=fwd_source, resolution=sim_res)

        J1_accum = [0.0]
        def record_fdm_efield(sim):
            hz_mon = sim.get_field_point(mp.Hz, monitor_position)
            J1_accum[0] += np.abs(hz_mon) ** 2

        sim_fdm.run(record_fdm_efield, until=T_f)
        J1 = 0.5 * J1_accum[0] * sim_dt

        fdm_gradient = (J1 - J0) / delta_eps
        fdm_gradients.append(fdm_gradient)

        sim_fdm.reset_meep()
        del sim_fdm
        del record_fdm_efield
        del J1_accum
        del x_pert
        gc.collect()

    fdm_gradientss.append(fdm_gradients)

fdm_matrix = np.array(fdm_gradientss)
del fdm_gradientss
print(fdm_matrix)

for ix in range(nx):
    for iy in range(ny):
        print("======================================")
        print(f"Pixel ({ix}, {iy}):")
        print(f"Adjoint Gradient: {gradient_exact[ix, iy].real}")
        print(f"FDM Numerical Gradient         : {fdm_matrix[ix, iy]}")
        print(f"Ratio (FDM / Adjoint)          : {fdm_matrix[ix, iy] / gradient_exact[ix, iy].real}")



# 6. Normalized Gradient Comparison
# ======================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("\n=== Normalized Gradient Comparison ===")

adj_real = gradient_exact.real
fdm_real = fdm_matrix.real if np.iscomplexobj(fdm_matrix) else fdm_matrix.astype(float)

adj_max = np.max(np.abs(adj_real)) + 1e-30
fd_max  = np.max(np.abs(fdm_real)) + 1e-30
adj_norm = adj_real / adj_max
fd_norm  = fdm_real / fd_max

a_flat = adj_norm.flatten()
f_flat = fd_norm.flatten()

corr_all   = np.corrcoef(a_flat, f_flat)[0, 1]
cosine_all = np.dot(a_flat, f_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(f_flat) + 1e-30)
rmse_all   = np.sqrt(np.mean((a_flat - f_flat) ** 2))

# interior-only comparison (exclude 1-pixel border)
m = 1
a_inner = adj_norm[m:-m, m:-m].flatten()
f_inner = fd_norm[m:-m, m:-m].flatten()
corr_in   = np.corrcoef(a_inner, f_inner)[0, 1]
cosine_in = np.dot(a_inner, f_inner) / (np.linalg.norm(a_inner) * np.linalg.norm(f_inner) + 1e-30)
rmse_in   = np.sqrt(np.mean((a_inner - f_inner) ** 2))

edge_mask = np.ones((nx, ny), dtype=bool)
edge_mask[m:-m, m:-m] = False
rmse_edge = np.sqrt(np.mean((adj_norm[edge_mask] - fd_norm[edge_mask]) ** 2))

print("=" * 60)
print(f"  |Adj|_max = {adj_max:.6e}")
print(f"  |FD |_max = {fd_max:.6e}")
print(f"  --- All pixels ({nx*ny}) ---")
print(f"  Correlation:   {corr_all:.6f}")
print(f"  Cosine sim:    {cosine_all:.6f}")
print(f"  RMSE (normed): {rmse_all:.6f}")
print(f"  --- Interior only ({a_inner.size}, exclude {m}px border) ---")
print(f"  Correlation:   {corr_in:.6f}")
print(f"  Cosine sim:    {cosine_in:.6f}")
print(f"  RMSE (normed): {rmse_in:.6f}")
print(f"  --- Edge-only RMSE: {rmse_edge:.6f} ---")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
cmap = "RdBu_r"

# (a) Adjoint gradient (normalized)
ax = axes[0, 0]
im = ax.imshow(adj_norm.T, origin="lower", cmap=cmap, vmin=-1, vmax=1, aspect="equal")
ax.set_title("(a) Adjoint Gradient (norm)", fontsize=13, fontweight="bold")
ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
ax.set_xticks(range(nx)); ax.set_yticks(range(ny))
plt.colorbar(im, ax=ax, shrink=0.85)
for i in range(nx):
    for j in range(ny):
        ax.text(i, j, f"{adj_norm[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(adj_norm[i,j]) > 0.5 else "black")

# (b) FD gradient (normalized)
ax = axes[0, 1]
im = ax.imshow(fd_norm.T, origin="lower", cmap=cmap, vmin=-1, vmax=1, aspect="equal")
ax.set_title(f"(b) FD Gradient (norm, δ={delta_eps})", fontsize=13, fontweight="bold")
ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
ax.set_xticks(range(nx)); ax.set_yticks(range(ny))
plt.colorbar(im, ax=ax, shrink=0.85)
for i in range(nx):
    for j in range(ny):
        ax.text(i, j, f"{fd_norm[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(fd_norm[i,j]) > 0.5 else "black")

# (c) Difference (Adj − FD)
ax = axes[1, 0]
diff = adj_norm - fd_norm
im = ax.imshow(diff.T, origin="lower", cmap=cmap, vmin=-0.1, vmax=0.1, aspect="equal")
ax.set_title("(c) Difference (Adj − FD)", fontsize=13, fontweight="bold")
ax.set_xlabel("x cell"); ax.set_ylabel("y cell")
ax.set_xticks(range(nx)); ax.set_yticks(range(ny))
plt.colorbar(im, ax=ax, shrink=0.85)
dmax = max(np.max(np.abs(diff)), 0.01)
for i in range(nx):
    for j in range(ny):
        fc = "white" if abs(diff[i, j]) > 0.5 * dmax else "black"
        ax.text(i, j, f"{diff[i,j]:.2f}", ha="center", va="center", fontsize=7, color=fc)

# (d) Scatter: Adj vs FD (normalized)
ax = axes[1, 1]
ax.scatter(f_flat, a_flat, s=40, alpha=0.7, c="steelblue", edgecolors="k", linewidths=0.5)
lim = 1.15
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal y=x")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("FD (normalized)", fontsize=12)
ax.set_ylabel("Adjoint (normalized)", fontsize=12)
ax.set_title("(d) Scatter: Adj vs FD (norm)", fontsize=13, fontweight="bold")
ax.set_aspect("equal")
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.95,
        f"Corr={corr_all:.3f}  Cosine={cosine_all:.3f}\n"
        f"RMSE={rmse_all:.3f}  N={nx*ny}\n"
        f"Interior: Corr={corr_in:.3f}  RMSE={rmse_in:.3f}",
        transform=ax.transAxes, fontsize=8, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

fig.suptitle(
    f"Normalized Gradient Comparison  |  Design {nx}×{ny}  |  FOM(J0)={J0:.4e}  |  δ={delta_eps}",
    fontsize=13, fontweight="bold", y=0.99,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out_path = "TDA4TEz_PJH_ver1_normalized_comparison(4).png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nNormalized comparison image saved: {out_path}")

np.savez(
    "TDA4TEz_PJH_ver1_gradient_data(4).npz",
    grad_adj=adj_real, grad_fd=fdm_real,
    adj_norm=adj_norm, fd_norm=fd_norm,
    fom=J0, rho=x0,
)
print("Data saved: TDA4TEz_PJH_ver1_gradient_data(4).npz")
