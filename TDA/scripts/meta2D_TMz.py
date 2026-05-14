# 2D metalens optimization with conditional field-history recording and MMA.
# Wavelength band: 400-800 nm, single focal point, NA = 0.9.

import meep as mp
import numpy as np
import nlopt
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import gc

# 1. Env Setup
# ======================================
resolution = 40
sx, sy = 6.25, 16.75
pml_thickness = 1.0
l_min, l_max = 0.4, 0.8
cell_size = mp.Vector3(sx + 2 * pml_thickness + 2 * l_max, sy + 2 * pml_thickness)
pml_layers = [mp.PML(pml_thickness)]

air = mp.Medium(index=1.0)
TiO2 = mp.Medium(index=2.44)

design_w, design_h = sx, 0.5
nx, ny = int(np.round(design_w * resolution)), int(np.round(design_h * resolution))

NA = 0.9
focal_length = (design_w/2) * np.sqrt((1-NA**2)/(NA**2))

T_f = 200
dx, dy = 1.0 / resolution, 1.0 / resolution
shift_x = 0.5 * dx if nx % 2 == 0 else 0.0
shift_y = 0.5 * dy if ny % 2 == 0 else 0.0

design_center = mp.Vector3(shift_x, shift_y)
monitor_position = mp.Vector3(shift_x, shift_y + design_h/2 + focal_length)
design_region_size = mp.Vector3(design_w, design_h)

design_variables = mp.MaterialGrid(mp.Vector3(nx, ny), air, TiO2)
geometry = [mp.Block(center=design_center, size=design_region_size, material=design_variables)]


fcen = (1/l_max + 1/l_min) / 2
fwidth = (1/l_min - 1/l_max)
fwd_source = [mp.Source(mp.GaussianSource(frequency=fcen, fwidth=fwidth), component=mp.Ez,
                        center=mp.Vector3(shift_x, shift_y - 4.0), size=mp.Vector3(sx + 2 * pml_thickness + 2 * l_max, 0))]

coords_x = [design_center.x + (i - (nx - 1) / 2) * dx for i in range(nx)]
coords_y = [design_center.y + (j - (ny - 1) / 2) * dy for j in range(ny)]
history_fom = []

def capture_field_map(weights):
    design_variables.update_weights(weights.reshape(nx, ny))
    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=pml_layers,
                        geometry=geometry, sources=fwd_source, resolution=resolution)
    sim_vol = mp.Volume(center=mp.Vector3(0, 0), size=cell_size)
    dft_obj = sim.add_dft_fields([mp.Ez], fcen, fcen, 1, where=sim_vol)
    sim.run(until=T_f)
    ez_data = sim.get_dft_array(dft_obj, mp.Ez, 0)
    sim.reset_meep()
    del sim, dft_obj, sim_vol
    gc.collect()
    return ez_data

# 2. Optimization Functions
# ======================================
dt = 0.5 / resolution

# Sampling interval for adjoint
N_sampling = 1

def calculate_fom_and_grad(x, need_gradient=True):
    """
    Run forward and adjoint simulations and return the FoM and gradient.
    When need_gradient is False, only the monitor history is recorded.
    """
    design_variables.update_weights(x.reshape(nx, ny))

    sim_fwd = mp.Simulation(cell_size=cell_size,
                            boundary_layers=pml_layers,
                            geometry=geometry, sources=fwd_source, resolution=resolution)
    e_mon_t = []
    e_fld_t = [] if need_gradient else None

    def record_fwd(s):
        e_mon_t.append(s.get_field_point(mp.Ez, monitor_position))

    if need_gradient:
        def record_fld(s):
            ez = np.array([[s.get_field_point(mp.Ez, mp.Vector3(xv, yv)) for yv in coords_y] for xv in coords_x])
            e_fld_t.append(ez)

        sim_fwd.run(record_fwd, mp.at_every(dt * N_sampling, record_fld), until=T_f)
    else:
        sim_fwd.run(record_fwd, until=T_f)

    T_actual, E_mon_hist = sim_fwd.round_time(), np.array(e_mon_t)
    e_mon_t.clear()
    sim_fwd.reset_meep()
    del sim_fwd
    gc.collect()

    objective_value = 0.5 * np.sum(np.abs(E_mon_hist)**2) * dt

    gradient = None
    if need_gradient:
        print("Computing Adjoint Gradient ...")
        E_fwd_hist = np.array(e_fld_t)
        e_fld_t.clear()
        del e_fld_t
        dt_eff = dt * N_sampling

        adj_sig = E_mon_hist[::-1].copy()
        t_adj = np.linspace(0, T_actual, len(adj_sig))
        adj_interp_obj = spi.interp1d(t_adj, adj_sig,
                                     kind='cubic', fill_value=0j, bounds_error=False)
        del E_mon_hist, adj_sig, t_adj
        gc.collect()

        adj_src = [mp.Source(mp.CustomSource(src_func=lambda t: complex(adj_interp_obj(t))),
                             component=mp.Ez, center=monitor_position)]

        sim_adj = mp.Simulation(cell_size=cell_size,
                                boundary_layers=pml_layers,
                                geometry=geometry, sources=adj_src, resolution=resolution)
        grad_2d = np.zeros((nx, ny), dtype=complex)
        adj_status = {'count': 0}

        def accum_adj(s):
            idx = len(E_fwd_hist) - 1 - adj_status['count']
            if idx >= 0:
                if idx == 0:
                    dE_dt_t = (E_fwd_hist[1] - E_fwd_hist[0]) / dt_eff
                elif idx == len(E_fwd_hist) - 1:
                    dE_dt_t = (E_fwd_hist[-1] - E_fwd_hist[-2]) / dt_eff
                else:
                    dE_dt_t = (E_fwd_hist[idx + 1] - E_fwd_hist[idx - 1]) / (2 * dt_eff)

                curr_adj = np.array([[s.get_field_point(mp.Ez, mp.Vector3(xv, yv)) for yv in coords_y] for xv in coords_x])
                grad_2d[:] += curr_adj * dE_dt_t
            adj_status['count'] += 1

        sim_adj.run(mp.at_every(dt_eff, accum_adj), until=T_actual)
        sim_adj.reset_meep()
        del sim_adj, E_fwd_hist, adj_src, adj_interp_obj
        gc.collect()

        grad_factor = (dx * dy * dt_eff * (2.44**2 - 1.0**2))
        gradient = grad_2d.real.flatten() * grad_factor
        del grad_2d
        gc.collect()
    else:
        del E_mon_hist
        gc.collect()

    return objective_value, gradient

def objective_function(x, grad):
    objective_function.count += 1
    it = objective_function.count
    print(f"\n--- Iteration {it} ---")

    if it == 1:
        objective_function.max_fom = -1e10
        objective_function.best_x = np.copy(x)

    need_grad = (grad.size > 0)
    fom, d_fom = calculate_fom_and_grad(x, need_gradient=need_grad)

    if need_grad:
        grad[:] = d_fom

    history_fom.append(fom)
    print(f"Objective Value: {fom:.6f}")

    if fom > objective_function.max_fom:
        objective_function.max_fom = fom
        objective_function.best_x = np.copy(x)
        print(f"*** New Record! ***")

    return fom

objective_function.count = 0

# 3. Optimization Run
# ======================================
n = nx * ny
opt = nlopt.opt(nlopt.LD_MMA, n)
opt.set_lower_bounds(0.0)
opt.set_upper_bounds(1.0)
opt.set_max_objective(objective_function)
opt.set_xtol_rel(1e-25)
opt.set_maxeval(100)

np.random.seed(42)
x_initial = np.random.rand(n)
print("Capturing Initial Field Map...")
initial_ez = capture_field_map(x_initial)

print("\nStarting Optimization (Conditional Field History + MMA)...")
try:
    opt.optimize(x_initial)
except Exception as e:
    print(f"\nOptimization Halted: {e}")

x_opt = objective_function.best_x
print(f"\n>>> Best Structure Recovered: Max FoM = {objective_function.max_fom:.6f}")
final_ez = capture_field_map(x_opt)

# 4. Final Visualization
# ======================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
extent = [-cell_size.x/2, cell_size.x/2, -cell_size.y/2, cell_size.y/2]

axes[0, 0].set_title("Initial Intensity"); axes[0, 0].imshow(np.abs(initial_ez).T**2, cmap='inferno', extent=extent, origin='lower')

x_opt_2d = x_opt.reshape(nx, ny)
axes[0, 1].set_title("Best Structure (Raw Grayscale)"); axes[0, 1].imshow(x_opt_2d.T, cmap='binary', extent=[-sx/2 + shift_x, sx/2 + shift_x, -design_h/2 + shift_y, design_h/2 + shift_y], origin='lower')

axes[1, 0].set_title("Optimized Intensity (Best Structure)"); axes[1, 0].imshow(np.abs(final_ez).T**2, cmap='inferno', extent=extent, origin='lower')
axes[1, 1].set_title("Convergence"); axes[1, 1].plot(range(1, len(history_fom) + 1), history_fom, 'o-b'); plt.savefig("meta2D_PJH_ver0m_optimized.png", dpi=200)
print("Results saved to 'meta2D_PJH_ver0m_optimized.png'.")
