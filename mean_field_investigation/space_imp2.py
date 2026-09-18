import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from numba import njit
import sympy as sp
from sympy import Symbol, root

# --- Parameters ---
NSpins = 8
J, mu, H = 0.25, 1.0, np.linspace(0.01, 0.3, 20)
T_right = np.linspace(1.0, 4.9, 20)
T_left = 5.0

H_grid, T_grid = np.meshgrid(H, T_right)
Z_x0 = np.zeros_like(H_grid)
Z_upper = np.full_like(H_grid, np.nan)
Z_middle = np.full_like(H_grid, np.nan)
Z_lower = np.full_like(H_grid, np.nan)

burn_in = 10000
sweeps = 100000
total_sweeps = burn_in + sweeps

@njit
def chain(NSpins, total_sweeps, burn_in, T_left, T_right, J, mu, H):
    spin = np.ones(NSpins, dtype=np.int8)
    current_M = float(np.sum(spin))

    beta_c = 1.0 / T_right
    beta_h = 1.0 / T_left
    m_hist = np.empty(total_sweeps, dtype=np.float64)

    for sweep in range(total_sweeps):
        for _ in range(NSpins):
            r = np.random.randint(0, NSpins)
            nb = spin[(r - 1) % NSpins] + spin[(r + 1) % NSpins]
            dE = 2.0 * spin[r] * (J * nb + mu * H)

            left_aligned = (spin[(r - 1) % NSpins] == spin[r])
            right_aligned = (spin[(r + 1) % NSpins] == spin[r])

            if left_aligned and not right_aligned:
                beta = beta_c
            elif right_aligned and not left_aligned:
                beta = beta_h
            else:
                beta = 0.5 * (beta_h + beta_c)

            if dE <= 0.0 or np.random.random() < np.exp(-dE * beta):
                current_M -= 2 * spin[r]
                spin[r] = -spin[r]

        m_hist[sweep] = current_M / NSpins
    return m_hist


def build_mfc_symbolic_func(T_left, T_right, h, mu, J):
    x = sp.Symbol('x')
    
    beta_c = 1.0 / T_right
    beta_h = 1.0 / T_left
    beta_m = (beta_c + beta_h) / 2.0

    dE_1 = 2.0 * (J * 2 + mu * h)
    p1 = np.exp(-dE_1 * beta_m) if dE_1 > 0 else 1.0

    dE_2 = 2.0 * (mu * h)
    p2 = np.exp(-dE_2 * beta_h) if dE_2 > 0 else 1.0

    dE_3 = -2.0 * (J * 2 + mu * h)
    p3 = np.exp(-dE_3 * beta_m) if dE_3 > 0 else 1.0

    dE_4 = 2.0 * (mu * h)
    p4 = np.exp(-dE_4 * beta_c) if dE_4 > 0 else 1.0

    dE_5 = -2.0 * (mu * h)
    p5 = np.exp(-dE_5 * beta_c) if dE_5 > 0 else 1.0

    dE_6 = -2.0 * (mu * h)
    p6 = np.exp(-dE_6 * beta_h) if dE_6 > 0 else 1.0

    dE_7 = 2.0 * (-2 * J + mu * h)
    p7 = np.exp(-dE_7 * beta_m) if dE_7 > 0 else 1.0

    dE_8 = -2.0 * (-2 * J + mu * h)
    p8 = np.exp(-dE_8 * beta_m) if dE_8 > 0 else 1.0

    eq_1 = ((1 + x)**3) * (1 - np.exp(-dE_1*beta_m) if dE_1 > 0 else 0)
    eq_2 = ((1 - x**2) * (1 + x)) * (1 - np.exp(-dE_2*beta_h) if dE_2 > 0 else 0)
    eq_3 = ((1 + x)**2 * (1 - x)) * (np.exp(-dE_3*beta_m) if dE_3 > 0 else 1)
    eq_4 = ((1 - x**2) * (1 + x)) * (1 - np.exp(-dE_4*beta_c) if dE_4 > 0 else 0)
    eq_5 = ((1 - x**2) * (1 - x)) * (np.exp(-dE_5*beta_c) if dE_5 > 0 else 1)
    eq_6 = ((1 - x**2) * (1 - x)) * (np.exp(-dE_6*beta_h) if dE_6 > 0 else 1)
    eq_7 = ((1 - x**2) * (1 + x)) * (1 - np.exp(-dE_7*beta_m) if dE_7 > 0 else 0)
    eq_8 = ((1 - x)**3) * (np.exp(-dE_8*beta_m) if dE_8 > 0 else 1)

    denom = eq_1 + eq_2 + eq_3 + eq_4 + eq_5 + eq_6 + eq_7 + eq_8

    x0=np.array([(eq_1+eq_2)/2, (eq_3+eq_5)/2, (eq_4+eq_7)/2, (eq_1+eq_2)/2, (eq_6+eq_8)/2, (eq_4+eq_7)/2, (eq_3+eq_5)/2, (eq_6+eq_8)/2])
    x2=np.array([(eq_1+eq_3)/2, (eq_2+eq_5)/2, (eq_1+eq_3)/2, (eq_4+eq_6)/2, (eq_2+eq_5)/2, (eq_4+eq_6)/2, (eq_7+eq_8)/2, (eq_7+eq_8)/2])

    x_og=(1/8)*denom
    # Polynomial definitions
    f1, g1, h1, w1 = 2 * (1 - p1) * (1 + x0[0])*(1+x)*(1+x2[0]), p1 * (1 + x0[0])*(1+x)*(1+x2[0]), p1 * (1 + x0[0])*(1+x)*(1+x2[0]), 0
    f2, g2, h2, w2 = (1 - p2) * (1 - x0[1])*(1+x) * (1 + x2[1]), (1 - x0[1])*(1+x) * (1 + x2[1]), 0, p2 * (1 - x0[1])*(1+x) * (1 + x2[1])
    f3, g3, h3, w3 = 2 * p3 * (1 +x0[2])*(1-x) * (1 + x2[2]), (1 - p3) * (1 +x0[2])*(1-x) * (1 + x2[2]), (1 - p3) * (1 +x0[2])*(1-x) * (1 + x2[2]), 0
    f4, g4, h4, w4 = (1 - p4) * (1 +x0[3])*(1+x) * (1 -x2[3]), 0, (1 +x0[3])*(1+x) * (1 -x2[3]), p4 * (1 +x0[3])*(1+x) * (1 -x2[3])
    f5, g5, h5, w5 = p5 * (1 - x0[4])*(1-x) * (1+x2[4]), (1 - x0[4])*(1-x) * (1+x2[4]), 0, (1 - p5) *(1 - x0[4])*(1-x) * (1+x2[4])
    f6, g6, h6, w6 = p6 * (1 +x0[5])*(1-x2[5]) * (1 - x), 0, (1 +x0[5])*(1-x2[5]) * (1 - x), (1 - p6) *(1 +x0[5])*(1-x2[5]) *(1-x)
    f7, g7, h7, w7 = 0, (1 - p7) * (1-x0[6]) * (1+x)*(1-x2[6]), (1 - p7) *(1-x0[6])*(1+x)*(1-x2[6]), 2 * p7 *(1-x0[6]) * (1+x)*(1-x2[6])
    f8, g8, h8, w8 = 0, p8 * (1 - x0[7])*(1-x)*(1-x2[7]), p8 *(1 - x0[7])*(1-x)*(1-x2[7]), 2 * (1 - p8) *(1 - x0[7])*(1-x)*(1-x2[7])

    # Polynomial components for denominators
    f=np.array([f1,f2,f3,f4,f5,f6,f7,f8])
    g=np.array([g1,g2,g3,g4,g5,g6,g7,g8])
    h0=np.array([h1,h2,h3,h4,h5,h6,h7,h8])
    w=np.array([w1,w2,w3,w4,w5,w6,w7,w8])

    total_residuals = (1/8)*(f + g- h0 -w)-x

    target_expr = sp.Add(*total_residuals.tolist())
    #print(f"Target Expression: {target_expr}")

    return sp.lambdify(x, target_expr, modules=['numpy'])

# --- Numerical Root Finder ---
def solve_mfc_roots(T_left, T_right, h, mu, J, domain=(-0.99, 0.99), n_samples=200):
    p_numeric = build_mfc_symbolic_func(T_left, T_right, h, mu, J)
    grid = np.linspace(domain[0], domain[1], n_samples)
    
    values = np.zeros_like(grid)
    for idx, v in enumerate(grid):
        try:
            val = p_numeric(v)
            values[idx] = float(np.real(val)) if np.isfinite(val) else np.nan
        except (ZeroDivisionError, OverflowError):
            values[idx] = np.nan

    roots = []
    for i in range(len(grid) - 1):
        if np.isnan(values[i]) or np.isnan(values[i + 1]):
            continue
        if np.sign(values[i]) != np.sign(values[i + 1]):
            try:
                res = root_scalar(p_numeric, bracket=[grid[i], grid[i + 1]], method='brentq')
                if res.converged:
                    r = float(res.root)
                    if not any(np.isclose(r, existing, atol=1e-3) for existing in roots):
                        roots.append(np.abs(r))
            except ValueError:
                pass
    #print(f"Roots found for T_left={T_left}, T_right={T_right}, h={h}: {roots}")
    return sorted(roots)

def solve_mfc_correlations(T_left, T_right, h, mu, J):
    mfc_func = build_mfc_symbolic_func(T_left, T_right, h, mu, J)
    
    grid = np.linspace(-0.99, 0.99, 100)
    vals = [mfc_func(v) for v in grid]
    x_star = 0.0
    for k in range(len(grid)-1):
        if np.sign(vals[k]) != np.sign(vals[k+1]):
            res = root_scalar(mfc_func, bracket=[grid[k], grid[k+1]], method='brentq')
            if res.converged:
                x_star = res.root
                break
    beta_c = 1.0 / T_right
    beta_h = 1.0 / T_left
    beta_m = (beta_c + beta_h) / 2.0

    dE = np.array([
        2.0 * (J * 2 + mu * h),
        2.0 * (mu * h),
        -2.0 * (J * 2 + mu * h),
        2.0 * (mu * h),
        -2.0 * (mu * h),
        -2.0 * (mu * h),
        2.0 * (-2 * J + mu * h),
        -2.0 * (-2 * J + mu * h)
    ])
    
    # Calculate transition rates/probabilities
    eq = np.zeros(8)
    eq[0] = ((1 + x_star)**3) * (1 - np.exp(-dE[0]*beta_m) if dE[0] > 0 else 0)
    eq[1] = ((1 - x_star**2) * (1 + x_star)) * (1 - np.exp(-dE[1]*beta_h) if dE[1] > 0 else 0)
    eq[2] = ((1 + x_star)**2 * (1 - x_star)) * (np.exp(-dE[2]*beta_m) if dE[2] > 0 else 1)
    eq[3] = ((1 - x_star**2) * (1 + x_star)) * (1 - np.exp(-dE[3]*beta_c) if dE[3] > 0 else 0)
    eq[4] = ((1 - x_star**2) * (1 - x_star)) * (np.exp(-dE[4]*beta_c) if dE[4] > 0 else 1)
    eq[5] = ((1 - x_star**2) * (1 - x_star)) * (np.exp(-dE[5]*beta_h) if dE[5] > 0 else 1)
    eq[6] = ((1 - x_star**2) * (1 + x_star)) * (1 - np.exp(-dE[6]*beta_m) if dE[6] > 0 else 0)
    eq[7] = ((1 - x_star)**3) * (np.exp(-dE[7]*beta_m) if dE[7] > 0 else 1)

    # Compute correlation arrays x0 and x2 at steady state
    x0_vals = np.array([
        (eq[0]+eq[1])/2, (eq[2]+eq[4])/2, (eq[3]+eq[6])/2, (eq[0]+eq[1])/2, 
        (eq[5]+eq[7])/2, (eq[3]+eq[6])/2, (eq[2]+eq[4])/2, (eq[5]+eq[7])/2
    ])
    
    x2_vals = np.array([
        (eq[0]+eq[2])/2, (eq[1]+eq[4])/2, (eq[0]+eq[2])/2, (eq[3]+eq[5])/2, 
        (eq[1]+eq[4])/2, (eq[3]+eq[5])/2, (eq[6]+eq[7])/2, (eq[6]+eq[7])/2
    ])

    results = {
        "x_magnetization": x_star,
        "x0_correlations": x0_vals,
        "x2_correlations": x2_vals,
        "mean_x0": float(np.mean(x0_vals)),
        "mean_x2": float(np.mean(x2_vals))
    }
    mag = results['x_magnetization']
    nearest_neighbor_corr = results['mean_x0']

#Connected correlation (fluctuations)
    connected_corr = nearest_neighbor_corr - mag**2

#Next-nearest neighbor / diagonal correlations (if using 3-spin or 4-spin clusters)
# Derived from the product of state probabilities across channels
    diag_corr = np.mean(results['x0_correlations'] * results['x2_correlations'])

    print(f"Magnetization <σ>:               {mag:.6f}")
    print(f"Nearest-Neighbor <σ_i σ_j>:      {nearest_neighbor_corr:.6f}")
    print(f"Connected Correlation <σ_i σ_j>c: {connected_corr:.6f}")
    print(f"Cluster Cross-Correlation:       {diag_corr:.6f}")

    return nearest_neighbor_corr

# --- Running Simulation ---
m_sim, sim_h, sim_t = [], [], []

for Tl in T_right:
    for h in H:
        m_hist= chain(NSpins, total_sweeps, burn_in, T_left, Tl, J, mu, h)
        m_sim.append(np.mean(m_hist[burn_in:]))
        sim_h.append(h)
        sim_t.append(Tl)


# --- Calculating Surface Roots ---
m_midpoint = 0.0

for i, T_l in enumerate(T_right):
    for j, h in enumerate(H):
        #roots = find_intersections(T_left, T_l, h, mu, J)
        roots = solve_mfc_roots(T_left, T_l, h, mu, J)
        clust_corr= solve_mfc_correlations(T_left, T_l, h, mu, J)
        Z_x0[i, j] = clust_corr

        if len(roots) == 1:
            if roots[0] >= m_midpoint:
                Z_upper[i, j] = roots[0]
            else:
                Z_lower[i, j] = roots[0]
        elif len(roots) == 2:
            Z_lower[i, j] = roots[0]
            Z_upper[i, j] = roots[1]
        elif len(roots) >= 3:
            Z_lower[i, j] = roots[0]
            Z_middle[i, j] = roots[1]
            Z_upper[i, j] = roots[-1]

m_sim_2d = np.array(m_sim).reshape(len(T_right), len(H))
abs_diff = np.abs(Z_upper - m_sim_2d)
mean_diff = np.nanmean(abs_diff)
max_diff = np.nanmax(abs_diff)
print(f"Mean difference: {mean_diff}, Max difference: {max_diff}")

# --- Plotting 3D Surface ---
fig = plt.figure(figsize=(11, 8), dpi=120)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(H_grid, T_grid, Z_upper, color='royalblue', alpha=0.35,
                edgecolor='navy', linewidth=0.3, rstride=1, cstride=1)
ax.plot_surface(H_grid, T_grid, Z_lower, color='lightskyblue', alpha=0.35,
                edgecolor='dodgerblue', linewidth=0.3, rstride=1, cstride=1)

step = 2
ax.scatter(sim_h[::step], sim_t[::step], m_sim[::step],
           color='crimson', marker='^', s=30, edgecolors='k',
           linewidth=0.5, depthshade=True, label='Monte Carlo Simulation', zorder=5)


ax.view_init(elev=28, azim=135)
ax.set_xlabel('Magnetic Field ($H$)', fontsize=11, labelpad=10)
ax.set_ylabel('$T_{right}$ (Cold Bath)', fontsize=11, labelpad=10)
ax.set_zlabel('Magnetization ($m$)', fontsize=11, labelpad=8)
ax.set_title(f'Phase Surface with Corrections: MFC Theory vs Simulation ($T_{{left}}={T_left}$)', fontsize=12, pad=15)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig("3d_bifurcation_surface_corrections2.png", dpi=300)
plt.show()
plt.close(fig)


fig_corr = plt.figure(figsize=(10, 7), dpi=150)
ax_corr = fig_corr.add_subplot(111, projection='3d')

surf = ax_corr.plot_surface(
    H_grid, T_grid, Z_x0, 
    cmap='viridis', 
    alpha=0.85, 
    edgecolor='k', 
    linewidth=0.2
)

# Formatting
ax_corr.view_init(elev=28, azim=135)
ax_corr.set_xlabel('Magnetic Field ($H$)', fontsize=11, labelpad=10)
ax_corr.set_ylabel('$T_{right}$ (Cold Bath)', fontsize=11, labelpad=10)
ax_corr.set_zlabel(r'Pair Correlation $\langle \sigma_i \sigma_j \rangle$', fontsize=11, labelpad=8)
ax_corr.set_title(f'Nearest-Neighbor Correlation Surface ($T_{{left}}={T_left}$)', fontsize=12, pad=15)

# Styling
fig_corr.colorbar(surf, ax=ax_corr, shrink=0.5, aspect=10, pad=0.1, label=r'$\langle \sigma_i \sigma_j \rangle$')
ax_corr.xaxis.pane.fill = False
ax_corr.yaxis.pane.fill = False
ax_corr.zaxis.pane.fill = False
ax_corr.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig("3d_pair_correlation_surface.png", dpi=300)
plt.show()
plt.close(fig_corr)