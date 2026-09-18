import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from numba import njit
import sympy as sp
from sympy import Symbol

# --- Parameters ---
NSpins = 8
J, mu, H = 0.25, 1.0, np.linspace(0.01, 0.3, 20)
T_right = np.linspace(0.1, 4.9, 20)
T_left = 5.0

H_grid, T_grid = np.meshgrid(H, T_right)
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

    # Polynomial definitions
    f1, g1, h1, w1 = 2 * (1 - p1) * (1 + x)**3, p1 * (1 + x)**3, p1 * (1 + x)**3, 0
    f2, g2, h2, w2 = (1 - p2) * (1 - x**2) * (1 + x), (1 - x**2) * (1 + x), 0, p2 * (1 - x**2) * (1 + x)
    f3, g3, h3, w3 = 2 * p3 * (1 - x**2) * (1 + x), (1 - p3) * (1 - x**2) * (1 + x), (1 - p3) * (1 - x**2) * (1 + x), 0
    f4, g4, h4, w4 = (1 - p4) * (1 - x**2) * (1 + x), 0, (1 - x**2) * (1 + x), p4 * (1 - x**2) * (1 + x)
    f5, g5, h5, w5 = p5 * (1 - x**2) * (1 - x), (1 - x**2) * (1 - x), 0, (1 - p5) * (1 - x**2) * (1 - x)
    f6, g6, h6, w6 = p6 * (1 - x**2) * (1 - x), 0, (1 - x**2) * (1 - x), (1 - p6) * (1 - x**2) * (1 - x)
    f7, g7, h7, w7 = 0, (1 - p7) * (1 - x**2) * (1 - x), (1 - p7) * (1 - x**2) * (1 - x), 2 * p7 * (1 - x**2) * (1 - x)
    f8, g8, h8, w8 = 0, p8 * (1 - x)**3, p8 * (1 - x)**3, 2 * (1 - p8) * (1 - x)**3

    # Polynomial components for denominators
    eq_1 = ((1 + x)**3) * (1 - np.exp(-dE_1*beta_m) if dE_1 > 0 else 0)
    eq_2 = ((1 - x**2) * (1 + x)) * (1 - np.exp(-dE_2*beta_h) if dE_2 > 0 else 0)
    eq_3 = ((1 + x)**2 * (1 - x)) * (np.exp(-dE_3*beta_m) if dE_3 > 0 else 1)
    eq_4 = ((1 - x**2) * (1 + x)) * (1 - np.exp(-dE_4*beta_c) if dE_4 > 0 else 0)
    eq_5 = ((1 - x**2) * (1 - x)) * (np.exp(-dE_5*beta_c) if dE_5 > 0 else 1)
    eq_6 = ((1 - x**2) * (1 - x)) * (np.exp(-dE_6*beta_h) if dE_6 > 0 else 1)
    eq_7 = ((1 - x**2) * (1 + x)) * (1 - np.exp(-dE_7*beta_m) if dE_7 > 0 else 0)
    eq_8 = ((1 - x)**3) * (np.exp(-dE_8*beta_m) if dE_8 > 0 else 1)

    eqs = [
        np.exp(-beta_m)*((f1 - g1 - h1 + w1)+4*x), np.exp(-beta_h)*((f2 - g2 - h2 + w2)+4*x),
        np.exp(-beta_m)*((f3 - g3 - h3 + w3)+4*x), np.exp(-beta_c)*((f4 - g4 - h4 + w4)+4*x),
        np.exp(-beta_c)*((f5 - g5 - h5 + w5)+4*x), np.exp(-beta_h)*((f6 - g6 - h6 + w6)+4*x),
        np.exp(-beta_m)*((f7 - g7 - h7 + w7)+4*x), np.exp(-beta_m)*((f8 - g8 - h8 + w8)+4*x)
    ]

    denom = eq_1 + eq_2 + eq_3 + eq_4 + eq_5 + eq_6 + eq_7 + eq_8
    target_expr = 4 * sum(eqs) / denom

    return sp.lambdify(x, target_expr, modules=['numpy'])

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

    return sorted(roots)


# --- Running Simulation ---
m_sim, sim_h, sim_t = [], [], []

for Tl in T_right:
    for h in H:
        m_hist = chain(NSpins, total_sweeps, burn_in, T_left, Tl, J, mu, h)
        m_sim.append(np.mean(m_hist[burn_in:]))
        sim_h.append(h)
        sim_t.append(Tl)

m_midpoint = 0.0

for i, T_l in enumerate(T_right):
    for j, h in enumerate(H):
        roots = solve_mfc_roots(T_left, T_l, h, mu, J)

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
plt.savefig("3d_bifurcation_surface_corrections.png", dpi=300)
plt.show()