import numpy as np
import sympy as sp
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

J, mu = 0.25, 1.0
H = np.linspace(0.01, 0.3, 20)
T_left = 5.0
T_right = np.linspace(0.1, 4.9, 20)

NSpins = 50
total_sweeps = 5000
burn_in = 1000

H_grid, T_grid = np.meshgrid(H, T_right)
E_mfc_matrix = np.zeros_like(H_grid)

def build_mfc_symbolic_func(T_left, T_right_val, h, mu, J):
    x = sp.Symbol('x')
    
    beta_c = 1.0 / T_right_val
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

    x0 = np.array([(eq_1+eq_2)/2, (eq_3+eq_5)/2, (eq_4+eq_7)/2, (eq_1+eq_2)/2, 
                   (eq_5+eq_7)/2, (eq_4+eq_7)/2, (eq_3+eq_5)/2, (eq_5+eq_7)/2])
    x2 = np.array([(eq_1+eq_3)/2, (eq_2+eq_5)/2, (eq_1+eq_3)/2, (eq_4+eq_6)/2, 
                   (eq_2+eq_5)/2, (eq_4+eq_6)/2, (eq_7+eq_8)/2, (eq_7+eq_8)/2])

    f1, g1, h1, w1 = 2 * (1 - p1) * (1 + x0[0])*(1+x)*(1+x2[0]), p1 * (1 + x0[0])*(1+x)*(1+x2[0]), p1 * (1 + x0[0])*(1+x)*(1+x2[0]), 0
    f2, g2, h2, w2 = (1 - p2) * (1 - x0[1])*(1+x) * (1 + x2[1]), (1 - x0[1])*(1+x) * (1 + x2[1]), 0, p2 * (1 - x0[1])*(1+x) * (1 + x2[1])
    f3, g3, h3, w3 = 2 * p3 * (1 +x0[2])*(1-x) * (1 + x2[2]), (1 - p3) * (1 +x0[2])*(1-x) * (1 + x2[2]), (1 - p3) * (1 +x0[2])*(1-x) * (1 + x2[2]), 0
    f4, g4, h4, w4 = (1 - p4) * (1 +x0[3])*(1+x) * (1 -x2[3]), 0, (1 +x0[3])*(1+x) * (1 -x2[3]), p4 * (1 +x0[3])*(1+x) * (1 -x2[3])
    f5, g5, h5, w5 = p5 * (1 - x0[4])*(1-x) * (1+x2[4]), (1 - x0[4])*(1-x) * (1+x2[4]), 0, (1 - p5) *(1 - x0[4])*(1-x) * (1+x2[4])
    f6, g6, h6, w6 = p6 * (1 +x0[5])*(1-x2[5]) * (1 - x), 0, (1 +x0[5])*(1-x2[5]) * (1 - x), (1 - p6) *(1 +x0[5])*(1-x2[5]) *(1-x)
    f7, g7, h7, w7 = 0, (1 - p7) * (1-x0[6]) * (1+x)*(1-x2[6]), (1 - p7) *(1-x0[6])*(1+x)*(1-x2[6]), 2 * p7 *(1-x0[6]) * (1+x)*(1-x2[6])
    f8, g8, h8, w8 = 0, p8 * (1 - x0[7])*(1-x)*(1-x2[7]), p8 *(1 - x0[7])*(1-x)*(1-x2[7]), 2 * (1 - p8) *(1 - x0[7])*(1-x)*(1-x2[7])

    f = np.array([f1,f2,f3,f4,f5,f6,f7,f8])
    g = np.array([g1,g2,g3,g4,g5,g6,g7,g8])
    h0 = np.array([h1,h2,h3,h4,h5,h6,h7,h8])
    w = np.array([w1,w2,w3,w4,w5,w6,w7,w8])

    total_residuals = (1/8)*(f + g - h0 - w) - x
    target_expr = sp.Add(*total_residuals.tolist())

    return sp.lambdify(x, target_expr, modules=['numpy'])

def solve_mfc_correlations(T_left, T_right_val, h, mu, J):
    mfc_func = build_mfc_symbolic_func(T_left, T_right_val, h, mu, J)
    
    grid = np.linspace(-0.99, 0.99, 100)
    vals = [mfc_func(v) for v in grid]
    x_star = 0.0
    for k in range(len(grid)-1):
        if np.sign(vals[k]) != np.sign(vals[k+1]):
            res = root_scalar(mfc_func, bracket=[grid[k], grid[k+1]], method='brentq')
            if res.converged:
                x_star = res.root
                break

    beta_c = 1.0 / T_right_val
    beta_h = 1.0 / T_left
    beta_m = (beta_c + beta_h) / 2.0

    dE = np.array([
        2.0 * (J * 2 + mu * h), 2.0 * (mu * h), -2.0 * (J * 2 + mu * h), 2.0 * (mu * h),
        -2.0 * (mu * h), -2.0 * (mu * h), 2.0 * (-2 * J + mu * h), -2.0 * (-2 * J + mu * h)
    ])
    
    eq = np.zeros(8)
    eq[0] = ((1 + x_star)**3) * (1 - np.exp(-dE[0]*beta_m) if dE[0] > 0 else 0)
    eq[1] = ((1 - x_star**2) * (1 + x_star)) * (1 - np.exp(-dE[1]*beta_h) if dE[1] > 0 else 0)
    eq[2] = ((1 + x_star)**2 * (1 - x_star)) * (np.exp(-dE[2]*beta_m) if dE[2] > 0 else 1)
    eq[3] = ((1 - x_star**2) * (1 + x_star)) * (1 - np.exp(-dE[3]*beta_c) if dE[3] > 0 else 0)
    eq[4] = ((1 - x_star**2) * (1 - x_star)) * (np.exp(-dE[4]*beta_c) if dE[4] > 0 else 1)
    eq[5] = ((1 - x_star**2) * (1 - x_star)) * (np.exp(-dE[5]*beta_h) if dE[5] > 0 else 1)
    eq[6] = ((1 - x_star**2) * (1 + x_star)) * (1 - np.exp(-dE[6]*beta_m) if dE[6] > 0 else 0)
    eq[7] = ((1 - x_star)**3) * (np.exp(-dE[7]*beta_m) if dE[7] > 0 else 1)

    x0_vals = np.array([
        (eq[0]+eq[1])/2, (eq[2]+eq[4])/2, (eq[3]+eq[6])/2, (eq[0]+eq[1])/2, 
        (eq[5]+eq[7])/2, (eq[3]+eq[6])/2, (eq[2]+eq[4])/2, (eq[5]+eq[7])/2
    ])
    print(f"Computed x_star: {x_star}, mean_x0: {np.mean(x0_vals)}")
    return {'x_magnetization': x_star, 'mean_x0': np.mean(x0_vals)}

def chain(NSpins, total_sweeps, burn_in, T_left, T_right_val, J, mu, H_val):
    spin = np.ones(NSpins, dtype=np.int8)
    current_M = float(np.sum(spin))

    beta_c = 1.0 / T_right_val
    beta_h = 1.0 / T_left

    E_hist = np.empty(total_sweeps, dtype=np.float64)

    for sweep in range(total_sweeps):
        for _ in range(NSpins):            
            r = np.random.randint(0, NSpins)
            nb = spin[(r-1)%NSpins] + spin[(r+1)%NSpins]
            dE = 2.0 * spin[r] * (J * nb + mu * H_val)

            left_aligned = (spin[(r-1)%NSpins] == spin[r])
            right_aligned = (spin[(r+1)%NSpins] == spin[r])
            if left_aligned and not right_aligned:
                beta = beta_c
            elif right_aligned and not left_aligned:
                beta = beta_h
            else:
                beta = 0.5 * (beta_h + beta_c)

            if dE <= 0.0 or np.random.random() < np.exp(-dE * beta):
                current_M -= 2 * spin[r]
                spin[r] = -spin[r]

        total_E = -J * np.sum(spin * np.roll(spin, -1)) - mu * H_val * current_M
        E_hist[sweep] = total_E / NSpins

    return np.mean(E_hist[burn_in:])

for i, Tl in enumerate(T_right):
    for j, h in enumerate(H):
        clust_corr = solve_mfc_correlations(T_left, Tl, h, mu, J)
        m = clust_corr['x_magnetization']
        corr_nn = clust_corr['mean_x0']
        E_mfc_matrix[i, j] = - J * corr_nn - (mu * h) * m

E_sim, sim_h, sim_t = [], [], []
for Tl in T_right:
    for h in H:
        e_val = chain(NSpins, total_sweeps, burn_in, T_left, Tl, J, mu, h)
        E_sim.append(e_val)
        sim_h.append(h)
        sim_t.append(Tl)

fig = plt.figure(figsize=(11, 8), dpi=120)
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(H_grid, T_grid, E_mfc_matrix, color='royalblue', alpha=0.35,
                edgecolor='navy', linewidth=0.3, rstride=1, cstride=1)

step = 2
ax.scatter(sim_h[::step], sim_t[::step], E_sim[::step],
           color='crimson', marker='^', s=30, edgecolors='k',
           linewidth=0.5, depthshade=True, label='Monte Carlo Simulation', zorder=5)

ax.view_init(elev=28, azim=135)
ax.set_xlabel('Magnetic Field ($H$)', fontsize=11, labelpad=10)
ax.set_ylabel('$T_{right}$ (Cold Bath)', fontsize=11, labelpad=10)
ax.set_zlabel('Average Energy per spin', fontsize=11, labelpad=8)
ax.set_title(f'Energy per spin: MFC Theory vs Simulation ($T_{{left}}={T_left}$)', fontsize=12, pad=15)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig("3d_energy.png", dpi=300)
plt.show()