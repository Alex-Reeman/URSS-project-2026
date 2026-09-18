import numpy as np
from sympy import Symbol, expand, solve,Poly
import sympy as sp
from scipy.optimize import root_scalar

J, mu, H = 0.25, 1.0,0.1
T_left=5.0
T_right=1.0

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
    f_p=((1+x)*(1+x2)+(1+x)*(1+x0))
    g_p=((1-x)*(1+x2)+(1+x)*(1-x0))
    h_p=((1+x)*(1-x2)+(1-x)*(1+x0))
    w_p=((1-x)*(1-x2)+(1-x)*(1-x0))

    total_residuals = (1/8)*(f + g- h0 -w)-x

    target_expr = sp.Add(*total_residuals.tolist())

    return sp.lambdify(x, target_expr, modules=['numpy'])

def solve_mfc_roots(T_left, T_right, h, mu, J, domain=(-1, 1), n_samples=200):
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
                        roots.append(r)
            except ValueError:
                pass

    return sorted(roots)


roots = solve_mfc_roots(T_left, T_right, H, mu, J)
formatted_roots = [f"{r:.4f}" for r in roots]
print(f"All physical roots: {formatted_roots}")

import numpy as np
import sympy as sp
from scipy.optimize import root_scalar

def solve_mfc_correlations(T_left, T_right, h, mu, J):

    mfc_func = build_mfc_symbolic_func(T_left, T_right, h, mu, J)

    sol = root_scalar(mfc_func, bracket=[-0.9999, 0.9999], method='brentq')
    if not sol.converged:
        raise ValueError("Root finding failed to converge.")
    
    x_star = sol.root
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
        "mean_x0": np.mean(x0_vals),
        "mean_x2": np.mean(x2_vals)
    }
    mag = results['x_magnetization']
    nearest_neighbor_corr = results['mean_x0']

# Connected correlation (fluctuations)
    connected_corr = nearest_neighbor_corr - mag**2

# Next-nearest neighbor / diagonal correlations (if using 3-spin or 4-spin clusters)
# Derived from the product of state probabilities across channels
    diag_corr = np.mean(results['x0_correlations'] * results['x2_correlations'])

    print(f"Magnetization <σ>:               {mag:.6f}")
    print(f"Nearest-Neighbor <σ_i σ_j>:      {nearest_neighbor_corr:.6f}")
    print(f"Connected Correlation <σ_i σ_j>c: {connected_corr:.6f}")
    print(f"Cluster Cross-Correlation:       {diag_corr:.6f}")
    return results


print(solve_mfc_correlations(T_left, T_right, H, mu, J))

clust_corr = solve_mfc_correlations(T_left, T_right, H, mu, J)
        
m = clust_corr['x_magnetization']
corr_nn = clust_corr['mean_x0']
        

E_avg = - 2*J * corr_nn - (mu * H) * m
        
print(E_avg)