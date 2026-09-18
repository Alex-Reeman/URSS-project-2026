import numpy as np
from sympy import Symbol, expand, solve,Poly
import sympy as sp
from numba import njit
import matplotlib.pyplot as plt

NSpins = 8
J, mu, H = 0.25, 1.0, np.linspace(0.01,0.3,20) 

T_right=np.linspace(0.1,4.9,20)
T_left=5.0

H_grid, T_grid = np.meshgrid(H, T_right)
Z_upper = np.full_like(H_grid, np.nan)
Z_middle = np.full_like(H_grid, np.nan)
Z_lower = np.full_like(H_grid, np.nan)

burn_in=10000
sweeps=100000
total_sweeps=burn_in+sweeps

@njit

def chain(NSpins, total_sweeps, burn_in, T_left, T_right, J, mu, H):
    
    spin = np.ones(NSpins, dtype=np.int8)
    current_M = float(np.sum(spin))

    beta_c=1.0/T_right
    beta_h=1.0/T_left

    m_hist=np.empty(total_sweeps,dtype=np.float64)

    for sweep in range(total_sweeps):
        for _ in range(NSpins):            
            r=np.random.randint(0,NSpins)
            #closed BCs
            nb=spin[(r-1)%NSpins]+spin[(r+1)%NSpins]
            #need to take coupling and neighbour alignment into account!
            dE=2.0*spin[r] *(J*nb+mu*H)

            #if spin interacting with spin at left end, use hot bath temperatur
            left_aligned=(spin[(r-1)%NSpins]==spin[r])
            right_aligned=(spin[(r+1)%NSpins]==spin[r])
            if left_aligned and not right_aligned:
                beta=beta_c
            elif right_aligned and not left_aligned:
                beta=beta_h
            else:
                beta=0.5*(beta_h+beta_c)

            #Metropolis
            if dE<=0.0 or np.random.random()<np.exp(-dE*beta):
                current_M-=2*spin[r]
                spin[r]=-spin[r]

        m_hist[sweep]=current_M/NSpins

    return m_hist


def v_prime(T_left,T_right,H,mu,J):
    beta_c=1/T_right
    beta_h=1/T_left
    beta_m=(beta_c+beta_h)/2
    
    dE_1=2.0*(J*2+mu*H)
    dE_2=2.0*(mu*H)
    dE_3=-2.0*(J*2+mu*H)
    dE_4=2.0*(mu*H)
    dE_5=-2.0*(mu*H)
    dE_6=-2.0*(mu*H)
    dE_7=2.0*(-2*J+mu*H)
    dE_8=-2.0*(-2*J+mu*H)
    x = Symbol('x')

    if dE_1>0:
        eq_1 = ((1 + x)**3) * np.array([1 - np.exp(-dE_1*beta_m), np.exp(-dE_1*beta_m)])
    else:
        eq_1= ((1 + x)**3) * np.array([0, 1])
    if dE_2>0:
        eq_2 = ((1 - x**2) * (1 + x)) * np.array([1 - np.exp(-dE_2*beta_h), np.exp(-dE_2*beta_h)])
    else:
        eq_2=((1 - x**2) * (1 + x)) * np.array([0,1])
    if dE_3>0:
        eq_3 = ((1 + x)**2 * (1 - x)) * np.array([np.exp(-dE_3*beta_m),1 - np.exp(-dE_3*beta_m)])
    else:
        eq_3=((1 + x)**2 * (1 - x)) * np.array([1,0])
    if dE_4>0:
        eq_4 = ((1 - x**2) * (1 + x)) * np.array([1 - np.exp(-dE_4*beta_c), np.exp(-dE_4*beta_c)])
    else:
        eq_4 = ((1 - x**2) * (1 + x)) * np.array([0, 1])
    if dE_5>0:
        eq_5 = ((1 - x**2) * (1 - x)) * np.array([np.exp(-dE_5*beta_c), 1 - np.exp(-dE_5*beta_c)])
    else:
        eq_5 = ((1 - x**2) * (1 - x)) * np.array([1,0])
    if dE_6>0:
        eq_6 = ((1 - x**2) * (1 - x)) * np.array([np.exp(-dE_6*beta_h),1 - np.exp(-dE_6*beta_h)])
    else:
        eq_6 = ((1 - x**2) * (1 - x)) * np.array([1,0])
    if dE_7>0:
        eq_7 = ((1 - x**2) * (1 + x)) * np.array([1 - np.exp(-dE_7*beta_m), np.exp(-dE_7*beta_m)])
    else:
        eq_7 = ((1 - x**2) * (1 + x)) * np.array([0, 1])
    if dE_8>0:
        eq_8 = ((1 - x)**3) * np.array([np.exp(-dE_8*beta_m),1 - np.exp(-dE_8*beta_m)])
    else:
        eq_8 = ((1 - x)**3) * np.array([1,0])
    
    res = 0.5*(1/8) * (eq_1 + eq_2 + eq_3 + eq_4 + eq_5 + eq_6 + eq_7 + eq_8)
    
    # Expand or simplify each element of the array
    p=np.array([expand(res[0]), expand(res[1])])
    p_tot=0.5*(p[0]-p[1])
    roots=solve(p_tot,x)
    real_roots = []
    for r in roots:
        val = complex(r.evalf())
        # Check if imaginary part is negligible (numerical noise)
        if abs(val.imag) < 1e-6 and -1.0 <= val.real <= 1.0:
            real_roots.append(val.real)
    
    return real_roots

m_sim=[]
h_points=[]
m_mfc=[]
T_l_points=[]
sim_h=[]
sim_t=[]

for Tl in T_right:
    for h in H:
        sweeps_arr=np.arange(total_sweeps)
        m_hist=chain(NSpins,total_sweeps,burn_in,T_left,Tl,J,mu,h)
        m_running_mean=np.cumsum(m_hist)/(sweeps_arr+1)
        m_sim.append(m_running_mean[-1])
        sim_h.append(h)
        sim_t.append(Tl)
        print(v_prime(T_left, Tl, h, mu, J))

m_midpoint = 0.5 

for i, T_l in enumerate(T_right):
    for j, h in enumerate(H):
        roots = sorted(v_prime(T_left, T_l, h, mu, J))
        
        if len(roots) == 1:
            # Assign single root to its corresponding physical sheet
            if roots[0] >= m_midpoint:
                Z_upper[i, j] = roots[0]
            else:
                Z_lower[i, j] = roots[0]
                
        elif len(roots) == 2:
            # Two roots: assign smaller to lower branch, larger to upper branch
            Z_lower[i, j] = roots[0]
            Z_upper[i, j] = roots[1]
            
        elif len(roots) >= 3:
            # Three roots: lower, middle (unstable), upper
            Z_lower[i, j] = roots[0]
            Z_middle[i, j] = roots[1]
            Z_upper[i, j] = roots[-1]

sheets = np.array([Z_upper, Z_lower, Z_middle])
m_sim_2d = np.array(m_sim).reshape(len(T_right), len(H))
branch_diffs = np.abs(sheets - m_sim_2d)
min_abs_diff = np.nanmin(branch_diffs, axis=0)
mean_diff = np.nanmean(min_abs_diff)
max_diff = np.nanmax(min_abs_diff)

print(f"Mean difference: {mean_diff}, Max difference: {max_diff}")

fig = plt.figure(figsize=(11, 8),dpi=120)
ax = fig.add_subplot(111, projection='3d')

# Plot upper MFC Surface Sheet
ax.plot_surface(H_grid, T_grid, Z_upper, color='royalblue', alpha=0.35, 
                edgecolor='navy', linewidth=0.3, rstride=1, cstride=1)

# Plot lower MFC Surface Sheet (if present)
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
ax.set_title(f'Phase Surface: MFC Theory vs Simulation ($T_{{left}}={T_left}$)', fontsize=12, pad=15)

# Pane aesthetics
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig("3d_bifurcation_surface_2.png", dpi=300)
plt.show()