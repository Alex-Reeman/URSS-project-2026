import numpy as np
from sympy import Symbol, expand, solve,Poly
import sympy as sp

def v_prime():
    J, mu, H = 0.25, 1.0,0.08
    T_left=3.0
    T_right=1.0
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
    
    return np.array([expand(res[0]), expand(res[1])])
x=Symbol('x')
p=v_prime()
p_tot=0.5*(p[0]-p[1])
roots=solve(p_tot,x)
formatted_roots = [f"{float(sp.re(r)):.3f}" for r in roots]
print(f"All roots: {formatted_roots}")

#print(v_prime())