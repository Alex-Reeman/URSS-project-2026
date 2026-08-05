import numpy as np
import time
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
NSpins_vals = [8,16,32,64,128]
J, mu, H = 1, 1.0, 0
#T = 1.5
T_vals=[1*J,1.5*J,2*J,2.5*J,3*J,3.5*J,4*J,4.5*J,5*J]

burn_in=1000
sweeps=10000
total_sweeps=burn_in+sweeps

@njit
def chain(NSpins, total_sweeps, burn_in, T, J, mu, H):
    
    spin = np.ones(NSpins, dtype=np.int8)
    current_E = 0.0
    current_M = float(NSpins)
    beta = 1.0/T

    #change to N-1 spins
    for i in range(NSpins-1):
        nb=spin[(i+1)]
        current_E+=-J*spin[i]*nb #-mu*H*spin[i]
    for i in range(NSpins):
        current_E+=-mu*H*spin[i]

    num_steps=total_sweeps-burn_in
    
    m_hist=np.empty(total_sweeps,dtype=np.float64)
    E_hist=np.empty(num_steps,dtype=np.float64)
    step_idx=0

    for sweep in range(total_sweeps):
        for _ in range(NSpins):
            r=np.random.randint(0,NSpins)
            #open BCs
            nb=0.0
            if r>0:
                nb+=spin[r-1]
            if r<NSpins-1:
                nb+=spin[r+1]
            #need to take coupling and neighbour alignment into account!
            dE=2.0*spin[r] *(J*nb+mu*H)

            #Metropolis
            if dE<=0.0 or np.random.random()<np.exp(-dE*beta):
                current_E+=dE
                current_M-=2.0*spin[r]
                spin[r]=-spin[r]
        #tracking magnetisation        
        m_hist[sweep]=current_M/NSpins
        #sample energy
        if sweep>=burn_in:
            E_hist[step_idx]=current_E
            step_idx+=1

    return m_hist,E_hist

def free_energy_ons(NSpins,T,J,mu,H):
    beta=1.0/T
    h=mu*H

    #eigenvals of transfer matrix
    #https://arxiv.org/pdf/2208.04751 eq 8
    sqrt_term=np.sqrt(np.sinh(beta*h)**2+np.exp(-4*beta*J))
    lambda_plus=np.exp(beta*J)*(np.cosh(beta*h)+sqrt_term)
    lambda_minus=np.exp(beta*J)*(np.cosh(beta*h)-sqrt_term)
    
    log_Z=np.log((lambda_plus**NSpins)+lambda_minus**NSpins)

    F_total=-(1.0/beta)*log_Z
    f_per_spin=F_total/NSpins
    return f_per_spin,lambda_plus,lambda_minus

def exact_Cv(N,T,J):
    beta=1/T
    #return (beta*J)**2*(1.0/np.cosh(beta*J))**2
    x=beta*J
    ch=np.cosh(x)
    sh=np.sinh(x)
    ch_N=ch**N
    sh_N=sh**N
    Z=ch_N+sh_N

    E_mean=-N*J*(ch**(N-1)*sh+sh**(N-1)*ch)/Z
    d2_Z=N*(J**2)*((N-1)*(ch**(N-2)*sh**2+sh**(N-2)*ch**2)+ch_N+sh_N)/Z

    Var_E=d2_Z-(E_mean**2)
    Cv_per_spin=(beta**2*Var_E)/N
    return Cv_per_spin
    #https://drive.uqu.edu.sa/_/quc_physics/files/[Pathria_R_K_,_Beale_P_D_]_Statistical_mechanics.pdf chap 13.2 eq. 16
    #return ((N-1)/N)*(x)**2*(1.0/ch)**2

#https://dfm.io/posts/autocorr/ -- autocorrelation time estimation 
def next_pow_2(n):
    i=2 
    while i<n:
        i=i<<1
    return i

def autocorr_time(x, norm=True):
    x=np.atleast_1d(x)
    if len(x.shape)!=1:
        raise ValueError("x must be 1D array")
    n=next_pow_2(len(x))
    #fourier transform of f is c^_f(t)
    f=np.fft.fft(x-np.mean(x),n=2*n)
    acf=np.fft.ifft(f*np.conjugate(f))[:len(x)].real
    acf/=4*n
    if norm and acf[0]!=0:
        acf/=acf[0]
    return acf

def auto_window(taus,c=5.0):
    #sokal automated windowing procedure
    m=np.arange(len(taus))<c*taus
    if np.any(m):
        return np.argmin(m)
    return len(taus)-1

def integrated_time(x,c=5.0):
    f=autocorr_time(x)
    taus=2.0*np.cumsum(f)-1.0
    window=auto_window(taus,c)
    return max(1.0,taus[window])

def sim():
    results_list=[]

    np.random.seed(int(time.time()))
    sweeps_arr=np.arange(total_sweeps)
    num_N=len(NSpins_vals)

    fig_main=plt.figure(figsize=(16,3*num_N))
    gs_main=fig_main.add_gridspec(num_N,2,width_ratios=[1.2,1])
    ax_cv_combined=fig_main.add_subplot(gs_main[:,1])

    fig_ediff,axes_ediff=plt.subplots(num_N,1,figsize=(12,3*num_N),sharex=True)
    if num_N==1:
        axes_ediff=[axes_ediff]

    T_smooth=np.linspace(0.4,5.5,300)
    T_smooth_over_J=T_smooth/J
    T_vals_over_J=[t/J for t in T_vals]

    colors=plt.cm.viridis(np.linspace(0.1,0.85,len(NSpins_vals)))
    t_colors=plt.cm.coolwarm(np.linspace(0.1,0.9,len(T_vals)))
    
    print("running simulation for all NSpins...")

    for idx,NSpins in enumerate(NSpins_vals):
        ax_m=fig_main.add_subplot(gs_main[idx,0])
        color=colors[idx]

        mc_Cv_list=[]
        mc_Cv_err_list=[]
        mc_E_list=[]
        exact_F_list=[]
        mc_m_dict={}
        E_diffs={}
        print(f"Running 1D Spin Chain Simulation (N={NSpins})...")
        t0=time.time()

    #thermo calc
        for T in T_vals:
            m_hist,E_hist=chain(NSpins, total_sweeps, burn_in, T, J, mu, H)

            #magnetisation running mean 
            m_running_mean=np.cumsum(m_hist)/(sweeps_arr+1)
       
            E_mean=np.mean(E_hist)/NSpins
            E_var=np.var(E_hist)
            E_diffs[T]=np.diff(E_hist)
            #print(f"time average energy difference={np.mean(E_diffs[T])}")

            Cv=E_var/(NSpins*(T**2))
       
            free_energy_approx=E_mean-T*np.log(2.0)

            free_energy_exact,_,_=free_energy_ons(NSpins,T,J,mu,H)
        
            num_samples=len(E_hist)
            #Cv_err=Cv*np.sqrt(2.0/(num_samples-1))
            #calculating c_v error using autocorrelation time
            tau_int=integrated_time(E_hist)
            N_eff=num_samples/tau_int
            Cv_err=Cv*np.sqrt(2.0/(N_eff-1))

            mc_E_list.append(E_mean)
            mc_Cv_list.append(Cv)
            mc_Cv_err_list.append(Cv_err)
            exact_F_list.append(free_energy_exact)
            mc_m_dict[T]=m_running_mean

            #results for saving to csv
            results_list.append({
                "NSpins":NSpins,
                "T":T,
                "E_mean":E_mean,
                "Cv":Cv,
                "Cv_err":Cv_err,
                "F_approx":free_energy_approx,
                "F_exact":free_energy_exact,
                "m_running_mean":m_running_mean[-1],
                "tau_int":tau_int,
                "Mean Energy Difference":np.mean(E_diffs[T])
            })
            print("-"*40)
            print(f"N={NSpins} ; T={T}")
            print(f"Avg. Energy <E>: {E_mean:.4f} per spin")
            print(f"Specific Heat (Cv):{Cv:.4f} per spin")
            print(f"Approx. Free Energy (F):{free_energy_approx:.4f} per spin")
            print(f"running mean magnetisation time avergage m(t): {m_running_mean[-1]:.4f}")
            print("-"*40)
        
        #ENERGY DIFF PLOT
        #running average
        
        for t_idx,T in enumerate(T_vals):
            window=100
            smoothed_dE=np.convolve(E_diffs[T],np.ones(window)/window,mode='valid')
            ax_ediff=axes_ediff[idx]
            ax_ediff.plot(
                np.arange(len(smoothed_dE)), smoothed_dE, color=t_colors[t_idx], alpha=0.7, label=f"$T={T:.1f}J$")
        ax_ediff.axhline(y=0.0,color='black',linestyle="--",alpha=0.7)
        ax_ediff.set_ylabel(r"Energy Difference ($\langle \Delta E \rangle$)")
        ax_ediff.set_title(f"Energy Difference Trajectories ($N={NSpins}$)")
        ax_ediff.grid(True, linestyle=":", alpha=0.6)
        ax_ediff.legend(fontsize='small',loc='upper right',ncol=2)
        
        #MAGNETISATION PLOT - only burn in sweeps
        sweeps_arr=np.arange(total_sweeps)
        for T in T_vals:
            ax_m.plot(
                sweeps_arr[:burn_in], mc_m_dict[T][:burn_in], alpha=0.7, label=f"$T={T:.1f}J$"
            )
        ax_m.axvline(x=burn_in, color='black', linestyle="--", alpha=0.7, label="Burn-in End")
        ax_m.set_xlabel("Monte Carlo Steps")
        ax_m.set_ylabel("Mean Magnetisation ($<m>$)")
        ax_m.set_title(f"Running Mean Magnetisation Trajectories ($N={NSpins}$)")
        ax_m.grid(True, linestyle=":", alpha=0.6)
        ax_m.legend(fontsize='small',loc='upper right',ncol=2)

        ax_cv_combined.plot(
            T_smooth_over_J,exact_Cv(NSpins,T_smooth,J),"-",color=color,lw=1,label=f"Theory ($N={NSpins}$)"
        )
        ax_cv_combined.errorbar(
            T_vals_over_J,mc_Cv_list,yerr=mc_Cv_err_list,fmt="o",
            color=color,ecolor=color,elinewidth=1.2,
            capsize=3, label=f"MC ($N={NSpins}$)"
        )


    df=pd.DataFrame(results_list)
    df.to_csv("1d_openvsclosed_spin_chain_data.csv",index=False)
        
#bottom label for ediff fig

    axes_ediff[-1].set_xlabel("Monte Carlo Steps")
    fig_ediff.tight_layout()
    fig_ediff.savefig("1d_energy_diff_openvsclosed.png",dpi=300)

    #CV VS TEMP
    #Cv_theoretical=exact_Cv(NSpins,T_smooth,J)

    #ax_cv_combined.plot(T_smooth,Cv_theoretical,"k-",lw=2,label=r"Theoretical")
    #ax_cv_combined.errorbar(T_vals,mc_Cv_list,yerr=mc_Cv_err_list,fmt="o",color="crimson",ecolor="black", elinewidth=1.5,capsize=4,label="MC Specific Heat $C_v$",)
    ax_cv_combined.set_xlabel("Temperature ($T/J$)")
    ax_cv_combined.set_ylabel("Specific Heat per Spin ($C_v$)")
    ax_cv_combined.set_title("Specific Heat vs Temperature")
    ax_cv_combined.grid(True, linestyle=":",alpha=0.6)
    ax_cv_combined.legend()
    fig_main.tight_layout()
    fig_main.savefig("1d_results_openvsclosed.png",dpi=300)
    plt.show()


if __name__=='__main__':
    sim()

