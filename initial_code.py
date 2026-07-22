import numpy as np
import time
from numba import njit
import matplotlib.pyplot as plt

# Parameters
NSpins_vals = [8,16,32,64,128]
J, mu, H = 0.4, 1.0, 0
#T = 1.5
T_vals=[1*J,2*J,3*J,4*J,5*J]

burn_in=10000
sweeps=100000
total_sweeps=burn_in+sweeps

@njit
def chain(NSpins, total_sweeps, burn_in, T, J, mu, H):
    
    spin = np.ones(NSpins, dtype=np.int8)
    current_E = 0.0
    current_M = float(NSpins)
    beta = 1.0/T
    
    for i in range(NSpins):
        nb=spin[(i-1)%NSpins]+spin[(i+1)%NSpins]
        current_E+=-0.5*J*spin[i]*nb+mu*H*spin[i]

    num_steps=total_sweeps-burn_in
    
    m_hist=np.empty(total_sweeps,dtype=np.float64)
    E_hist=np.empty(num_steps,dtype=np.float64)
    step_idx=0

    for sweep in range(total_sweeps):
        for _ in range(NSpins):
            r=np.random.randint(0,NSpins)
            #periodic BCs
            nb=spin[(r-1)%NSpins]+spin[(r+1)%NSpins]
            #need to take coupling and neighbour alignment into account!
            dE=2.0*spin[r] *(J*nb-mu*H)

            #Metropolis
            if dE<=0.0 or np.random.random()<np.exp(-dE*beta):
                current_E+=dE
                current_M-=2.0*spin[r]
                spin[r]*=-1
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

def exact_Cv(T,J):
    beta=1/T
    return (beta*J)**2*(1.0/np.cosh(beta*J))**2


def sim():
    np.random.seed(int(time.time()))
    sweeps_arr=np.arange(total_sweeps)
    num_N=len(NSpins_vals)

    fig=plt.figure(figsize=(16,3*num_N))
    gs=fig.add_gridspec(num_N,2,width_ratios=[1.2,1])

    colors=plt.cm.plasma(np.linspace(0.1,0.85,len(T_vals)))
    print("running simulation for all NSpins...")

    ax_cv_combined=fig.add_subplot(gs[:,1])

    for idx,NSpins in enumerate(NSpins_vals):
        ax_m=fig.add_subplot(gs[idx,0])

        mc_Cv_list=[]
        mc_Cv_err_list=[]
        mc_E_list=[]
        exact_F_list=[]
        mc_m_dict={}
        print(f"Running 1D Spin Chain Simulation (N={NSpins})...")
        t0=time.time()
        print(f"Simulation completed")

    #thermo calc
        for T in T_vals:
            m_hist,E_hist=chain(NSpins, total_sweeps, burn_in, T, J, mu, H)
       
            E_mean=np.mean(E_hist)/NSpins
            E_var=np.var(E_hist)

            Cv=E_var/(NSpins*(T**2))
       
            free_energy_approx=E_mean-T*np.log(2.0)

            free_energy_exact,_,_=free_energy_ons(NSpins,T,J,mu,H)
        
            num_samples=len(E_hist)
            Cv_err=Cv*np.sqrt(2.0/(num_samples-1))

            mc_E_list.append(E_mean)
            mc_Cv_list.append(Cv)
            mc_Cv_err_list.append(Cv_err)
            exact_F_list.append(free_energy_exact)
            mc_m_dict[T]=m_hist

            print("-"*40)
            print(f"N={NSpins} ; T={T}")
            print(f"Avg. Energy <E>: {E_mean:.4f} per spin")
            print(f"Specific Heat (Cv):{Cv:.4f} per spin")
            print(f"Approx. Free Energy (F):{free_energy_approx:.4f} per spin")
            print("-"*40)

        
        #MAGNETISATION PLOT
        sweeps_arr=np.arange(total_sweeps)
        for T in T_vals:
            ax_m.plot(
                sweeps_arr, mc_m_dict[T], alpha=0.7, label=f"$T={T:.1f}J$"
            )
        ax_m.axvline(x=burn_in, color='black', linestyle="--", alpha=0.7, label="Burn-in End")
        ax_m.set_xlabel("Monte Carlo Steps")
        ax_m.set_ylabel("Magnetisation per spin ($m$)")
        ax_m.set_title("Magnetisation Trajectories across Temperatures")
        ax_m.grid(True, linestyle=":", alpha=0.6)
        ax_m.legend(fontsize='small',loc='upper right',ncol=2)

        ax_cv_combined.errorbar(
            T_vals,mc_Cv_list,yerr=mc_Cv_err_list,fmt="o--",
            capsize=3, label=f"MC ($N={NSpins}$)"
        )
    
    #CV VS TEMP
    T_smooth=np.linspace(0.4,5.5,300)
    Cv_theoretical=exact_Cv(T_smooth,J)

    ax_cv_combined.plot(T_smooth,Cv_theoretical,"k-",lw=2,label=r"Theoretical")
    ax_cv_combined.errorbar(T_vals,mc_Cv_list,yerr=mc_Cv_err_list,fmt="o",color="crimson",ecolor="black", elinewidth=1.5,capsize=4,label="MC Specific Heat $C_v$",)
    ax_cv_combined.set_xlabel("Temperature ($T/J$)")
    ax_cv_combined.set_ylabel("Specific Heat per Spin ($C_v$)")
    ax_cv_combined.set_title("Specific Heat vs Temperature")
    ax_cv_combined.grid(True, linestyle=":",alpha=0.6)
    ax_cv_combined.legend()

    plt.tight_layout()
    plt.savefig("1d_results_J=0.4.png",dpi=300)
    plt.show()


if __name__=='__main__':
    sim()

