import numpy as np
import time
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
NSpins_vals = [8,16,32,64,128]
J, mu, H = 0.25, 1.0,0.08

T_left=3.0
T_right=2.0

burn_in=10000
sweeps=100000
total_sweeps=burn_in+sweeps

@njit

def chain(NSpins, total_sweeps, burn_in, T_left, T_right, J, mu, H):
    
    spin = np.ones(NSpins, dtype=np.int8)
    bond_currents=np.zeros(NSpins,dtype=np.float64)
    current_M = float(np.sum(spin))

    beta_c=1.0/T_right
    beta_h=1.0/T_left

    m_hist=np.empty(total_sweeps,dtype=np.float64)
    E_hist=np.empty(total_sweeps,dtype=np.float64)

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
                if sweep>=burn_in:
                    bond_currents[(r-1)%NSpins]-=2.0*J*spin[r]*spin[(r-1)%NSpins]
                    bond_currents[r]+=2.0*J*spin[r]*spin[(r+1)%NSpins]
                current_M-=2*spin[r]
                spin[r]=-spin[r]

        m_hist[sweep]=current_M/NSpins
        total_E=0.0
        for i in range(NSpins):
            total_E+=spin[i]*spin[(i+1)%NSpins]
        total_E=-J*total_E-mu*H*current_M
        E_hist[sweep]=total_E/NSpins

    num_measurements=total_sweeps-burn_in
    mean_energy_current=bond_currents/num_measurements
    total_energy_current=np.sum(mean_energy_current)

    return mean_energy_current,m_hist, E_hist, total_energy_current


def sim():
    results_list=[]
    sweeps_arr=np.arange(total_sweeps)

    np.random.seed(int(time.time()))

    fig, (ax_mag,ax_energy,ax_current,ax_total_energy)=plt.subplots(4,1,figsize=(15,15))
    colors=plt.cm.viridis(np.linspace(0.1,0.85,len(NSpins_vals)))

    print("running simulation for all NSpins...")
    print(f"baths: T_left={T_left}, T_right={T_right}")

    for idx,NSpins in enumerate(NSpins_vals):
        
        t0=time.time()
        mean_energy_current,m_hist,E_hist,total_energy_current=chain(NSpins,total_sweeps,burn_in,T_left,T_right,J,mu,H)
        #pe_heat_flux, pe_avg_bond_energies=pe_chain(NSpins,total_sweeps,burn_in,T_left,T_right,J,mu,H)
        t_elapsed=time.time()-t0
        #fluxes.append(heat_flux)
        #pe_fluxes.append(pe_heat_flux)
        m_running_mean=np.cumsum(m_hist)/(sweeps_arr+1)
        E_running_mean=E_hist.cumsum()/(sweeps_arr+1)


        results_list.append({ 'NSpins': NSpins, 'mean_energy_current': mean_energy_current,'energy_running_mean': E_running_mean, 'm_running_mean': m_running_mean, 'time_elapsed': t_elapsed })
        print(f"NSpins={NSpins}, mean_energy_current={mean_energy_current}, time_elapsed={t_elapsed:.2f}s, mean magnetization={m_running_mean[-1]:.4f}")

        #x_node_open=np.linspace(0,1,NSpins-1)
        #x_node_periodic=np.linspace(0,1,NSpins)
        
        ax_mag.plot(np.arange(total_sweeps),m_running_mean,label=f"N={NSpins}",color=colors[idx],alpha=0.7)

        ax_energy.plot(sweeps_arr,E_running_mean,label=f"N={NSpins}",color=colors[idx],alpha=0.7)

        print(f"total energy current for NSpins={NSpins}: {total_energy_current}")

        bond_positions=np.linspace(0,1,len(mean_energy_current))
        ax_current.plot(bond_positions,mean_energy_current,label=f"N={NSpins}",color=colors[idx],alpha=0.7)

        ax_total_energy.plot(NSpins,total_energy_current,marker='o',label=f"N={NSpins}",color=colors[idx],alpha=0.7)
    ax_total_energy.set_xlabel("Number of spins (N)")
    ax_total_energy.set_ylabel("Total energy current")
    ax_total_energy.set_title("Total energy current vs number of spins")
    ax_total_energy.grid(True,which='both',linestyle=":",alpha=0.6)
    ax_total_energy.legend()

    ax_mag.set_xlabel("Monte Carlo sweeps")
    ax_mag.set_ylabel("Magnetization per spin")
    ax_mag.set_title("Magnetization vs sweeps")
    ax_mag.grid(True,which='both',linestyle=":",alpha=0.6)
    ax_mag.legend()

    ax_energy.set_xlabel("Monte Carlo sweeps")
    ax_energy.set_ylabel("Energy per spin")
    ax_energy.set_title("Running mean of energy per spin vs sweeps")
    ax_energy.grid(True,which='both',linestyle=":",alpha=0.6)
    ax_energy.legend()

    ax_current.set_xlabel("Bond position (normalized)")
    ax_current.set_ylabel("Mean energy current")
    ax_current.set_title("Mean energy current vs bond position")
    ax_current.grid(True,which='both',linestyle=":",alpha=0.6)
    ax_current.legend()
    
    
    plt.tight_layout()
    plt.savefig("closed_heat_flux_and_profile.png",dpi=300)
    df = pd.DataFrame(results_list)
    df.to_csv("closed_heat_transport.csv", index=False)
    plt.show()



if __name__=='__main__':
    sim()

