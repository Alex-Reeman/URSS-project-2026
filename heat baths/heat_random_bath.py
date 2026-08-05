##heat baths for 1d spin chain with open boundary conditions
#each spin is coupled to a local bath randomly 

import numpy as np
import time
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
NSpins_vals = [8,16,32,64,128]
J, mu, H = 1, 1.0, 0

T_left=5.0*J
T_right=1.0*J

burn_in=10000
sweeps=100000
total_sweeps=burn_in+sweeps

@njit
def chain(NSpins, total_sweeps, burn_in, T_left, T_right, J, mu, H):
    
    spin = np.ones(NSpins, dtype=np.int8)
    bond_currents=np.zeros(NSpins-1,dtype=np.float64)
    current_M = float(NSpins)

    beta_left = 1.0/T_left
    beta_right = 1.0/T_right

    m_hist=np.empty(total_sweeps,dtype=np.float64)
    E_hist=np.empty(total_sweeps,dtype=np.float64)
    
    for sweep in range(total_sweeps):
        for _ in range(NSpins):
            r=np.random.randint(0,NSpins)
            heat=np.random.randint(0,2) #randomly choose which bath to couple to
            #open BCs
            nb=0.0
            if r>0:
                nb+=spin[r-1]
            if r<NSpins-1:
                nb+=spin[r+1]
            #need to take coupling and neighbour alignment into account!
            dE=2.0*spin[r] *(J*nb+mu*H)

            #assigning local bath temperature
            if heat==0:
                beta=beta_left
                
            elif heat==1:
                beta=beta_right
                
            #Metropolis
            if dE<=0.0 or np.random.random()<np.exp(-dE*beta):
                current_M-=2.0*spin[r]

                #if connected to hot bath
                if sweep>=burn_in:
                    if heat==0:
                        bond_currents[r-1]+=2.0*J*spin[r]*spin[r-1]
                    if heat==1:
                        bond_currents[r]-=2.0*J*spin[r]*spin[r+1]
                spin[r]=-spin[r]
                #tracking heat flow

        m_hist[sweep]=current_M/NSpins
        total_E=0.0
        for i in range(NSpins-1):
            total_E+=-J*spin[i]*spin[i+1]-mu*H*spin[i]

        E_hist[sweep]=total_E/NSpins

        num_measurements=total_sweeps-burn_in
        mean_energy_current=bond_currents/num_measurements
    return mean_energy_current,m_hist, E_hist

#def pe_chain(NSpins, total_sweeps, burn_in, T_left, T_right, J, mu, H):
    
    spin = np.ones(NSpins, dtype=np.int8)
    pe_bond_energies=np.zeros(NSpins,dtype=np.float64)
    beta_left = 1.0/T_left
    beta_right = 1.0/T_right

    pe_heat_left_to_right=0.0

    for sweep in range(total_sweeps):
        for _ in range(NSpins):            
            r=np.random.randint(0,NSpins)
            heat=np.random.randint(0,2)
            #periodic
            nb=spin[(r-1)%NSpins]+spin[(r+1)%NSpins]
            #need to take coupling and neighbour alignment into account!
            dE=2.0*spin[r] *(J*nb+mu*H)

            if heat==0:
                beta=beta_left
                    
            elif heat==1:
                beta=beta_right
                
            #Metropolis
            if dE<=0.0 or np.random.random()<np.exp(-dE*beta):
                spin[r]=-spin[r]
                if sweep>=burn_in and heat==0:
                    pe_heat_left_to_right+=dE
                #tracking heat flow

        if sweep>=burn_in:
            for i in range(NSpins):
                pe_bond_energies[i]+=-J*spin[i]*spin[(i+1)%NSpins]

        num_measurements=total_sweeps-burn_in
        pe_heat_flux=pe_heat_left_to_right/num_measurements
        pe_avg_bond_energies=pe_bond_energies/num_measurements
    return pe_heat_flux, pe_avg_bond_energies

def sim():
    results_list=[]
    sweeps_arr=np.arange(total_sweeps)

    np.random.seed(int(time.time()))

    fig, (ax_mag,ax_energy,ax_current)=plt.subplots(1,3,figsize=(22,5))
    colors=plt.cm.viridis(np.linspace(0.1,0.85,len(NSpins_vals)))

    print("running simulation for all NSpins...")
    print(f"baths: T_left={T_left}, T_right={T_right}")

    for idx,NSpins in enumerate(NSpins_vals):
        
        t0=time.time()
        mean_energy_current,m_hist,E_hist=chain(NSpins,total_sweeps,burn_in,T_left,T_right,J,mu,H)
        #pe_heat_flux, pe_avg_bond_energies=pe_chain(NSpins,total_sweeps,burn_in,T_left,T_right,J,mu,H)
        t_elapsed=time.time()-t0
        #fluxes.append(heat_flux)
        #pe_fluxes.append(pe_heat_flux)
        m_running_mean=np.cumsum(m_hist)/(sweeps_arr+1)
        E_running_mean=E_hist.cumsum()/(sweeps_arr+1)


        results_list.append({ 'NSpins': NSpins, 'mean_energy_current': mean_energy_current, 'm_running_mean': m_running_mean, 'time_elapsed': t_elapsed })
        print(f"NSpins={NSpins}, mean_energy_current={mean_energy_current}, time_elapsed={t_elapsed:.2f}s")

        #x_node_open=np.linspace(0,1,NSpins-1)
        #x_node_periodic=np.linspace(0,1,NSpins)
        
        ax_mag.plot(np.arange(total_sweeps),m_running_mean,label=f"N={NSpins}",color=colors[idx],alpha=0.7)

        ax_energy.plot(sweeps_arr,E_running_mean,label=f"N={NSpins}",color=colors[idx],alpha=0.7)

        bond_positions=np.linspace(0,1,len(mean_energy_current))
        ax_current.plot(bond_positions,mean_energy_current,label=f"N={NSpins}",color=colors[idx],alpha=0.7)

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
    plt.savefig("randbath_heat_flux_and_profile.png",dpi=300)

    df = pd.DataFrame(results_list)
    df.to_csv("randbath_heat_transport.csv", index=False)
    plt.show()



if __name__=='__main__':
    sim()

