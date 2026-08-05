##heat baths for 1d spin chain with open boundary conditions
#heat bath in middle, cold bath at ends 

import numpy as np
import time
from numba import njit
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
NSpins_vals = [8,16,32,64,128]
J, mu, H = 1, 1.0, 0

T_middle=5.0*J
T_ends=1.0*J

burn_in=10000
sweeps=100000
total_sweeps=burn_in+sweeps

@njit
def chain(NSpins, total_sweeps, burn_in, T_middle, T_ends, J, mu, H):
    
    spin = np.ones(NSpins, dtype=np.int8)
    bond_energies=np.zeros(NSpins-1,dtype=np.float64)


    heat_left_to_right=0.0

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

            if r<(NSpins)//4 or r>3*(NSpins)//4:
                beta=1.0/T_ends

            else:
                beta=1.0/T_middle

            #Metropolis
            if dE<=0.0 or np.random.random()<np.exp(-dE*beta):
                spin[r]=-spin[r]
                if sweep>=burn_in and (r<(NSpins)//4 or r>3*(NSpins)//4):
                    heat_left_to_right+=dE
                #tracking heat flow

        if sweep>=burn_in:
            for i in range(NSpins-1):
                bond_energies[i]+=-J*spin[i]*spin[i+1]

        num_measurements=total_sweeps-burn_in
        heat_flux=heat_left_to_right/num_measurements
        avg_bond_energies=bond_energies/num_measurements
    return heat_flux, avg_bond_energies

def pe_chain(NSpins, total_sweeps, burn_in, T_middle, T_ends, J, mu, H):
    
    spin = np.ones(NSpins, dtype=np.int8)
    pe_bond_energies=np.zeros(NSpins,dtype=np.float64)

    pe_heat_left_to_right=0.0

    for sweep in range(total_sweeps):
        for _ in range(NSpins):            
            r=np.random.randint(0,NSpins)
            #open BCs
            nb=spin[(r-1)%NSpins]+spin[(r+1)%NSpins]
            #need to take coupling and neighbour alignment into account!
            dE=2.0*spin[r] *(J*nb+mu*H)

            if r<(NSpins)//4 or r>3*(NSpins)//4:
                beta=1.0/T_ends
            else:
                beta=1.0/T_middle

            #Metropolis
            if dE<=0.0 or np.random.random()<np.exp(-dE*beta):
                spin[r]=-spin[r]
                if sweep>=burn_in and (r<(NSpins)//4 or r>3*(NSpins)//4):
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

    np.random.seed(int(time.time()))

    fig, (ax_flux,ax_profile)=plt.subplots(1,2,figsize=(14,5))
    colors=plt.cm.viridis(np.linspace(0.1,0.85,len(NSpins_vals)))

    print("running simulation for all NSpins...")
    print(f"baths: T_middle={T_middle}, T_ends={T_ends}")

    pe_fluxes=[]
    fluxes=[]
    for idx,NSpins in enumerate(NSpins_vals):
        t0=time.time()
        heat_flux, avg_bond_energies=chain(NSpins,total_sweeps,burn_in,T_middle,T_ends,J,mu,H)
        pe_heat_flux, pe_avg_bond_energies=pe_chain(NSpins,total_sweeps,burn_in,T_middle,T_ends,J,mu,H)
        t_elapsed=time.time()-t0
        fluxes.append(heat_flux)
        pe_fluxes.append(pe_heat_flux)

        results_list.append({ 'NSpins': NSpins, 'heat_flux': heat_flux, 'avg_bond_energies': avg_bond_energies, 'time_elapsed': t_elapsed,
                             'pe_heat_flux': pe_heat_flux, 'pe_avg_bond_energies': pe_avg_bond_energies })
        print(f"NSpins={NSpins}, heat_flux={heat_flux:.6f}, pe_heat_flux={pe_heat_flux:.6f}, time_elapsed={t_elapsed:.2f}s")

        x_node_open=np.linspace(0,1,NSpins-1)
        x_node_periodic=np.linspace(0,1,NSpins)
        ax_profile.plot(x_node_open,avg_bond_energies,'x-',label=f"N={NSpins}",color=colors[idx])
        ax_profile.plot(x_node_periodic,pe_avg_bond_energies,'o--',label=f"N={NSpins} (periodic)",color=colors[idx],alpha=0.5)
    ax_flux.plot(NSpins_vals,np.abs(fluxes),'x--',color='crimson',lw=2,ms=8, label="Open BCs")
    ax_flux.plot(NSpins_vals,np.abs(pe_fluxes),'o--',color='navy',lw=2,ms=8,alpha=0.7, label="Periodic BCs")
    ax_flux.set_xlabel("NSpins")
    ax_flux.set_ylabel("Absolute Heat flux")
    ax_flux.set_title("Heat flux vs NSpins")
    ax_flux.grid(True,which='both',linestyle=":",alpha=0.6)

    ax_profile.set_xlabel("Normalized position along chain")
    ax_profile.set_ylabel("Average bond energy")
    ax_profile.set_title("Average bond energy profile")
    ax_profile.grid(True,which='both',linestyle=":",alpha=0.6)
    ax_profile.legend()

    plt.tight_layout()
    plt.savefig("middle_heat_flux_and_profile.png",dpi=300)

    df = pd.DataFrame(results_list)
    df.to_csv("middle_heat_transport.csv", index=False)
    plt.show()



if __name__=='__main__':
    sim()

