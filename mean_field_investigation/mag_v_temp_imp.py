import matplotlib.pyplot as plt
import numpy as np

# Bath temperatures
t_cold  = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0,2.0,3.0,2.0])
t_hot   = np.array([5.0, 4.0, 3.0, 2.0, 5.0, 5.0, 5.0,4.0,4.0,3.0])

# Mean Field Calculations (MFC)
H=0.1 
mag_mfc = np.array([0.0966, 0.0989, 0.1028, 0.1106, 0.0684, 0.0590, 0.0543 ,0.0707 , 0.0613, 0.0745])
mag_sim = np.array([0.0798,0.0842,0.0916,0.1074,0.0413,0.0305,0.0255,0.0446,0.0339,0.0506])

#H=0.02
#mag_mfc = np.array([0.0327706,0.0351513 ,0.0394279,0.049346 , 0.0148443, 0.0105068,0.00855851,0.0162828,0.0117461,0.0188319])
#mag_sim = np.array([0.0165,0.0170,0.0185,0.0217,0.0088,0.0062,0.0048,0.0085,0.0069,0.0096])

#H=0.04
#mag_mfc = np.array([0.0663726,0.0713696191510557,0.080414083636910, 0.101796275764772,0.0298131481020301 ,0.0210743196179813 ,0.0171566851744617,0.0327334375748275,0.0235746979337126,0.0379188681620913])
#mag_sim = np.array([0.0317,0.0342,0.0365,0.0439,0.0169,0.0123,0.0102,0.0183,0.0133,0.0206])

#H=0.06
#mag_mfc = np.array([0.101,0.109,0.124,0.159,0.045,0.032,0.026,0.049,0.035,0.057])
#mag_sim = np.array([0.0478,0.0510,0.0559,0.0651,0.0255,0.0182,0.0155,0.0272,0.0206,0.0306])

#H=0.08
#mag_mfc = np.array([0.137,0.148,0.170,0.224,0.060,0.042,0.034,0.066,0.048,0.077])
#mag_sim = np.array([0.0641,0.0670,0.0732,0.0868,0.0341,0.0237,0.0204,0.0359,0.0271,0.0406])

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    t_cold, t_hot, mag_mfc, 
    color='royalblue', marker='o', s=70, edgecolors='k', label='MFC (Calculated)'
)

ax.scatter(
    t_cold, t_hot, mag_sim, 
    color='crimson', marker='^', s=80, edgecolors='k', label='Simulation'
)

for i in range(len(t_cold)):
    ax.plot(
        [t_cold[i], t_cold[i]], 
        [t_hot[i], t_hot[i]], 
        [mag_mfc[i], mag_sim[i]], 
        color='black', linestyle='--', alpha=0.5, linewidth=1
    )

ax.set_xlabel('$T_{cold}$', fontsize=12, labelpad=10)
ax.set_ylabel('$T_{hot}$', fontsize=12, labelpad=10)
ax.set_zlabel('Magnetization ($m$)', fontsize=12, labelpad=10)
ax.set_title('Magnetization Comparison: MFC vs. Simulation, H=0.08', fontsize=13)

ax.legend(loc='upper left', fontsize=11)
ax.view_init(elev=25, azim=135)

plt.tight_layout()
plt.savefig("magvtemp_h=0.1_corrections2.png", dpi=300)
plt.show()