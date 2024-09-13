import numpy as np
from numpy import matmul as mm
from numpy.lib.scimath import sqrt as csqrt
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import scipy.linalg as spla
import sys

fontsize = 24
params = {'legend.fontsize': fontsize,
          'figure.figsize': (14, 10),
          'figure.titlesize': fontsize,
         'axes.labelsize': fontsize,
         'axes.titlesize': fontsize,
         'xtick.labelsize': fontsize,
         'ytick.labelsize': fontsize}
pylab.rcParams.update(params)

# Structured bath size (#resonators)
N = 100

# Energy units where J=1 throughout
# Detuning of atom from bath
Delta = 0. #2.71
# Bath-Atom couplings
G = 0.25
nCpts = 4
g = G / np.sqrt(nCpts)

# Coupling points
nRad = 2*0+1
mRad = 2*0+1
offsets = np.array([[-(nRad+1)//2 - (mRad-1)//2, -(nRad-1)//2 + (mRad-1)//2],
                    [(nRad-1)//2 - (mRad-1)//2, (nRad+1)//2 + (mRad-1)//2],
                    [(nRad-1)//2 + (mRad+1)//2, (nRad+1)//2 - (mRad+1)//2],
                    [-(nRad+1)//2 + (mRad+1)//2, -(nRad-1)//2 - (mRad+1)//2]
                    ]) # Offset for GA coupling points relative to the centre
                    
if (len(offsets) != nCpts):
    sys.exit(f'Offset list has too many or too few entries (length = {len(offsets)}): Should be equal to nCpts = {nCpts}')

centrePos = np.array([[N//2 - 1, N//2 - 1]]*nCpts) # Centre of the lattice
couplingPts = centrePos + offsets # Position of 1st coupling points
print(couplingPts)
#Make a cycle out of the coupling points for plotting later
x, y = couplingPts.T
x = np.append(x, x[0])
y = np.append(y, y[0])
# Indices for the coupling points (flattened array of lattice positions)
cpLocs = [couplingPts[p,0]+couplingPts[p,1]*N for p in range(nCpts)]
# Sign & Indices for the interaction points
signIpLocs = [((-1)**(mi+ni+(nRad-1)//2+(mRad-1)//2+1),centrePos[0,0]+centrePos[0,1]*N + ni*(1+N) + mi*(1-N)) for ni in range(-(nRad-1)//2,(nRad+1)//2) for mi in range(-(mRad-1)//2,(mRad+1)//2)]

# Momentum space stuff for bath
dk = 2. * np.pi / N # Spacing
ks = np.linspace(-np.pi, np.pi-dk, N) # List of allowed ks
ws = np.array([[-2 * (np.cos(ki) + np.cos(kj)) for kj in ks] for ki in ks]) # Dispersion relation

######################################## Time Evolution ########################################
# Time step & #steps
dt = 1e-2
nSteps = int(N/(2*dt))

# Matrix to hold all info about our state's time evolution (real space)
# Atom first, then bath
stateEvol = np.zeros((nSteps+1,N*N+1), dtype='complex64')

# Initial state: GA fully excited (dressed state)
stateEvol[0,0] = 1.+0.j #0.9449088 #1./(1. + g**2 * nRad * mRad)

#Populating the BIC in the dressed state (uncomment for bare excited GA)
# BICpeaks = [g,-g] # even peak, odd peak
# for s,ipLoc in signIpLocs:
#     stateEvol[0,1+ipLoc] = BICpeaks[int((1-s)/2)]

#cPtPop = 9.1196506e-07
#for cpLoc in cpLocs:
#    stateEvol[0,1+cpLoc] = -np.sqrt(cPtPop)

# Force unitarity
stateEvol[0,:] /= np.sqrt(np.sum(stateEvol[0,:].real**2 + stateEvol[0,:].imag**2))


# Effective ham_AI & the t-ev op for ham_AI_eff
ham_AI_eff = np.zeros((2,2), dtype='complex64')
ham_AI_eff[0,0] = Delta
ham_AI_eff[0,1] = np.sqrt(nCpts) * g
ham_AI_eff[1,0] = np.sqrt(nCpts) * g
tEvOp_eff = spla.expm(-1.j * ham_AI_eff * dt)
print(tEvOp_eff)

# Time evolution operator for bath in k-space
tEvOp_kspace = np.exp(-1.j * ws * dt)

for ti in range(nSteps):
    # Evolve using H_AI in real space
    stateEvol[ti+1,:] = stateEvol[ti,:]
    intState = 0
    for cpLoc in cpLocs:
        intState += stateEvol[ti,1+cpLoc]

    stateEvol[ti+1,0] = tEvOp_eff[0,0] * stateEvol[ti,0] + tEvOp_eff[0,1]/np.sqrt(nCpts) * intState
    for cpLoc in cpLocs:
        stateEvol[ti+1,1+cpLoc] += tEvOp_eff[0,1]/np.sqrt(nCpts) * stateEvol[ti,0] + (tEvOp_eff[1,1] - 1.)/nCpts * intState

    # Force unitarity to limit numerical errors
    stateEvol[ti+1,:] /= np.sqrt(np.sum(stateEvol[ti+1,:].real**2 + stateEvol[ti+1,:].imag**2))

    # Go into k-space
    psi_bath_k = np.fft.fft2(np.reshape(stateEvol[ti+1,1:],(N,N)))

    # Apply exact time evolution operator for H_B = diag(ws)
    psi_bath_k = tEvOp_kspace * psi_bath_k

    # Go back into real space
    stateEvol[ti+1,1:] = np.fft.ifft2(psi_bath_k).flatten()

# Plotting
fig,axs = plt.subplots(2,2)

# Plot atomic pop
tiMax = (7*nSteps)//8
ts = np.linspace(0., dt*nSteps, nSteps+1)
pAtomExcs = stateEvol[:,0].real**2 + stateEvol[:,0].imag**2
axs[0,0].plot(ts[:tiMax+1], pAtomExcs[:tiMax+1], linewidth=3, color='C1')

# Final atomic pop (theoretical)
Csq_infty = 1./(1. + g**2 * nRad * mRad)**2
print('Theoretical steady-state population:')
print(Csq_infty)
axs[0,0].plot([ts[0],ts[tiMax]], [Csq_infty,Csq_infty], '--k', label=r'$|C_e{(\infty)}|^2$')

#Plot interference buildup time
tIntf = max(nRad,mRad)*1
# axs[0,0].plot([ts[int(tIntf/dt)],ts[int(tIntf/dt)]], [Csq_infty, 1], 'k--', label=r'Interference starts ($\tau = %d/J$)' %tIntf)
#axs[0,0].plot(ts[:tiMax+1], np.exp(-2* g**2 * ts[:tiMax+1]), ":", label=r'exp(-2g^2t/J)')
axs[0,0].set(xlabel='$tJ$', ylabel='$|C_e(t)|^2$')
axs[0,0].legend()
axs[0,0].set_title(r"$\bf{(a)}$"+' Atomic population', loc="left")


#Plot center cavity over time
#calculate the bath population at time tiMax:
pCenter = np.abs(stateEvol[tiMax+1, 1:])**2
#take the most populated cavity and show the time evolution:
pCenterExc = np.abs(stateEvol[:tiMax+1, pCenter.argmax()+1])**2 
axs[0,1].plot(ts[:tiMax+1], pCenterExc[:tiMax+1], linewidth=3, color='C2', label=r'$|C_B{(t)}|^2$')
axs[0,1].set(xlabel='$tJ$', ylabel='$|C_{B}(t)|^2$')
axs[0,1].set_title(r"$\bf{(b)}$"+r' Bath population at $\vec{n} = (50,50)$', loc='right')


# Plot the state of the bath at 2 time steps
# Maximum bath excitation for consistent scaling
#tiMin = (3*nSteps)//4
maxBathExc = np.max(abs(stateEvol[:,1:])**2)

tJ = 1
tiPlot = int(tJ/dt)
margin = 2 # Margin around the atoms when plotting
pBathExc = np.reshape(stateEvol[tiPlot,1:].real**2 + stateEvol[tiPlot,1:].imag**2,(N,N))
plotSize = [N//2-((nRad+1)//2+(mRad-1)//2+margin+1), N//2+((nRad-1)//2+(mRad+1)//2+margin), N//2-((nRad-1)//2+(mRad+1)//2+margin+1), N//2+((nRad+1)//2+(mRad-1)//2+margin)]
axs[1,0].imshow(pBathExc[plotSize[0]:plotSize[1],plotSize[2]:plotSize[3]], origin="lower", extent=[xyVal + 0.5 for xyVal in plotSize], norm=colors.LogNorm(vmin=maxBathExc/150, vmax=maxBathExc), aspect='auto', cmap='Greens')
# for i in range(nCpts):
#     axs[0,1].scatter(couplingPts[i,0]+1, couplingPts[i,1]+1, marker='o', color='r', linewidth=4, s=200)
axs[1,0].plot(x+1,y+1, marker='o', markersize=15, ls='-', lw=5, color='C1')
axs[1,0].set(xlabel='Resonator index, $n_x$', ylabel='Resonator index, $n_y$')
axs[1,0].set_title(r"$\bf{(c)}$"+f' Bath population at $tJ = {tJ}$', loc="left")

tJ = 40
tiPlot = int(tJ/dt)
pBathExc = np.reshape(stateEvol[tiPlot,1:].real**2 + stateEvol[tiPlot,1:].imag**2,(N,N))
bathFinal = axs[1,1].imshow(pBathExc[plotSize[0]:plotSize[1],plotSize[2]:plotSize[3]], origin="lower", extent=[xyVal + 0.5 for xyVal in plotSize], norm=colors.LogNorm(vmin=maxBathExc/150, vmax=maxBathExc), aspect='auto' , cmap='Greens')
# for i in range(nCpts):
#     axs[1,1].scatter(couplingPts[i,0]+1, couplingPts[i,1]+1, marker='o', color='r', linewidth=4, s=200)
axs[1,1].plot(x+1,y+1, marker='o', markersize=15, ls='-', lw=5, color='C1')
axs[1,1].set(xlabel='Resonator index, $n_x$', ylabel='Resonator index, $n_y$')
axs[1,1].set_title(r"$\bf{(d)}$"+f' Bath population at $tJ = {tJ}$', loc='right')

print('Final population in simulation:')
print(pAtomExcs[tiMax])
print('Peaks of BIC standing wave at the end:')
print(pBathExc[N//2,N//2])
print(pBathExc[N//2-1,N//2-1])
print('NB: The locations for these are in the centre of the grid and [-1,-1] away from this point.')
print('Pop @ coupling points at the end:')
print(pBathExc[couplingPts[0,0],couplingPts[0,1]])
print('--------------------')
print('Atom & BIC states at the end:')
print(stateEvol[tiMax,0])
print([stateEvol[tiMax,1+ipLoc] for _,ipLoc in signIpLocs])
print('Population retained:')
print(pAtomExcs[tiMax] + sum([(pBathExc.flatten())[ipLoc] for _,ipLoc in signIpLocs]))

fig.tight_layout()
fig.subplots_adjust(right=0.85, top=0.87, wspace=0.4, hspace=0.5)
cbar_ax = fig.add_axes([0.1, -0.05, 0.75, 0.03]) #XY displacement, XY extent
cbar = fig.colorbar(bathFinal, cax=cbar_ax, orientation="horizontal", ticks=ticker.LogLocator(base=10, subs=(0.5, )), format='%.1e')
cbar.set_label(label="Bath population")

# Uncomment to save figure and define file path:
# fig.savefig('YourPath/1-GA-Subradiant_Dynamics.pdf', dpi=fig.dpi, bbox_inches='tight')
