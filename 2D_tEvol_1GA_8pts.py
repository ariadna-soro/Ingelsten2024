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
N = 200

# Energy units where J=1 throughout
# Detuning of atom from bath
Delta = 0. #2.71
# Bath-Atom couplings
G = 0.25
nCpts = 8
g = G / np.sqrt(nCpts)

# Coupling points

#Constructive interference
# n1 = 1 
# m1 = 2
# n2 = 2
# m2 = 1
# shmDist = 0
# shmVec = np.array([0,0]) 
#Destructive interference
# n1 = 1 
# m1 = 2
# n2 = 2
# m2 = 0
# shmDist = 0
# shmVec = np.array([0,0])
#No interference
n1 = 1 
m1 = 2
n2 = 2
m2 = 0
shmDist = 1
shmVec = np.array([0,1])

nRad1 = 2*n1+1 #[1,1]
mRad1 = 2*m1+1 #[1,-1]
nRad2 = 2*n2+1 #[1,1]
mRad2 = 2*m2+1 #[1,-1]

nRad = max(nRad1,nRad2)
mRad = max(mRad1,mRad2)

# Displacement of the second subset from being perfectly centred
shmove = shmVec*shmDist #if shmDist=0, then shmVec needs to be [0,0]

offsets = np.array([[-(nRad1+1)//2-(mRad1-1)//2,
                     -(nRad1-1)//2+(mRad1-1)//2],
                    [(nRad1-1)//2-(mRad1-1)//2,
                     (nRad1+1)//2+(mRad1-1)//2],
                    [(nRad1-1)//2+(mRad1+1)//2,
                     (nRad1+1)//2-(mRad1+1)//2],
                    [(mRad1+1)//2-(nRad1+1)//2,
                     -(mRad1+1)//2-(nRad1-1)//2],
                    [shmove[0]-(nRad2+1)//2-(mRad2-1)//2,
                     shmove[1]-(nRad2-1)//2+(mRad2-1)//2],
                    [shmove[0]+(nRad2-1)//2-(mRad2-1)//2,
                     shmove[1]+(nRad2+1)//2+(mRad2-1)//2],
                    [shmove[0]+(nRad2-1)//2+(mRad2+1)//2,
                     shmove[1]+(nRad2+1)//2-(mRad2+1)//2],
                    [shmove[0]+(mRad2+1)//2-(nRad2+1)//2,
                     shmove[1]-(mRad2+1)//2-(nRad2-1)//2]]) # Offset for GA coupling points relative to the centre
if (len(offsets) != nCpts):
    sys.exit(f'Offset list has too many or too few entries (length = {len(offsets)}): Should be equal to nCpts = {nCpts}')

couplingPts = np.array([[N//2 - 1, N//2 - 1]]*nCpts) + offsets # Position of 1st coupling points
print(couplingPts)
#Make a cycle out of the coupling points for plotting later
x, y = couplingPts.T
x = np.insert(x, int(nCpts/2), x[0])
x = np.insert(x, int(nCpts+1), x[int(nCpts/2+1)])
y = np.insert(y, int(nCpts/2), y[0])
y = np.insert(y, int(nCpts+1), y[int(nCpts/2+1)])


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

# Initial state: GA fully excited
stateEvol[0,0] = 1.+0.j

# Effective ham_AI & the t-ev op for ham_AI_eff
ham_AI_eff = np.zeros((2,2), dtype='complex64')
ham_AI_eff[0,0] = Delta
ham_AI_eff[0,1] = np.sqrt(nCpts) * g
ham_AI_eff[1,0] = np.sqrt(nCpts) * g
tEvOp_eff = spla.expm(-1.j * ham_AI_eff * dt)
print(tEvOp_eff)

# # Resulting full t-ev op
# tEvOp_full = np.diag(np.ones(N*N+1, dtype='complex64'))
# tEvOp_full[0,0] = tEvOp_eff[0,0]
# for p in range(nCpts):
#     cpLoc1 = couplingPts[p,0] + couplingPts[p,1]*N
#     tEvOp_full[1+cpLoc1,0] = tEvOp_eff[1,0]/np.sqrt(nCpts)
#     tEvOp_full[0,1+cpLoc1] = tEvOp_eff[0,1]/np.sqrt(nCpts)
#     tEvOp_full[1+cpLoc1,1+cpLoc1] = 1. + (tEvOp_eff[1,1]-1.)/nCpts
#     for q in range(p+1,nCpts):
#         cpLoc2 = couplingPts[q,0] + couplingPts[q,1]*N
#         tEvOp_full[1+cpLoc1,1+cpLoc2] = (tEvOp_eff[1,1]-1.)/nCpts
#         tEvOp_full[1+cpLoc2,1+cpLoc1] = (tEvOp_eff[1,1]-1.)/nCpts


# Time evolution operator for bath in k-space
tEvOp_kspace = np.exp(-1.j * ws * dt)

for ti in range(nSteps):
    # Evolve using H_AI in real space
    stateEvol[ti+1,:] = stateEvol[ti,:]
    intState = 0
    cpLocs = [couplingPts[p,0]+couplingPts[p,1]*N for p in range(nCpts)]
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
plt.rcParams['figure.figsize'] = [14, 10]
fig,axs = plt.subplots(2,2)

# Plot atomic pop
tiMax = (7*nSteps)//8
ts = np.linspace(0., dt*nSteps, nSteps+1)
pAtomExcs = stateEvol[:,0].real**2 + stateEvol[:,0].imag**2
axs[0,0].plot(ts[:tiMax+1], pAtomExcs[:tiMax+1], color="C1", linewidth=3)

# Final atomic pop (theoretical)
if np.sum(shmVec) % 2 == 1:
    intRadsq = 0
else:
    intRadsq = 2 * min(nRad1,nRad2) * min(mRad1,mRad2) * (-1.)**(n1+n2+m1+m2+shmDist)

effRadsq = nRad1*mRad1 + nRad2*mRad2 + intRadsq
funS = g**2 * effRadsq
Csq_infty = 1./(1. + funS)**2
print('Theoretical steady-state population:')
print(Csq_infty)
axs[0,0].plot([ts[0],ts[tiMax]], [Csq_infty,Csq_infty], '--k', label=r'$|C_e(\infty)|^2$')

#def decayFun(t):
#    alpha = 5*nCpts * g**2
#    h = 0.9738
#    return (h - Csq_infty) * np.exp(-2*alpha*t) + Csq_infty
#axs[0,0].plot(ts[:tiMax+1], decayFun(ts[:tiMax+1]), '-.k', label=r'Decay')

axs[0,0].set(xlabel='$tJ$', ylabel='$|C_e(t)|^2$')
axs[0,0].legend()
axs[0,0].set_title(r"$\bf{(a)}$"+' Atomic population')

# Plot the state of the bath at 3 time steps

# Maximum bath excitation for consistent scaling
maxBathExc = np.max(abs(stateEvol[:,1:])**2)

#padding around the atom for plotting
pad = 0

tJ = 1
tiPlot = int(tJ/dt)
pBathExc = np.reshape(stateEvol[tiPlot,1:].real**2 + stateEvol[tiPlot,1:].imag**2,(N,N))
plotSize = [N//2-((nRad+1)//2+(mRad-1)//2+pad+1), N//2+((nRad-1)//2+(mRad+1)//2+pad), N//2-((nRad-1)//2+(mRad+1)//2+pad+1), N//2+((nRad+1)//2+(mRad-1)//2+pad)]
axs[0,1].imshow(pBathExc[plotSize[0]:plotSize[1],plotSize[2]:plotSize[3]], origin="lower", extent=[xyVal + 0.5 for xyVal in plotSize], norm=colors.LogNorm(vmin=maxBathExc/150, vmax=maxBathExc), aspect='auto', cmap='Greens')
#First subset of coupling points
axs[0,1].plot(x[:int(nCpts/2+1)]+1, y[:int(nCpts/2+1)]+1, marker='o', markersize=15, ls='-', lw=5, color='C1')
#Second subset of coupling points
axs[0,1].plot(x[int(nCpts/2+1):]+1, y[int(nCpts/2+1):]+1, marker='o', markersize=15, ls='-', lw=5, color='C1')
axs[0,1].set(xlabel='Resonator index, $n_x$', ylabel='Resonator index, $n_y$')
axs[0,1].set_title(r"$\bf{(b)}$" + f' Bath population at $tJ = {tJ}$', loc="right")

tJ = 5
tiPlot = int(tJ/dt)
pad = 7
pBathExc = np.reshape(stateEvol[tiPlot,1:].real**2 + stateEvol[tiPlot,1:].imag**2,(N,N))
plotSize = [N//2-((nRad+1)//2+(mRad-1)//2+pad+1), N//2+((nRad-1)//2+(mRad+1)//2+pad), N//2-((nRad-1)//2+(mRad+1)//2+pad+1), N//2+((nRad+1)//2+(mRad-1)//2+pad)]
axs[1,0].imshow(pBathExc[plotSize[0]:plotSize[1],plotSize[2]:plotSize[3]], origin="lower", extent=[xyVal + 0.5 for xyVal in plotSize], norm=colors.LogNorm(vmin=maxBathExc/150, vmax=maxBathExc), aspect='auto', cmap='Greens')
#First subset of coupling points
axs[1,0].plot(x[:int(nCpts/2+1)]+1, y[:int(nCpts/2+1)]+1, marker='o', markersize=7, ls='-', lw=2, color='C1')
#Second subset of coupling points
axs[1,0].plot(x[int(nCpts/2+1):]+1, y[int(nCpts/2+1):]+1, marker='o', markersize=7, ls='-', lw=2, color='C1')
axs[1,0].set(xlabel='Resonator index, $n_x$', ylabel='Resonator index, $n_y$')
axs[1,0].set_title(r"$\bf{(c)}$" + f' Bath population at $tJ = {tJ}$', loc="right")

tJ = 80
tiPlot = int(tJ/dt)
pad = 0
pBathExc = np.reshape(stateEvol[tiPlot,1:].real**2 + stateEvol[tiPlot,1:].imag**2,(N,N))
plotSize = [N//2-((nRad+1)//2+(mRad-1)//2+pad+1), N//2+((nRad-1)//2+(mRad+1)//2+pad), N//2-((nRad-1)//2+(mRad+1)//2+pad+1), N//2+((nRad+1)//2+(mRad-1)//2+pad)]
bathFinal = axs[1,1].imshow(pBathExc[plotSize[0]:plotSize[1],plotSize[2]:plotSize[3]], origin="lower", extent=[xyVal + 0.5 for xyVal in plotSize], norm=colors.LogNorm(vmin=maxBathExc/150, vmax=maxBathExc), aspect='auto', cmap='Greens')
#First subset of coupling points
axs[1,1].plot(x[:int(nCpts/2+1)]+1, y[:int(nCpts/2+1)]+1, marker='o', markersize=15, ls='-', lw=5, color='C1')
#Second subset of coupling points
axs[1,1].plot(x[int(nCpts/2+1):]+1, y[int(nCpts/2+1):]+1, marker='o', markersize=15, ls='-', lw=5, color='C1')
axs[1,1].set(xlabel='Resonator index, $n_x$', ylabel='Resonator index, $n_y$')
axs[1,1].set_title(r"$\bf{(d)}$" + f' Bath population at $tJ = {tJ}$', loc="right")

print('Final population in simulation:')
print(pAtomExcs[tiMax])
print('Peaks of BIC standing wave at the end:')
print(pBathExc[N//2,N//2])
print(pBathExc[N//2-1,N//2-1])
print('NB: The locations for these are in the centre of the grid and [-1,-1] away from this point.')

fig.tight_layout()
fig.subplots_adjust(right=0.85, top=0.87, wspace=0.4, hspace=0.5)
cbar_ax = fig.add_axes([0.1, -0.05, 0.75, 0.03]) #XY displacement, XY extent
cbar = fig.colorbar(bathFinal, cax=cbar_ax, orientation="horizontal", ticks=ticker.LogLocator(base=10, subs=(0.5, )), format='%.1e')
cbar.set_label(label="Bath population")

# fig.suptitle(r'Time evolution of a GA with $\Delta/J = 0$ and $g/J = %.3f$' %g) # {Delta}; and coupling points at {couplingPts[0]+1}, {couplingPts[1]+1}, {couplingPts[2]+1} and {couplingPts[3]+1}

# fig.savefig('/Users/ari/Dropbox/My Mac (Ariadnas-MacBook-Pro.local)/Documents/Chalmers/PhD thesis/Projects/2023 2D Structured/1-GA-8cPts_constructive.pdf', dpi=fig.dpi, bbox_inches='tight')
# fig.savefig('/Users/ari/Dropbox/My Mac (Ariadnas-MacBook-Pro.local)/Documents/Chalmers/PhD thesis/Projects/2023 2D Structured/1-GA-8cPts_destructive.pdf', dpi=fig.dpi, bbox_inches='tight')
fig.savefig('/Users/ari/Dropbox/My Mac (Ariadnas-MacBook-Pro.local)/Documents/Chalmers/PhD thesis/Projects/2023 2D Structured/1-GA-8cPts_no_interference.pdf', dpi=fig.dpi, bbox_inches='tight')