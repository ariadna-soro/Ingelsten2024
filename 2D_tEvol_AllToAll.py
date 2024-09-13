import numpy as np
from numpy import matmul as mm
from numpy.lib.scimath import sqrt as csqrt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib import colors
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

# Structured bath size (#resonators along one side - total size is NxN)
N = 250
# Number of atoms
nAtoms = 3
# Energy units where J=1 throughout
# Detuning of atoms from bath
Delta = 0.
# The different GAs are displaced by these amounts [dx,dy] from the centre of the grid:
offsets = np.array([[0, 0], [0, 0], [0, 0]])
if (len(offsets) != nAtoms):
    sys.exit(f'Offset list has too many or too few entries (length = {len(offsets)}): Should be equal to nAtoms = {nAtoms}')
# Number of coupling points per atom
nCpts = 8
# For simplicity, I assume that all the atoms have the same number of coupling points.
# It would be pretty straightforward to generalise to the case where each atom has a different nCpts, though
# Coupling point spacing (“radii”) along diagonals for the different atoms,
# (mRad,nRad) = 2*(m,n)+1 w/ integer m,n for perfect subradiance at Delta = 0
nRad1 = 2*0+1
mRad1 = 2*0+1
nRad2 = 2*0+1
mRad2 = 2*0+1
# Positions of the coupling points of each atom relative to the “centre” of the atom
cPtsRel = np.array([[[-1, 1+0],
                      [nRad2-1, 1+nRad2],
                      [nRad2+mRad2-1, 1+nRad2-mRad2],
                      [mRad2-1, 1-mRad2],
                      [-1-1, -1+0],
                      [-1+nRad2-1, -1+nRad2],
                      [-1+nRad2+mRad2-1, -1+nRad2-mRad2],
                      [-1+mRad2-1, -1-mRad2]],
                    [[1-1, 1+0],
                      [1+nRad2-1, 1+nRad2],
                      [1+nRad2+mRad2-1, 1+nRad2-mRad2],
                      [1+mRad2-1, 1-mRad2],
                      [2-1, -1+0],
                      [2+nRad2-1, -1+nRad2],
                      [2+nRad2+mRad2-1, -1+nRad2-mRad2],
                      [2+mRad2-1, -1-mRad2]],
                    [[-1-1, -2+0],
                      [-1+nRad2-1, -2+nRad2],
                      [-1+nRad2+mRad2-1, -2+nRad2-mRad2],
                      [-1+mRad2-1, -2-mRad2],
                      [2-1, -2+0],
                      [2+nRad2-1, -2+nRad2],
                      [2+nRad2+mRad2-1, -2+nRad2-mRad2],
                      [2+mRad2-1, -2-mRad2]]])



if (len(cPtsRel) != nAtoms):
    sys.exit(f'Too many or too few coupling point lists (#lists = {len(cPtsRel)}): Should be equal to nAtoms = {nAtoms}')
for a in range(nAtoms):
    if (len(cPtsRel[a,:,:]) != nCpts):
        sys.exit(f'Too many or too few coupling points for atom {a} (list length = {len(cPtsRel[a,:,:])}): Should be equal to nCpts = {nCpts}')

# Actual position of all coupling points in the lattice
couplingPts = np.array([[[N//2 - 1, N//2 - 1]]*nCpts]*nAtoms) + np.array([cPtsRel[a] + os for a,os in enumerate(offsets)]) # Position of coupling points for all GAs
print(couplingPts)

# maximum radii used for plotting later (should probably also use offsets in some way ideally, but this is not implemented)
nRad = max(nRad1,nRad2)*2
mRad = max(mRad1,mRad2)*2

# “Effective coupling strength” for the atoms
G = 0.5
# Actual bath-atom coupling strength
g = G / np.sqrt(nCpts)

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
stateEvol = np.zeros((nSteps+1,N*N+nAtoms), dtype='complex64')

# Initial state: GA 1 excited
stateEvol[0,0] = 1+0.j
#stateEvol[0,6] = 1/np.sqrt(2.)+0.j

# Effective hamiltonian corresponding to H_A + H_int & corresponding effective time evolution operator
ham_AI_eff = np.zeros((2,2), dtype='complex64')
ham_AI_eff[0,0] = Delta
ham_AI_eff[0,1] = G
ham_AI_eff[1,0] = G
tEvOp_eff = spla.expm(-1.j * ham_AI_eff * dt)
print(tEvOp_eff)


# Time evolution operator for bath in k-space
tEvOp_kspace = np.exp(-1.j * ws * dt)

for ti in range(nSteps):
    # Evolve using H_A + H_int in real space – the following is
    # equivalent to applying the full real-space time evolution operator for H_A + H_int
    stateEvol[ti+1,:] = stateEvol[ti,:]
    for a in range(nAtoms):
        intState = 0
        cpLocs = [couplingPts[a,p,0]+couplingPts[a,p,1]*N for p in range(nCpts)]
        for cpLoc in cpLocs:
            intState += stateEvol[ti,nAtoms+cpLoc]

        stateEvol[ti+1,a] = tEvOp_eff[0,0] * stateEvol[ti,a] + tEvOp_eff[0,1]/np.sqrt(nCpts) * intState
        for cpLoc in cpLocs:
            stateEvol[ti+1,nAtoms+cpLoc] += tEvOp_eff[0,1]/np.sqrt(nCpts) * stateEvol[ti,a] + (tEvOp_eff[1,1] - 1.)/nCpts * intState

    # Force unitarity to limit numerical errors
    stateEvol[ti+1,:] /= np.sqrt(np.sum(stateEvol[ti+1,:].real**2 + stateEvol[ti+1,:].imag**2))

    # Go into k-space
    psi_bath_k = np.fft.fft2(np.reshape(stateEvol[ti+1,nAtoms:],(N,N)))

    # Apply exact time evolution operator for H_B = diag(ws)
    psi_bath_k = tEvOp_kspace * psi_bath_k

    # Go back into real space
    stateEvol[ti+1,nAtoms:] = np.fft.ifft2(psi_bath_k).flatten()

##############################################################################
# Initialise plotting
fig,axs = plt.subplots(2,2)

# Times for each time step
ts = np.linspace(0., dt*nSteps, nSteps+1)
# Excited population over time in atoms
pAtomExcs = stateEvol[:,:nAtoms].real**2 + stateEvol[:,:nAtoms].imag**2
# Limit time plotted to avoid wraparound effects
tiMax = (7*nSteps)//8


# Population transfers (As-is only cares about atoms 1 and 2)
# Care about no more than this many population transfers:
nPopTfs = 15

# Times when the atoms are maximally excited.
# To get first value: When is atom 2 maximally populated?
tiMaxPopTfs = np.zeros(nPopTfs, dtype=int)
tiMaxPopTfs[0] = np.argmax(pAtomExcs[:,1])

# Times when the atoms are minimally excited.
# To get first value: What is the first minimum in the population for atom 1?
# (This should be almost simultaneous to atom 2 reaching its peak population,
# but the minima are more reliable data points for calculating the interaction
# speed since there is less noise from transient oscillations related to populating the BIC)
tiMinPopTfs = np.zeros(nPopTfs, dtype=int)
tiMinPopTfs[0] = np.argmin(pAtomExcs[:2*tiMaxPopTfs[0],0])

# Calculate further times of max/min pop transfer
index = 1
atomExc = 0
atomEmpty = 1
while (2*index+1)*tiMaxPopTfs[0]//2 < tiMax:
    if index >= nPopTfs:
        break
    tiMaxPopTfs[index] = (2*index+1)*tiMaxPopTfs[0]//2 + np.argmax(pAtomExcs[(2*index+1)*tiMaxPopTfs[0]//2:,atomExc])
    tiMinPopTfs[index] = (2*index+1)*tiMinPopTfs[0]//2 + np.argmin(pAtomExcs[(2*index+1)*tiMinPopTfs[0]//2:,atomEmpty])
    index += 1
    atomExc = (atomExc + 1) % 2
    atomEmpty = (atomEmpty + 1) % 2

tiMaxPopTfs = np.trim_zeros(tiMaxPopTfs)
tiMinPopTfs = np.trim_zeros(tiMinPopTfs)
tVals_MaxPopTf = dt * tiMaxPopTfs
tVals_MinPopTf = dt * tiMinPopTfs
#print(pAtomExcs[tiMaxPopTfs,1])
maxPops = np.maximum(pAtomExcs[tiMaxPopTfs,0],pAtomExcs[tiMaxPopTfs,1])
print('Maximum population transfer:')
print(tVals_MaxPopTf)
print(maxPops)

# Period of population oscillations
period = 2*np.mean(np.diff(tVals_MinPopTf))
z_R = np.pi/period
print('Period and z_R:')
print(period)
print(z_R)

#NB: The following decay models are only really designed to work for the 2GA case with both atoms identical

#Linear regression to find best-fit exponential decay
model = np.polyfit(tVals_MaxPopTf, -np.log(maxPops), 1)
tInit = 0.
initVal = np.exp(-model[1])
z_I = 0.5 * model[0]
print('Initial decay magnitude:')
print(initVal)

# “More accurate” hack for decay:
# (The accuracy of this is situational)
tiInit = int(max(nRad,mRad)/dt)+np.argmin(pAtomExcs[int(max(nRad,mRad)/dt):int(1.5*max(nRad,mRad)/dt),0])
tInit = dt * tiInit
initVal = pAtomExcs[tiInit,0] / np.cos(z_R * tInit)**2
z_I = 0.5 * np.log(initVal/maxPops[-1]) / (tVals_MaxPopTf[-1] - tInit)

print('“Initial time” for decay, and magnitude at this time:')
print(tInit)
print(initVal)
print('Value of z_I (= decay rate / 2):')
print(z_I)
print('------')


# Actual plotting ############################################################
lines = ["-", "-", ":"]
for a in range(nAtoms):
    axs[0,0].plot(ts[:tiMax+1], pAtomExcs[:tiMax+1,a], linewidth=4, linestyle=lines[a], label=f'Atom {a+1}')
axs[0,0].set(xlabel='$tJ$', ylabel='$|C_e(t)|^2$')
axs[0,0].set_title(r"$\bf{(a)}$" + ' Atomic population')

tIntf = max(nRad1,mRad1)

fig.delaxes(axs[0,1])


# Maximum excitation in the bath for consistent colouring
maxBathExc = np.max(abs(stateEvol[:,nAtoms:])**2)

# Margin around the atoms when plotting
margin = 2

# Plot the state of the bath at 2 time steps
colorList = ['C0', 'C1', 'C2', 'C3', 'C4']

tJ = 1
tiPlot = int(tJ/dt)
pBathExc = np.reshape(stateEvol[tiPlot,nAtoms:].real**2 + stateEvol[tiPlot,nAtoms:].imag**2,(N,N))
plotSize = [N//2-((nRad+1)//2+(mRad-1)//2+margin+1), N//2+((nRad-1)//2+(mRad+1)//2+margin), N//2-((nRad-1)//2+(mRad+1)//2+margin+1), N//2+((nRad+1)//2+(mRad-1)//2+margin)]
axs[1,0].imshow(pBathExc[plotSize[0]:plotSize[1],plotSize[2]:plotSize[3]], origin="lower", extent=[xyVal + 0.5 for xyVal in plotSize], norm=colors.LogNorm(vmin=maxBathExc/150, vmax=maxBathExc), aspect='auto', cmap="Greens")
axs[1,0].set(xlabel='Resonator index, $n_x$', ylabel='Resonator index, $n_y$')
axs[1,0].set_title(r"$\bf{(b)}$" + f' Bath population at $tJ = {tJ}$', loc="right")

tJ = 75
tiPlot = int(tJ/dt) #tiMaxPopTfs[0]
pBathExc = np.reshape(stateEvol[tiPlot,nAtoms:].real**2 + stateEvol[tiPlot,nAtoms:].imag**2,(N,N))
# plotSize = [N//2-((nRad+1)//2+(mRad-1)//2+margin), N//2+((nRad-1)//2+(mRad+1)//2+margin-1), N//2-((nRad-1)//2+(mRad+1)//2+margin), N//2+((nRad+1)//2+(mRad-1)//2+margin-1)]
bath1stPopTf = axs[1,1].imshow(pBathExc[plotSize[0]:plotSize[1],plotSize[2]:plotSize[3]], origin="lower", extent=[xyVal + 0.5 for xyVal in plotSize], norm=colors.LogNorm(vmin=maxBathExc/150, vmax=maxBathExc), aspect='auto', cmap="Greens")
for a in range(nAtoms):
    x, y = couplingPts[a].T
    #ALL-TO-ALL
    x = np.insert(x, int(nCpts/2), x[0])
    x = np.insert(x, int(nCpts+1), x[int(nCpts/2+1)])
    y = np.insert(y, int(nCpts/2), y[0])
    y = np.insert(y, int(nCpts+1), y[int(nCpts/2+1)])
    #First subset of coupling points
    axs[1,0].plot(x[:int(nCpts/2+1)]+1, y[:int(nCpts/2+1)]+1, marker='o', markersize=17, ls='-', lw=6, color="w")
    axs[1,0].plot(x[:int(nCpts/2+1)]+1, y[:int(nCpts/2+1)]+1, marker='o', markersize=15, ls='-', lw=4, color=colorList[a], label=f"Atom {a+1}")
    #Second subset of coupling points
    axs[1,0].plot(x[int(nCpts/2+1):]+1, y[int(nCpts/2+1):]+1, marker='o', markersize=17, ls='-', lw=6, color="w")
    axs[1,0].plot(x[int(nCpts/2+1):]+1, y[int(nCpts/2+1):]+1, marker='o', markersize=15, ls='-', lw=4, color=colorList[a])
    #First subset of coupling points
    axs[1,1].plot(x[:int(nCpts/2+1)]+1, y[:int(nCpts/2+1)]+1, marker='o', markersize=17, ls='-', lw=6, color="w")
    axs[1,1].plot(x[:int(nCpts/2+1)]+1, y[:int(nCpts/2+1)]+1, marker='o', markersize=15, ls='-', lw=4, color=colorList[a], label=f"Atom {a+1}")
    #Second subset of coupling points
    axs[1,1].plot(x[int(nCpts/2+1):]+1, y[int(nCpts/2+1):]+1, marker='o', markersize=17, ls='-', lw=6, color="w")
    axs[1,1].plot(x[int(nCpts/2+1):]+1, y[int(nCpts/2+1):]+1, marker='o', markersize=15, ls='-', lw=4, color=colorList[a])
    
axs[1,1].legend(bbox_to_anchor=[0.5, 2], loc="center")
axs[1,1].set(xlabel='Resonator index, $n_x$', ylabel='Resonator index, $n_y$')
axs[1,1].set_title(r"$\bf{(c)}$" + f' Bath population at $tJ = {tJ}$', loc="right")


print('Peaks of BIC standing wave at the end:')
print(pBathExc[N//2,N//2])
print(pBathExc[N//2-1,N//2-1])
print('NB: The locations for these are in the centre of the grid and [-1,-1] away from this point.')

fig.tight_layout()
fig.subplots_adjust(right=0.85, top=0.87, wspace=0.4, hspace=0.5)
cbar_ax = fig.add_axes([0.1, -0.05, 0.75, 0.03]) #XY displacement, XY extent
cbar = fig.colorbar(bath1stPopTf, cax=cbar_ax, orientation="horizontal", ticks=ticker.LogLocator(base=10, subs=(0.5, )), format='%.1e')
cbar.set_label(label="Bath population")

plt.show()

# Modify path name and uncomment to save figure:
# fig.savefig('YourPath/3-GA-AllToAll.pdf', dpi=fig.dpi, bbox_inches='tight')
