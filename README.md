# Avoiding decoherence with giant atoms in a two-dimensional structured environment

This repository contains all the Python scripts to reproduce the results from [arXiv:2402.10879](https://arxiv.org/abs/2402.10879).
All code was written by Emil Raaholt Ingelsten and Ariadna Soro during 2023-2024.

To reproduce Figure 3 of the paper, use the script [2D_tEvol_1GA_4pts](2D_tEvol_1GA_4pts.py). This simulates the dynamics of a single giant atom with 4 coupling points.

To reproduce Figures 4-6 of the paper, use the script [2D_tEvol_1GA_8pts](2D_tEvol_1GA_8pts.py). This simulates the dynamics of a single giant atom with 8 coupling points. (Un)comment the indicated parts to achieve constructive interference, destructive interference or no interference between the photonic parts of the bound states.

To reproduce Figure 9 of the paper, use the script [2D_tEvol_DFI](2D_tEvol_DFI.py). This simulates the dynamics of two giant atoms with 4 coupling points each, interacting in a decoherence-free manner.

To reproduce Figure 11 of the paper, use the script [2D_tEvol_AllToAll](2D_tEvol_AllToAll.py). This simulates the dynamics of three giant atoms with 8 coupling points each, interacting in a decoherence-free manner.
