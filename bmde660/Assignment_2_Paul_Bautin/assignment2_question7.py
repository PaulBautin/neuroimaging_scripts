from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Assignement 2 question 7
#
# ---------------------------------------------------------------------------------------
# Author: Paul Bautin with help from ChatGPT 3
#
#########################################################################################


import scipy.io
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

params = {'font.size': 22}
plt.rcParams.update(params)


####### QUESTION 7a
# Parameters
M0 = 1  # Equilibrium magnetization
TI = 0.8  # Inversion time in seconds
TR = 2  # Repetition time in seconds
alpha = np.radians(9)  # Flip angle in radians
T1 = 0.9  # T1 relaxation time in seconds
nb_rep = 10  # Number of TR repetitions
Tes = 0.008  # Echo time in seconds
nb_echo = 88

# Initialize magnetization array
Mz = np.zeros((nb_rep + 1, nb_echo + 2))
Mz[0, :] = M0  # Start with equilibrium magnetization
Mxy = []

# Compute Mz for each TR cycle
for tr in range(1, nb_rep + 1):
    Mz[tr, 0] = Mz[tr - 1, -1]  # Transfer last Mz value
    Mz[tr, 0] = (Mz[tr, 0] * np.cos(np.pi) - M0) * np.exp(-TI / T1) + M0 # Apply Inversion recovery
    for i in range(1, nb_echo + 1):
        Mz[tr, i] = (np.cos(alpha) * Mz[tr, i - 1] - M0) * np.exp(-Tes / T1) + M0 
        if tr==4:
            Mxy.append(Mz[tr, i] * np.tan(alpha))
    Mz[tr, -1] = (Mz[tr, i] - M0) * np.exp(-(TR - TI - (88 * Tes)) / T1) + M0

# Plotting
print(Mz[:, -1])
plt.plot(range(0, nb_rep + 1), Mz[:, -1], marker='o', linestyle='-')
plt.xlabel("Repetition Number (TR)")
plt.ylabel("Mz Magnetization")
plt.title("Evolution of the Longitudinal Magnetization Mz")
plt.grid()
plt.show()


####### QUESTION 7b
# Define k-space sampling grid
k_y = np.linspace(-88, 88, 176)

# steady-state magnetization values
Mz_TE = Mz[4, 1:-1]

# --- Sequential Phase Encoding ---
k_space_seq = np.zeros(176)
k_space_seq[:88] = Mxy / Mxy[0]  # First half of k-space
k_space_seq[88:] = Mxy  / Mxy[0]  # Second half of k-space

# Compute PSF
psf_seq = np.abs(np.fft.fftshift(np.fft.fft(k_space_seq)))

# --- Interleaved Phase Encoding ---
k_space_int = np.zeros(176)
k_space_int[::2] = Mxy  / Mxy[0] # Every other line
k_space_int[1::2] = Mxy  / Mxy[0] # Remaining lines

# Compute PSF
psf_int = np.abs(np.fft.fftshift(np.fft.fft(k_space_int)))

# --- Plot Modulation Function (W(k)) ---
plt.figure(figsize=(10, 5))
plt.plot(k_y, k_space_seq, label="Sequential", linestyle='-', marker='o')
plt.plot(k_y, k_space_int, label="Interleaved", linestyle='--', marker='x')
plt.xlabel(r'$k_y$ (Phase Encode Direction)')
plt.ylabel(r'$W(k)$')
plt.title('modulation transfer function W(k)')
plt.legend()
plt.grid(True)

# --- Plot Point Spread Function (PSF) ---
plt.figure(figsize=(10, 5))
plt.plot(k_y, psf_seq, label="Sequential", linestyle='-', marker='o')
plt.plot(k_y, psf_int, label="Interleaved", linestyle='--', marker='x')
plt.xlabel(r'$y$ (Phase Encode Direction)')
plt.ylabel(r'Absolute PSF')
plt.title('absolute point spread function')
plt.legend()
plt.grid(True)

# Show plots
plt.show()

    
