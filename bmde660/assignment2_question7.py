from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Effective connectivity of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import scipy.io
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

params = {'font.size': 22}
plt.rcParams.update(params)


####### QUESTION 7
# Compute Mz for 10 TR
# Parameters
M0 = 1  # Equilibrium magnetization
TI = 0.8  # Inversion time in seconds
TR = 2  # Repetition time in seconds
alpha = 9 * (np.pi / 180)  # Flip angle in radians
T1 = 900E-3  # T1 relaxation time in seconds
repetitions = 10  # Number of TR repetitions
Tes = 8E-3

# Initialize magnetization array
Mz = np.zeros((repetitions + 1, 88))
Mz[0, :] = M0  # Start with equilibrium magnetization

# Loop over TR cycles
for tr in range(repetitions):
    for i in range(88):
        Mz_next = (np.cos(alpha) * ((Mz[tr-1, i] * np.cos(np.pi) - M0) * np.exp(-(TI + i * Tes) / T1) + M0) - M0) * np.exp(-(TR - (TI + i * Tes)) / T1) + M0
        Mz[tr+1, i] = Mz_next
print(Mz)

print(Mz)
# Time axis for plotting
time = np.arange(repetitions + 1)

# Plotting
plt.plot(time, Mz[:,0], marker='o', linestyle='-')
plt.xlabel("Repetition Number (TR)")
plt.ylabel("Mz Magnetization")
plt.title("Evolution of the longitudinal magnetization Mz")
plt.grid()
plt.show()


####### question b
# plot the modulation transfer function W(k) and absolute point
# spread function in the phase encode
# Define k-space sampling grid
k_y = np.linspace(-88, 88, 176)

# steady-state magnetization values
Mz_TE = Mz[4, :]

# --- Sequential Phase Encoding ---
k_space_seq = np.zeros(176)
k_space_seq[:88] = Mz_TE  # First half of k-space
k_space_seq[88:] = Mz_TE  # Second half of k-space

# Compute PSF
psf_seq = np.abs(np.fft.fftshift(np.fft.fft(k_space_seq)))

# --- Interleaved Phase Encoding ---
k_space_int = np.zeros(176)
k_space_int[::2] = Mz_TE  # Every other line
k_space_int[1::2] = Mz_TE  # Remaining lines

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

    
