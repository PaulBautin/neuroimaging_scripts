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


####### QUESTION 4
# Define AM and FM functions
def adiabatic_sech(t, beta, n):
    """Amplitude modulation (AM) using a hyperbolic secant function."""
    AM = 1 / np.cosh(beta * ((t - 4E-3) ** n))
    FM = integrate.cumulative_simpson(AM ** 2, x=t, initial=0)
    FM = beta * np.tanh(beta * (t - 4E-3))
    #FM = 2000 * FM / np.max(FM)
    return AM, FM

# Values of n
n_values = [1, 2, 4, 8]
t = np.linspace(0, 8E-3, 1000)
beta = 1000  # Modulation parameter

# Plot AM and FM for different values of n
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

for n in n_values:
    am_sech, fm_sech = adiabatic_sech(t, beta, n)
    axes[0].plot(t * 1e3, am_sech, label=f"n={n}")  # Convert time to ms
    axes[1].plot(t * 1e3, fm_sech, label=f"n={n}")

# Formatting the plots
axes[0].set_ylabel("Amplitude Modulation")
axes[0].legend()
axes[0].set_title("Amplitude Modulation (AM) of Adiabatic HS Pulses")

axes[1].set_ylabel("Frequency Modulation (Hz)")
axes[1].set_xlabel("Time (ms)")
axes[1].legend()
axes[1].set_title("Frequency Modulation (FM) of Adiabatic HS Pulses")

plt.tight_layout()
plt.show()


