from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Assignement 2 question 4
#
# ---------------------------------------------------------------------------------------
# Author: Paul Bautin with help from ChatGPT 3
#
#########################################################################################


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

params = {'font.size': 22}
plt.rcParams.update(params)


####### QUESTION 4
# Define AM and FM functions
def adiabatic_sech(t, beta, n, w_max):
    """Amplitude modulation (AM) using a hyperbolic secant function."""
    # AM with normalized time [-1, 1]
    AM = 1 / np.cosh(beta * (((t - 4E-3)/4E-3) ** n))
    FM = 2000 * (integrate.cumulative_simpson(AM ** 2, x=t, initial=0) / np.max(integrate.cumulative_simpson(AM ** 2, x=t, initial=0)) - 0.5)
    #FM = beta * np.tanh(beta * (t - 4E-3))
    #FM = 2000 * FM / np.max(FM)
    return AM, FM

# Values of n
n_values = [1, 2, 4, 8]
t = np.linspace(0, 8E-3, 1000)
beta = 5.29829 # Modulation parameter based on article doi:10.1006/jmre.2001.2340

# Plot AM and FM for different values of n
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

for n in n_values:
    am_sech, fm_sech = adiabatic_sech(t, beta, n, w_max=1000)
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


