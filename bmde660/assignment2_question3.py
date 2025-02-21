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

def rotate_z(vector: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotates a 3D vector around the Z-axis by an angle theta (in radians).

    Parameters:
    - vector (np.ndarray): Input 3D vector as a NumPy array [x, y, z].
    - theta (float): Rotation angle in radians.

    Returns:
    - np.ndarray: Rotated 3D vector.
    """
    # Define the rotation matrix for Rz(theta)
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    
    # Apply the rotation matrix to the vector
    return Rz @ vector


def rotate_y(vector: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotates a 3D vector around the Y-axis by an angle theta (in radians).

    Parameters:
    - vector (np.ndarray): Input 3D vector as a NumPy array [x, y, z].
    - theta (float): Rotation angle in radians.

    Returns:
    - np.ndarray: Rotated 3D vector.
    """
    # Define the rotation matrix for Ry(theta)
    Ry = np.array([
        [np.cos(theta),  0, -np.sin(theta)],
        [0,              1, 0            ],
        [np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Apply the rotation matrix to the vector
    return Ry @ vector


def rotate_x(vector: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotates a 3D vector around the X-axis by an angle theta (in radians).

    Parameters:
    - vector (np.ndarray): Input 3D vector as a NumPy array [x, y, z].
    - theta (float): Rotation angle in radians.

    Returns:
    - np.ndarray: Rotated 3D vector.
    """
    # Define the rotation matrix for Rx(theta)
    Rx = np.array([
        [1, 0,              0             ],
        [0, np.cos(theta), np.sin(theta)],
        [0, -np.sin(theta),  np.cos(theta)]
    ])
    
    # Apply the rotation matrix to the vector
    return Rx @ vector


params = {'font.size': 22}
plt.rcParams.update(params)

########### Load the .mat file
df = pd.DataFrame(np.array(h5py.File("/local_raid/data/pbautin/downloads/RFpulse.mat")['RFpulse']).T, columns=["phase", "amplitude"])
df['t'] = np.linspace(0, 6E-3, df.shape[0]) 
df['B1'] =  df['amplitude'] * np.exp(df['phase'] * ((np.pi)/180) * 1j)


########### PLOT 
# Create a figure with two subplots
fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
# Plot phase
ax[0].plot(df['t'], df['phase'], linestyle="-", label="Phase")
ax[0].set_ylabel("Phase (degrees)")
ax[0].set_title("RF Pulse Phase and Amplitude waveforms ")
ax[0].grid(True)
ax[0].legend()
# Plot real and imaginary parts of amplitude
ax[1].plot(df['t'], np.real(df['amplitude']), linestyle="-", label="Amplitude")
#ax[1].plot(df['t'], np.imag(df['amplitude_exp']), linestyle="--", label="Imag(Amplitude)")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Amplitude")
ax[1].grid(True)
ax[1].legend()
plt.tight_layout()
plt.show()

########### FIND B1 MAX
integral_B1 = np.trapz(np.real(df['B1']), df['t'])
B1_max = (np.pi) / (26.7522E7 * integral_B1)
B1_max_microtesla = B1_max * 1E6
# Normalize B1 based on max value
df['B1'] *= B1_max
df['amplitude'] *= B1_max


########### BLOCH SIMULATION
# Choose offset frequency range
offset_frequency = np.linspace(-4000, 4000, 1000)  
# Define time step
time_step = 6E-3 / df.shape[0]

M_freq = []  # Store magnetization response for each frequency
for n in offset_frequency:
    # Initialize magnetization vector (aligned with z-axis)
    M = np.array([0, 0, 1], dtype=np.float64)
    for i in range(df.shape[0]):
        B1_i = df['amplitude'].values[i]
        B1_eff_i = np.linalg.norm([df['B1'].values[i], (n / 26.7522E7)])

        phi = df['phase'].values[i] * (np.pi / 180)  # Convert phase to radians
        theta = 26.7522E7 * B1_eff_i * time_step  # Flip angle per time step
        alpha = np.arctan(B1_i / (n / 26.7522E7))
        
        # Apply composite rotation: Beff in rotating frame
        M = rotate_z(rotate_y(rotate_z(rotate_y(rotate_z(M, phi), alpha), theta), -alpha), -phi)

    # Compute longitudinal magnetization
    M_freq.append(M[2])

# Plot the frequency response
plt.plot(offset_frequency/1000, M_freq)
plt.xlabel("Offset Frequency (kHz)")
plt.ylabel("inversion profile (Mz)")
plt.title("Bloch Simulation Frequency Response")
plt.grid()
plt.show()


