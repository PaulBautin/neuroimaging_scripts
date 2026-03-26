"""Neural field model simulation and resting-state fMRI analysis."""

from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import mesh_elements, mesh_operations
from brainspace.utils.parcellation import reduce_by_labels, map_to_labels
from lapy import TriaMesh, Solver
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh, splu
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
from brainspace.gradient.gradient import GradientMaps


def model_balloon_fourier(mode_coeff, dt):
    # Default independent model parameters
    kappa = 0.65   # signal decay rate [s^-1]
    gamma = 0.41   # rate of elimination [s^-1]
    tau = 0.98     # hemodynamic transit time [s]
    alpha = 0.32   # Grubb's exponent [unitless]
    rho = 0.34     # resting oxygen extraction fraction [unitless]
    V0 = 0.02      # resting blood volume fraction [unitless]

    # Other parameters
    w_f = 0.56
    Q0 = 1
    rho_f = 1000
    eta = 0.3
    Xi_0 = 1
    beta = 3
    V_0 = 0.02
    k1 = 3.72
    k2 = 0.527
    k3 = 0.48
    beta = (rho + (1 - rho) * np.log(1 - rho)) / rho

    # --- Use the same causal Fourier procedure as model_wave_fourier ---
    # Zero-pad input for t < 0 (causality)
    nt = len(mode_coeff)
    t_full = np.arange(-nt * dt, nt * dt + dt, dt)  # Symmetric time vector
    nt_full = len(t_full)

    mode_coeff_padded = np.concatenate([np.zeros(nt-1), mode_coeff])
    # Frequencies for full signal
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2*nt-1, d=dt))
    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    mode_coeff_f = np.fft.fftshift(np.fft.ifft(mode_coeff_padded))

    # Calculate the frequency response of the system
    phi_hat_Fz = 1 / (-(omega + 1j * 0.5 * kappa) ** 2 + w_f ** 2)
    phi_hat_yF = V_0 * (alpha * (k2 + k3) * (1 - 1j * tau * omega) 
                                - (k1 + k2) * (alpha + beta - 1 
                                - 1j * tau * alpha * beta * omega)) / ((1 - 1j * tau * omega)
                                *(1 - 1j * tau * alpha * omega))
    phi_hat = phi_hat_yF * phi_hat_Fz

    # Apply frequency response
    out_fft = phi_hat * mode_coeff_f
    # Inverse transform: use fft (not ifft) to return to the time domain, matching above convention
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft)))
    # Return only the non-negative time part (t >= 0)
    return out_full[nt-1:]


def model_wave_fourier(mode_coeff, dt, r, gamma, eval):
    """
    Simulate the response of a damped wave equation using Fourier methods.

    Parameters:
    mode_coeff (ndarray): Coefficients of the input signal in mode space.
    dt (float): Time step size.
    r (float): Spatial coupling parameter.
    gamma (float): Damping coefficient.
    eval (float): Eigenvalue associated with the mode.

    Returns:
    ndarray: Simulated response of the system for t >= 0.
    """
    nt = len(mode_coeff)
    # Pad input with zeros on negative side to ensure causality (system is only driven for t >= 0)
    # This is required for the correct Green's function solution of the damped wave equation.
    mode_coeff_padded = np.concatenate([np.zeros(nt-1), mode_coeff])
    # Apply inverse Fourier transform to get frequency-domain representation of the causal signal.
    mode_coeff_f = np.fft.fftshift(np.fft.ifft(mode_coeff_padded))
    # Frequencies for full signal (from fourier of derivative)
    omega = 1j * 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(2*nt-1, d=dt))
    # Compute transfer function
    denom = omega**2 + 2 * gamma * omega  +  gamma**2 - gamma**2 * r**2 * eval

    transfer_function = gamma**2 / denom
    # Apply frequency response
    out_fft = transfer_function * mode_coeff_f
    # Inverse transform: use fft (not ifft) to return to the time domain, matching above convention
    out_full = np.real(np.fft.fft(np.fft.ifftshift(out_fft)))
    # Return only the non-negative time part (t >= 0)
    return out_full[nt-1:]


def solve_laplace_beltrami(surface, curvature_param=10):
    """
    Solve the Laplace-Beltrami eigenvalue problem on a cortical surface.

    Parameters:
    surface (tuple): A tuple containing vertices and faces of the cortical surface.
    curvature_param (int): Parameter for curvature computation. Default is 10.

    Returns:
    eigvals   (ndarray): Eigenvalues of the Laplace-Beltrami operator.
    eigvecs   (ndarray): Eigenvectors of the Laplace-Beltrami operator.
    stiffness (ndarray): Stiffness matrix of FEM model.
    mass      (ndarray): Mass matrix of FEM model.
    """
    tm = TriaMesh(v=mesh_elements.get_points(surface), t=mesh_elements.get_cells(surface))
    u1, u2, _, _ = tm.curvature_tria(curvature_param)
    hetero_tri = tm.map_vfunc_to_tfunc(np.ones(mesh_elements.get_points(surface).shape[0]))
    hetero_mat = np.tile(hetero_tri[:, np.newaxis], (1, 2))
    stiffness, mass = Solver._fem_tria_aniso(tm, u1, u2, hetero_mat)

     # Solve the eigenvalue problem
    sigma = -0.01
    lu = splu(stiffness - sigma * mass)
    op_inv = LinearOperator(matvec=lu.solve, shape=stiffness.shape,dtype=stiffness.dtype)
    eigvals, eigvecs = eigsh(stiffness, k=200, M=mass, sigma=sigma, OPinv=op_inv)
    return np.asarray(eigvals), np.asarray(eigvecs), stiffness, mass


def generate_synthetic_fmri(eigvals, eigvecs, mass, time_points=200, dt=0.1, model_bold=False):
    """
    Simulate neural activity or BOLD signals on the surface mesh using Neural Field Theory.
    
    Parameters:
    eigvals (ndarray): Eigenvalues of the Laplace-Beltrami operator.
    eigvecs (ndarray): Eigenvectors of the Laplace-Beltrami operator.
    mass (ndarray): Mass matrix of the FEM model.
    time_points (int): Number of time points to simulate. Default is 200.
    dt (float): Time step size for the simulation. Default is 0.1 seconds.
    model_bold (bool): If True, apply the Balloon-Windkessel model to convert neural activity to BOLD signal. Default is False.

    Returns:
    sim_activity (ndarray): Simulated activity on the surface mesh over time, shape (n_verts, time_points).
    """
    n_verts, n_modes = eigvecs.shape
    external_input = np.random.randn(n_verts, time_points)

    # do a mode decomposition of the external input
    input_coeffs = eigvecs.T @ mass @ external_input

    # Initialize simulated activity vector
    mode_coeffs = np.zeros((n_modes, time_points))
    for mode_ind in range(n_modes):
        input_coeffs_i = input_coeffs[mode_ind, :]
        eval = eigvals[mode_ind]
        neural = model_wave_fourier(input_coeffs_i, dt=dt, r=28.9, gamma=0.116, eval=eval)
        if model_bold:
            bold = model_balloon_fourier(neural, dt)
            mode_coeffs[mode_ind, :] = bold
        else:
            mode_coeffs[mode_ind, :] = neural
            
    # Combine the mode activities to get the total simulated activity
    sim_activity = zscore(eigvecs @ mode_coeffs, axis=1)
    return sim_activity



def main():
    micapipe='/local_raid/data/pbautin/software/micapipe'
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure2_df.tsv')
    surfs = load_conte69(join=True)
    surfs = mesh_operations.mask_points(surfs, np.array(~df_yeo_surf.hemisphere.isna()))
    
    eigvals, eigvecs, stiffness, mass = solve_laplace_beltrami(surfs, curvature_param=10)
    sim_fmri = generate_synthetic_fmri(eigvals, eigvecs, mass, time_points=750)[:, 50:]
    # create a matrix of nan the size of the full surface and fill in the simulated data
    full_n_verts = df_yeo_surf.hemisphere.values.shape[0]
    sim_fmri_full = np.full((full_n_verts, sim_fmri.shape[1]), np.nan)
    sim_fmri_full[~df_yeo_surf.hemisphere.isna(), :] = sim_fmri
    print(sim_fmri_full.shape)
    # Correlation matrix of the simulated data
    sim_fmri_parcel = reduce_by_labels(sim_fmri_full, df_yeo_surf.mics.values, axis=1)
    corr = np.corrcoef(sim_fmri_parcel)
    print(corr.shape)
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.show()

    # Compute gradient using brainspace
    valid = ~np.all(np.isnan(corr), axis=1)
    print(valid.shape)
    corr_z = corr[np.ix_(valid, valid)]
    corr_z = np.arctanh(corr_z)
    corr_z = np.nan_to_num(corr_z, nan=0.0, posinf=0.0, neginf=0.0)
    gm_t1 = GradientMaps(n_components=3, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm_t1.fit(corr_z, sparsity=0)

    parcel_grad = np.full((corr.shape[0],3), np.nan)
    parcel_grad[valid,:] = gm_t1.gradients_[:, :]
    plot_val = map_to_labels(parcel_grad.T, df_yeo_surf.mics.values)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, plot_val, cmap='coolwarm', color_bar=True, color_range='sym')


    print(sim_fmri.shape)
    # plot sim_fmri with brainspace
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, sim_fmri_full[:, ::15].T, cmap='coolwarm', color_bar=True, color_range=(-3,3))


if __name__ == "__main__":
    main()
