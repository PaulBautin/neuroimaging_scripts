"""
This is the interface API to compute MT-related metrics

Code is based on QMRLab: https://github.com/neuropoly/qMRLab

Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
License: see the file LICENSE
"""


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import logging
from scipy.ndimage import zoom, gaussian_filter
from scipy.interpolate import interp1d
logger = logging.getLogger(__name__)


def MP2RAGE_lookuptable(
    nimages,
    MPRAGE_tr,
    inversiontimes,
    flipangles,
    nZslices,
    FLASH_tr,
    sequence,
    inversion_efficiency=0.96,
    alldata=0,
    T1vector=np.arange(0.05, 8.05, 0.05),
):
    """
    Calculate the MP2RAGE lookup table.

    Parameters:
        nimages (int): Number of images.
        MPRAGE_tr (float): Repetition time of the MP2RAGE sequence (in seconds).
        inversiontimes (array_like): Inversion times [TI1, TI2] in seconds.
        flipangles (array_like): Flip angles [FA1, FA2] in degrees.
        nZslices (array_like): Number of slices before and after inversion.
        FLASH_tr (float): Repetition time of the FLASH sequence (in seconds).
        sequence (str): 'normal' or 'waterexcitation'.
        inversion_efficiency (float, optional): Inversion efficiency (default is 0.96).
        alldata (int, optional): If 1, all data is shown; if 0, only the monotonic part is shown (default is 0).
        T1vector (array_like): Create T1 values ranging from 0.05 to 8 seconds in 0.05-second increments

    Returns:
        Tuple containing:
            - Intensity (ndarray): The calculated intensity values.
            - T1vector (ndarray): The T1 values corresponding to the intensities.
            - IntensityBeforeComb (ndarray): The signal intensities before combination.
    """
    # Ensure inversiontimes and flipangles are NumPy arrays and validate their lengths
    inversiontimes = np.asarray(inversiontimes)
    flipangles = np.asarray(flipangles)
    nZslices = np.asarray(nZslices)
    if inversiontimes.size != 2 or flipangles.size != 2:
        raise ValueError("inversiontimes, flipangles and nZslices must be arrays of length 2. (MP2RAGE sequence)")

    # Check acquisition conditions
    conditions_met = (
        (inversiontimes[1] - inversiontimes[0]) >= (nZslices[0] + nZslices[1]) * FLASH_tr and
        inversiontimes[0] >= nZslices[0] * FLASH_tr and
        inversiontimes[1] <= MPRAGE_tr - nZslices[1] * FLASH_tr
    )

    if not conditions_met:
        # If conditions are not met, return zeros
        Signal = np.zeros((len(T1vector), 2), dtype=complex)
    else:
        # Calculate the signal for all T1 values at once
        Signal = MPRAGEfunc(
            nimages,
            MPRAGE_tr,
            inversiontimes,
            nZslices,
            FLASH_tr,
            flipangles,
            sequence,
            T1vector,
            inversionefficiency=inversion_efficiency
        ).T

    # Calculate the intensity using the combination of two images
    S1, S2 = Signal[:, 0], Signal[:, 1]
    numerator = np.real(S1 * np.conj(S2))
    denominator = np.abs(S1) ** 2 + np.abs(S2) ** 2

    # Handle division by zero and invalid values
    with np.errstate(divide='ignore', invalid='ignore'):
        Intensity = numerator / denominator
        Intensity = np.nan_to_num(Intensity, nan=0.0)

    if alldata == 0:
        # Find indices where the intensity is monotonic
        max_index = np.argmax(Intensity)
        min_index = np.argmin(Intensity)

        # Ensure correct ordering
        start_index = min(max_index, min_index)
        end_index = max(max_index, min_index) + 1  # Include end index

        # Extract monotonic part
        Intensity = Intensity[start_index:end_index]
        T1vector = T1vector[start_index:end_index]
        IntensityBeforeComb = Signal[start_index:end_index, :]

        # Pad the lookup table to avoid out-of-range values
        Intensity[0] = 0.5
        Intensity[-1] = -0.5
    else:
        # Use all data
        IntensityBeforeComb = Signal

    return Intensity, T1vector, IntensityBeforeComb


def MPRAGEfunc(
    nimages,
    MPRAGE_tr,
    inversiontimes,
    nZslices,
    FLASH_tr,
    flipangle,
    sequence,
    T1s,
    inversionefficiency=0.96,
):
    """
    Calculate the MPRAGE signal for given parameters.

    Parameters:
        nimages (int): Number of images.
        MPRAGE_tr (float): Repetition time of the MPRAGE sequence (in seconds).
        inversiontimes (array_like): Inversion times for each image (in seconds).
        nZslices (array_like): Number of Z slices before and after inversion.
        FLASH_tr (float): Repetition time of the FLASH sequence (in seconds).
        flipangle (float or array_like): Flip angle(s) in degrees.
        sequence (str): 'normal' or 'waterexcitation'.
        T1s (float or array_like): T1 relaxation times (in seconds).
        inversionefficiency (float, optional): Inversion efficiency (default is 0.96).

    Returns:
        ndarray: Calculated signal for each image and T1 value.
    """
    # Convert inputs to NumPy arrays for vectorized operations
    T1s = np.atleast_1d(T1s)
    # Convert flip angles from degrees to radians
    fliprad = np.deg2rad(np.atleast_1d(flipangle))
    # Equilibrium magnetization
    M0 = 1.0
    # Calculate time intervals
    TA = nZslices * FLASH_tr
    # Time delays (TD) between inversions and acquisitions
    TD = np.zeros(nimages + 1)
    TD[0] = inversiontimes[0] - TA[0]  # Time before the first inversion
    TD[1] = inversiontimes[1] - inversiontimes[0] - (TA[0] + TA[1])
    TD[2] = MPRAGE_tr - inversiontimes[1] - TA[1]  # Time after the last inversion


    # Sequence-specific parameters
    if sequence == 'normal':
        # Normal sequence calculations
        E1_FLASH = np.exp(-FLASH_tr / T1s)  # Relaxation during FLASH_TR
        cos_alpha_E1 = np.cos(fliprad)[:, np.newaxis] * E1_FLASH  # Shape: (nimages, len(T1s))
        sin_alpha = np.sin(fliprad)
    elif sequence == 'waterexcitation':
        # Water excitation sequence calculations
        B0 = 7  # Tesla
        FatWaterCSppm = 3.3  # ppm
        gamma = 42.576  # MHz/T
        pulseSpace = 1 / (2 * FatWaterCSppm * B0 * gamma)  # Pulse spacing

        E1_A = np.exp(-pulseSpace / T1s)
        E2_A = np.exp(-pulseSpace / 0.06)  # T2* ~ 60 ms
        E1_B = np.exp(-(FLASH_tr - pulseSpace) / T1s)

        cos_half_fliprad = np.cos(fliprad / 2)
        sin_half_fliprad = np.sin(fliprad / 2)

        cos_alpha_E1 = (
            (cos_half_fliprad ** 2)[:, np.newaxis] * (E1_A * E1_B)
            - (sin_half_fliprad ** 2)[:, np.newaxis] * (E2_A * E1_B)
        )
        sin_alpha = sin_half_fliprad * cos_half_fliprad * (E1_A + E2_A)
        E1_FLASH = np.exp(-FLASH_tr / T1s)  # Reuse E1_FLASH for consistency
    else:
        raise ValueError("Invalid sequence type. Must be 'normal' or 'waterexcitation'.")

    # Exponential relaxation during TDs
    E_TD = np.exp(-TD[:, np.newaxis] / T1s)  # Shape: (nimages + 1, len(T1s))

    # Steady-state longitudinal magnetization (Mz)
    prod_cos_alpha_E1 = np.prod(cos_alpha_E1 ** (nZslices[0] + nZslices[1]), axis=0)  # Shape: (len(T1s),)
    prod_E_TD = np.prod(E_TD, axis=0)  # Shape: (len(T1s),)
    denominator = 1 + inversionefficiency * prod_cos_alpha_E1 * prod_E_TD
    Mz_ss = M0 / denominator  # Steady-state Mz, shape: (len(T1s),)

    # Numerator for Mz steady-state calculation
    Mz_numerator = M0 * (1 - E_TD[0])

    for k in range(nimages):
        cos_alpha_E1_k = cos_alpha_E1[k] ** (nZslices[0] + nZslices[1])  # Shape: (len(T1s),)
        term1 = Mz_numerator * cos_alpha_E1_k
        term2 = M0 * (1 - E1_FLASH) * (1 - cos_alpha_E1_k) / (1 - cos_alpha_E1[k])

        Mz_numerator = term1 + term2

        if k + 1 < nimages + 1:
            # Update Mz_numerator for the next inversion
            Mz_numerator = Mz_numerator * E_TD[k + 1] + M0 * (1 - E_TD[k + 1])

    # Final steady-state magnetization
    Mz_ss *= Mz_numerator  # Shape: (len(T1s),)

    # Initialize the signal array
    signal = np.zeros((nimages, len(T1s)))

    # Calculate the signal for the first image
    temp = (
        (-inversionefficiency * Mz_ss * E_TD[0] + M0 * (1 - E_TD[0]))
        * (cos_alpha_E1[0] ** nZslices[0])
        + M0 * (1 - E1_FLASH) * (1 - cos_alpha_E1[0] ** nZslices[0]) / (1 - cos_alpha_E1[0])
    )
    signal[0] = sin_alpha[0] * temp  # Signal for the first image

    # Calculate the signal for the second image
    # Update temp with relaxation and flip angle effects
    temp = (
        temp * (cos_alpha_E1[0] ** nZslices[1])
        + M0
        * (1 - E1_FLASH)
        * (1 - cos_alpha_E1[0] ** nZslices[1])
        / (1 - cos_alpha_E1[0])
    )
    temp = (
        (temp * E_TD[1] + M0 * (1 - E_TD[1])) * (cos_alpha_E1[1] ** nZslices[0])
        + M0 * (1 - E1_FLASH) * (1 - cos_alpha_E1[1] ** nZslices[0]) / (1 - cos_alpha_E1[1])
    )
    signal[1] = sin_alpha[1] * temp

    return signal  # Shape: (nimages, len(T1s))


def resample_image(source_img, target_img):
    source_data = source_img.get_fdata()
    target_shape = target_img.shape
    source_shape = source_img.shape

    # Calculate the zoom factors
    zoom_factors = [t / s for t, s in zip(target_shape, source_shape)]

    # Resample the data (default order of the spline interpolation is 3)
    resampled_data = zoom(source_data, zoom_factors, order=3)

    # Create new NIfTI image with the resampled data
    resampled_img = nib.Nifti1Image(resampled_data, target_img.affine, target_img.header)
    return resampled_img


def mask_for_B1(ref_img):
    # Placeholder function for generating a mask
    data = ref_img.get_fdata()
    mask = np.ones(data.shape, dtype=bool)
    return mask


def smoothB1(B1map_img, B1map_norm, B1FWHM, mask):
    # Smooth the B1 map using a Gaussian filter and apply the mask
    voxel_size = B1map_img.header.get_zooms()[:3]
    # FWHM = σ √(8 ln(2)) = σ * 2.35482
    sigma = [fw / 2.35482 / vs for fw, vs in zip(B1FWHM, voxel_size)]
    smoothed = gaussian_filter(B1map_norm, sigma=sigma)
    smoothed_masked = np.where(mask, smoothed, 0)
    return smoothed_masked


def main():
    # MP2RAGE parameters
    MP2RAGE = {
        'B0': 7,  # in Tesla
        'TR': 5.17,  # MP2RAGE TR in seconds
        'TRFLASH': 8.9e-3,  # GRE readout TR in seconds
        'TIs': [1.0, 3.2],  # Inversion times in seconds
        'NZslices': [80, 160],  # Slices per slab
        'FlipDegrees': [4, 4],  # Flip angles in degrees
        'filenameUNI': '/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_acq-uni_0p5-T1map.nii.gz',
        'filenameINV2': '/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_acq-inv2_0p5-T1map.nii.gz',
        'filenameINV1': '/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_acq-inv1_0p5-T1map.nii.gz',
        'filenameB1': '/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/fmap/sub-Pilot014_ses-02_acq-b1sag_run-1_fieldmap.nii.gz',
        'filenameMTon': '/local_raid/data/pbautin/data/pilot_dataset/sub-Pilot014/ses-02/anat/sub-Pilot014_ses-02_mt-on_MTR.nii.gz',
    }
    print(nib.load(MP2RAGE['filenameMTon']).header)

    # # Call the MPRAGEfunc function
    # signal = MPRAGEfunc(
    #     nimages=len(MP2RAGE['TIs']),
    #     MPRAGE_tr=MP2RAGE['TR'],
    #     inversiontimes=MP2RAGE['TIs'],
    #     nZslices=MP2RAGE['NZslices'],
    #     FLASH_tr=MP2RAGE['TRFLASH'],
    #     flipangle=MP2RAGE['FlipDegrees'],
    #     sequence='normal',
    #     T1s=np.linspace(0.1, 5.0, 1000),
    # )

    # Call the MP2RAGE_lookuptable function
    Intensity, T1vector, IntensityBeforeComb = MP2RAGE_lookuptable(
        nimages=len(MP2RAGE['TIs']),
        MPRAGE_tr=MP2RAGE['TR'],
        inversiontimes=MP2RAGE['TIs'],
        flipangles=MP2RAGE['FlipDegrees'],
        nZslices=MP2RAGE['NZslices'],
        FLASH_tr=MP2RAGE['TRFLASH'],
        sequence='normal',
        inversion_efficiency=0.96,
        alldata=0,  # Set to 1 if you want to include all data
    )

    # Display the results
    print("Intensity shape:", Intensity.shape)
    print("T1vector shape:", T1vector.shape)
    print("IntensityBeforeComb shape:", IntensityBeforeComb.shape)

    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(10, 6))
    plt.plot(T1vector, Intensity)
    plt.xlabel('T1 (s)')
    plt.ylabel('Intensity (UNI)')
    plt.title('MP2RAGE UNI Lookup Table')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(T1vector, IntensityBeforeComb[:,1])
    plt.xlabel('T1 (s)')
    plt.ylabel('Intensity(INV2)')
    plt.title('MP2RAGE INV2 Lookup Table')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(T1vector, IntensityBeforeComb[:,0])
    plt.xlabel('T1 (s)')
    plt.ylabel('Intensity(INV1)')
    plt.title('MP2RAGE INV1 Lookup Table')
    plt.grid(True)
    plt.show()

    ### Create T1 map
    # Map intensity values to T1 values
    uni_img = nib.load(MP2RAGE['filenameUNI'])
    # Correct for Siemens scaling: 0 -> 4095 to -0.5 -> 0.5
    uni_data = uni_img.get_fdata() / 4095 - 0.5
    # Create an interpolation function
    interp_func = interp1d(Intensity, T1vector, bounds_error=False, fill_value="extrapolate")
    t1_data = interp_func(uni_data)

    plt.figure(figsize=(8, 6))
    plt.imshow(t1_data[:,:,250].T, origin='lower')
    plt.colorbar()
    plt.title('T1 (s) map')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(uni_data[:,:,250].T, origin='lower')
    plt.colorbar()
    plt.title('MP2RAGE UNI image')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(nib.load(MP2RAGE['filenameINV1']).get_fdata()[:,:,250].T, origin='lower')
    plt.colorbar()
    plt.title('MP2RAGE INV1 image')
    plt.show()

    ### Create M0 map
    inv2_img = nib.load(MP2RAGE['filenameINV2'])
    inv2_data = inv2_img.get_fdata()
    # Create an interpolation function
    interp_func = interp1d(T1vector, IntensityBeforeComb[:,1], bounds_error=False, fill_value="extrapolate")
    m0_data = inv2_data / interp_func(t1_data)

    plt.figure(figsize=(8, 6))
    plt.imshow(interp_func(t1_data[:,:,250]).T, origin='lower')
    plt.colorbar()
    plt.title('theoric MP2RAGE INV2 (M0=1)')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(inv2_data[:,:,250].T, origin='lower')
    plt.colorbar()
    plt.title('MP2RAGE INV2 image')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(m0_data[:,:,250].T, origin='lower')
    plt.colorbar()
    plt.title('M0 map')
    plt.show()

    ### Create B1 map
    b1_img = nib.load(MP2RAGE['filenameB1'])
    b1_data = b1_img.get_fdata()
    # Permute dimensions (equivalent to MATLAB's permute with [3, 1, 2])
    b1_data = np.transpose(b1_data, (2, 0, 1))
    # Apply flips on individual dimensions
    b1_data = np.flip(b1_data, axis=0)  # Flip along the 1st dimension
    b1_data = np.flip(b1_data, axis=1)  # Flip along the 2nd dimension
    b1_data = np.flip(b1_data, axis=2)  # Flip along the 3rd dimension
    b1_img = nib.Nifti1Image(b1_data, b1_img.affine, b1_img.header)
    b1_data = resample_image(b1_img, uni_img).get_fdata() / 899
    b1_img = nib.Nifti1Image(b1_data, b1_img.affine, b1_img.header)
    mask = mask_for_B1(b1_img)
    b1_data = smoothB1(b1_img, b1_data, [8,8,8], mask)

    plt.figure(figsize=(8, 6))
    plt.imshow(b1_data[:,:,250].T, origin='lower')
    plt.colorbar()
    plt.title('ΔB1 map')
    plt.show()

    ### Calculate MTsat image
    tr_mt = 0.095;
    fa_mt_rad = np.deg2rad(5)
    mton_img = nib.load(MP2RAGE['filenameMTon'])
    mton_img = resample_image(mton_img, uni_img)
    mton_data = mton_img.get_fdata()

    mtsat_data = np.abs((((m0_data  * fa_mt_rad * b1_data) / mton_data) - 1) * (tr_mt / t1_data) - ((fa_mt_rad * b1_data)**2 / 2))
    mtsat_data = mtsat_data * ((1 - 0.4) / (1 - (0.4 * b1_data)))

    plt.figure(figsize=(8, 6))
    plt.imshow(mton_data[:,:,250].T, origin='lower')
    plt.colorbar()
    plt.title('MT-on image')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.imshow(mtsat_data[:,:,250].T, origin='lower')
    plt.colorbar()
    plt.title('MT-sat map')
    plt.show()


if __name__ == "__main__":
    main()
