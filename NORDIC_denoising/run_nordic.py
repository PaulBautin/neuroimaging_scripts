# ------------------------------------------------------------------------------
# Script to Run NORDIC Denoising on Diffusion MRI Data
# ------------------------------------------------------------------------------
#
# Feature justification
# 7T dMRI images SNR is poor.
#
# References:
# - Original Paper: https://doi.org/10.1016/j.neuroimage.2020.117539
# - GitHub Repository: https://github.com/SteenMoeller/NORDIC_Raw
#
# Before Running:
# - Ensure MATLAB version 2017b or newer is installed with the Image Processing
#   Toolbox and Signal Processing Toolbox.
# - Clone the NORDIC GitHub repository.
#
# Input Data:
# - This script is tested with a 2mm dMRI image from the Edden dataset
#   available at: https://openneuro.org/datasets/ds004666/versions/1.0.5
# - Note: Requires dMRI magnitude and phase images.
# - Note: Not yet able to test 1.5mm dMRI images, > 16 GB RAM is required.
#
# Previous Implementation:
# - The dMRI data was previously denoised using the MPPCA (Marchenko-Pastur Principal
#   Component Analysis) method, implemented in MRtrix3's `dwidenoise` command.
#   Reference: https://mrtrix.readthedocs.io/en/dev/reference/commands/dwidenoise.html
#
# Questions:
# - When should NORDIC be run?. Example. before or after FSL-eddy
# - How to evaluate improvements?
#
# Future Considerations:
# - The NORDIC method could be adapted for other MRI modalities such as fMRI
#   or ASL.
# - Implementing NORDIC similarly to the FIX tool used in micapipe fMRI (similar Matlab requirements?).
#
# Alternatives:
# - DIPY's patch2self https://github.com/nipreps/dmriprep/issues/132
#   Reference: https://docs.dipy.org/stable/examples_built/preprocessing/denoise_patch2self.html
# ------------------------------------------------------------------------------

import matlab.engine
import os

class NordicRawWrapper:
    def __init__(self, matlab_script_dir: str):
        """
        Initialize the NordicRawWrapper with the path to the MATLAB script directory.

        :param matlab_script_dir: Path to the directory containing NORDIC Raw MATLAB scripts.
        """
        self.matlab_script_dir = matlab_script_dir
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(matlab_script_dir)

    def run_denoising(self, mag_file: str, phase_file: str, output_file: str, matlab_args: str):
        """
        Run the NORDIC Raw denoising on a specified image file.

        :param input_file: Path to the input image file.
        :param output_file: Path to the output (denoised) image file.
        :param noise_level: A parameter for noise level, passed to the MATLAB function (customize based on your needs).
        """
        # Assuming that NORDIC Raw has a function named 'denoise_image', adapt if needed
        self.eng.NIFTI_NORDIC(mag_file, phase_file, output_file, matlab_args, nargout=0)
        print(f"Denoising completed for {input_file}. Output saved to {output_file}.")

    def close(self):
        """
        Close the MATLAB engine.
        """
        self.eng.quit()
        print("MATLAB engine closed.")

# Example Usage:
if __name__ == "__main__":
    # Set your MATLAB directory where NORDIC Raw scripts are located
    matlab_script_dir = "/local_raid/data/pbautin/software/NORDIC_Raw"

    # Input and output file paths
    mag_image = "/local_raid/data/pbautin/data/sub-Pilot014/ses-01/dwi/sub-Pilot014_ses-01_acq_multib_70d_dir-AP_dwi.nii.gz"
    phase_image = "/local_raid/data/pbautin/data/sub-Pilot014/ses-01/dwi/sub-Pilot014_ses-01_acq_multib_70d_dir-AP_part-phase_dwi.nii.gz"
    output_image = "sub-Pilot014_ses-01_acq_multib_70d_dir-AP_nordic_"
    matlab_args = "ARG.DIROUT = '/local_raid/data/pbautin/results/NORDIC_denoising/';"

    # Create wrapper instance
    nordic_wrapper = NordicRawWrapper(matlab_script_dir)

    # Run the denoising process
    nordic_wrapper.run_denoising(mag_image, phase_image, output_image, matlab_args)

    # Close the MATLAB engine
    nordic_wrapper.close()
