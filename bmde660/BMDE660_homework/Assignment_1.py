import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the .mat file containing the MR image
def load_mr_image(file_path):
    # Load the .mat file
    data = loadmat(file_path)['slice']
    return data

# Compute the k-space data (Fourier transform of the image)
def compute_k_space(image):
    # Apply 2D Fast Fourier Transform (FFT) and shift the zero-frequency component to the center
    k_space = np.fft.fftshift(np.fft.fft2(image))
    return k_space

def reduce_and_zero_fill_k_space(k_space, factor=2):
    rows, cols = k_space.shape
    new_rows, new_cols = rows // factor, cols // factor
    row_center, col_center = rows // 2, cols // 2
    zero_filled_k_space = np.zeros_like(k_space, dtype=complex)
    zero_filled_k_space[
        row_center - new_rows // 2: row_center + new_rows // 2,
        col_center - new_cols // 2: col_center + new_cols // 2
    ] = k_space[
        row_center - new_rows // 2: row_center + new_rows // 2,
        col_center - new_cols // 2: col_center + new_cols // 2
    ]
    return zero_filled_k_space

def zero_out_central_k_space(k_space, percent=10):
    rows, cols = k_space.shape
    
    # Calculate the size of the central region to zero out
    central_rows = int(rows * percent / 100)
    central_cols = int(cols * percent / 100)
    
    # Calculate the indices for the central region
    row_start = rows // 2 - central_rows // 2
    row_end = rows // 2 + central_rows // 2
    col_start = cols // 2 - central_cols // 2
    col_end = cols // 2 + central_cols // 2
    
    # Zero out the central region
    k_space[row_start:row_end, col_start:col_end] = 0
    return k_space

# Zero-pad k-space by a factor of 2
def zero_pad_k_space(k_space, factor=2):
    rows, cols = k_space.shape

    # Calculate the new dimensions
    new_rows, new_cols = rows * factor, cols * factor

    # Create a zero-padded array
    zero_padded_k_space = np.zeros((new_rows, new_cols), dtype=complex)

    # Calculate the center indices for the original and new arrays
    row_start = (new_rows - rows) // 2
    row_end = row_start + rows
    col_start = (new_cols - cols) // 2
    col_end = col_start + cols

    # Place the original k-space data in the center of the zero-padded array
    zero_padded_k_space[row_start:row_end, col_start:col_end] = k_space
    print(zero_padded_k_space.shape)

    return zero_padded_k_space

def discard_alternate_k_space_lines(k_space):
    # Set every other row to zero
    k_space[::2, :] = 0
    return k_space


def shift_k_space_lines(k_space):
    # rows, cols = k_space.shape
    # ky = np.fft.fftfreq(rows)  # Frequency indices along k_y

    # # Create a phase shift array
    # phase_shift = np.exp(1j * 2 * np.pi * delta_ky * ky[:, None])

    # # Apply shifts: even rows get +Δk_y, odd rows get -Δk_y
    # k_space_shifted = k_space.copy()
    # k_space_shifted[0::2, :] *= phase_shift[0::2, :]
    # k_space_shifted[1::2, :] *= np.conj(phase_shift[1::2, :])  # Negative shift for odd rows


    k_space_shifted = np.zeros(k_space.shape)
    k_space_shifted[1:,::2] = k_space[:-1,::2]
    k_space_shifted[:-1,1::2] = k_space[1:,1::2]
    return k_space_shifted

# Display the original image and k-space data
def display_image_and_k_space(k_space):
    # Set up the figure
    plt.figure(figsize=(12, 6))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(np.abs(np.fft.ifft2(np.fft.ifftshift(k_space))), cmap='gray')
    plt.title('2D MR image')
    plt.axis('off')

    # Display the magnitude of the k-space data
    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(k_space)+1), cmap='gray')  # log scaling for better visualization
    plt.title('Corresponding k-space data (log scaling)')
    plt.axis('off')

    plt.show()


# Main execution
if __name__ == "__main__":
    # Path to the .mat file
    file_path = "/home/pabaua/Downloads/MRimage.mat"

    try:
        # Load the MR image
        image = load_mr_image(file_path)

        # Compute k-space data
        k_space = compute_k_space(image)
        #k_space = reduce_and_zero_fill_k_space(k_space, factor=2)
        #k_space = zero_out_central_k_space(k_space, percent=10)
        #k_space = zero_pad_k_space(k_space, factor=2)
        #k_space = discard_alternate_k_space_lines(k_space)
        k_space = shift_k_space_lines(k_space)

        # Display the image and k-space
        display_image_and_k_space(k_space)

    except Exception as e:
        print(f"Error: {e}")