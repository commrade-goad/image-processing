import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct, dctn, idctn

def dct_1d_example():
    """
    Demonstrate 1D Discrete Cosine Transform
    """
    # Create a sample signal
    N = 100
    n = np.arange(N)
    # Create a signal with multiple frequency components
    signal = 3 * np.cos(0.1 * np.pi * n) + 2 * np.cos(0.4 * np.pi * n) + np.cos(0.8 * np.pi * n)

    # Compute the DCT
    dct_result = dct(signal, type=2, norm='ortho')

    # Compute inverse DCT to reconstruct the signal
    reconstructed_signal = idct(dct_result, type=2, norm='ortho')

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot original signal
    plt.subplot(311)
    plt.plot(n, signal)
    plt.title('Original Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot DCT coefficients
    plt.subplot(312)
    plt.stem(n, np.abs(dct_result))
    plt.title('DCT Coefficients')
    plt.xlabel('Frequency bin')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Plot reconstructed signal
    plt.subplot(313)
    plt.plot(n, reconstructed_signal)
    plt.title('Reconstructed Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("dct1.png")
    plt.close()

    # Demonstrate signal compression using DCT
    # Keep only the first 10 coefficients (90% reduction)
    compressed_dct = np.zeros_like(dct_result)
    compressed_dct[:10] = dct_result[:10]

    # Reconstruct from compressed DCT
    compressed_signal = idct(compressed_dct, type=2, norm='ortho')

    # Calculate compression error
    error = np.sum((signal - compressed_signal) ** 2) / np.sum(signal ** 2)

    # Plot compression results
    plt.figure(figsize=(10, 6))
    plt.plot(n, signal, 'b-', label='Original Signal')
    plt.plot(n, compressed_signal, 'r--', label=f'Compressed Signal (10% coefficients)')
    plt.title(f'Signal Compression using DCT\nRelative Error: {error:.6f}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

    plt.savefig("dct1-2.png")
    plt.close()

def dct_2d_example():
    """
    Demonstrate 2D Discrete Cosine Transform
    """
    # Create a simple test image (8x8 checkerboard pattern)
    img_size = 8
    checkerboard = np.zeros((img_size, img_size))
    for i in range(img_size):
        for j in range(img_size):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = 1

    # Create a larger test image
    large_size = 256
    x = np.linspace(0, 1, large_size)
    y = np.linspace(0, 1, large_size)
    X, Y = np.meshgrid(x, y)
    test_image = np.sin(10 * np.pi * X) * np.sin(10 * np.pi * Y)

    # Compute the 2D DCT
    dct_result = dctn(test_image, norm='ortho')

    # Plot original image and DCT coefficients
    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.imshow(test_image, cmap='viridis')
    plt.title('Original Image')
    plt.colorbar()

    plt.subplot(122)
    # Display log scale for better visualization
    plt.imshow(np.log(abs(dct_result) + 1e-5), cmap='viridis')
    plt.title('2D DCT Coefficients (Log Scale)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("dct2.png")
    plt.close()

    # Demonstrate image compression
    # Keep only a fraction of the coefficients (largest ones)
    dct_copy = dct_result.copy()
    threshold = np.percentile(abs(dct_result), 95)  # Keep top 5% of coefficients
    mask = abs(dct_copy) < threshold
    dct_copy[mask] = 0

    # Calculate compression ratio
    nonzero_count = np.count_nonzero(dct_copy)
    total_count = dct_copy.size
    compression_ratio = (total_count - nonzero_count) / total_count * 100

    # Reconstruct from compressed DCT
    compressed_image = idctn(dct_copy, norm='ortho')

    # Calculate RMSE
    rmse = np.sqrt(np.mean((test_image - compressed_image) ** 2))

    # Plot compression results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(test_image, cmap='viridis')
    plt.title('Original Image')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(compressed_image, cmap='viridis')
    plt.title(f'Compressed Image\n{compression_ratio:.1f}% of coefficients set to zero')
    plt.colorbar()

    plt.subplot(133)
    diff = np.abs(test_image - compressed_image)
    plt.imshow(diff, cmap='hot')
    plt.title(f'Absolute Difference\nRMSE: {rmse:.4f}')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("dct2-2.png")
    plt.close()

    # JPEG-like compression demonstration using 8x

if __name__ == "__main__":
    dct_1d_example()
    dct_2d_example()
