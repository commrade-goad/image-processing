import numpy as np
import matplotlib.pyplot as plt

def fourier_transform_1d_example():
    """
    Demonstrate 1D Fourier Transform using NumPy
    """
    # Create a sample signal: sum of sine waves with different frequencies
    sample_rate = 1000  # sample rate in Hz
    t = np.linspace(0, 1, sample_rate)  # 1 second time array

    # Create a signal with multiple frequency components
    freq1, freq2, freq3 = 5, 20, 50  # Hz
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t) + 0.3 * np.sin(2 * np.pi * freq3 * t)

    # Compute the FFT (Fast Fourier Transform)
    fft_result = np.fft.fft(signal)

    # Calculate frequency bins
    freqs = np.fft.fftfreq(len(signal), 1/sample_rate)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot original signal
    plt.subplot(211)
    plt.plot(t[:200], signal[:200])  # Show first 0.2 seconds
    plt.title('Original Signal (First 0.2 seconds)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Plot magnitude spectrum (take half of the FFT result - up to Nyquist frequency)
    plt.subplot(212)
    plt.plot(freqs[:sample_rate//2], np.abs(fft_result[:sample_rate//2]) / sample_rate)
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    # Mark the frequencies used to create the signal
    plt.axvline(x=freq1, color='r', linestyle='--', alpha=0.7, label=f'{freq1} Hz')
    plt.axvline(x=freq2, color='g', linestyle='--', alpha=0.7, label=f'{freq2} Hz')
    plt.axvline(x=freq3, color='b', linestyle='--', alpha=0.7, label=f'{freq3} Hz')
    plt.legend()

    plt.tight_layout()
    plt.savefig("fourier1.png")
    plt.close()

def fourier_transform_2d_example():
    """
    Demonstrate 2D Fourier Transform using NumPy
    """
    # Create a simple 2D pattern: checkerboard pattern
    size = 128
    checkerboard = np.zeros((size, size))
    checkerboard[::2, ::2] = 1
    checkerboard[1::2, 1::2] = 1

    # Compute the 2D FFT
    fft_result = np.fft.fft2(checkerboard)

    # Shift the zero frequency component to the center
    fft_shifted = np.fft.fftshift(fft_result)

    # Calculate the magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = 20 * np.log10(np.abs(fft_shifted) + 1)

    # Plot the results
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.imshow(checkerboard, cmap='gray')
    plt.title('Original 2D Pattern (Checkerboard)')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap='viridis')
    plt.title('2D Fourier Transform Magnitude (Log Scale)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("fourier2.png")
    plt.close()

    # Additional example: Create and transform an image with concentric circles
    x = np.linspace(-4, 4, size)
    y = np.linspace(-4, 4, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Create a pattern with concentric circles
    circles = np.cos(R * np.pi * 2) * 0.5 + 0.5

    # Compute the 2D FFT
    fft_result = np.fft.fft2(circles)
    fft_shifted = np.fft.fftshift(fft_result)
    magnitude_spectrum = 20 * np.log10(np.abs(fft_shifted) + 1)

    # Plot the results
    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.imshow(circles, cmap='gray')
    plt.title('Concentric Circles Pattern')
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(magnitude_spectrum, cmap='viridis')
    plt.title('2D Fourier Transform Magnitude (Log Scale)')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("fourier-extra.png")
    plt.close()

if __name__ == "__main__":
    print("Running 1D Fourier Transform example...")
    fourier_transform_1d_example()

    print("\nRunning 2D Fourier Transform example...")
    fourier_transform_2d_example()
