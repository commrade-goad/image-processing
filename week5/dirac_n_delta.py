import numpy as np
import matplotlib.pyplot as plt

def dirac_1d(x, epsilon=0.1):
    """
    Create an approximation of the Dirac delta function in 1D
    using a narrow Gaussian
    """
    return np.exp(-x**2 / (2 * epsilon**2)) / (epsilon * np.sqrt(2 * np.pi))

# 1D convolution example
def convolution_1d_example():
    # Create a signal (e.g., a rectangular pulse)
    x = np.linspace(-5, 5, 1000)
    signal = np.zeros_like(x)
    signal[(x >= -2) & (x <= 2)] = 1.0

    # Create a Dirac-like impulse (narrow Gaussian)
    impulse = dirac_1d(x, epsilon=0.2)

    # Perform convolution
    conv_result = np.convolve(signal, impulse / np.sum(impulse), mode='same')

    # Plot results
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.plot(x, signal)
    plt.title('Original Signal')
    plt.grid(True)

    plt.subplot(132)
    plt.plot(x, impulse)
    plt.title('Dirac-like Impulse')
    plt.grid(True)

    plt.subplot(133)
    plt.plot(x, conv_result)
    plt.title('Convolution Result')
    plt.grid(True)

    plt.savefig("dirac1.png")
    plt.close()

# 2D Dirac delta function approximation
def dirac_2d(x, y, epsilon=0.1):
    """
    Create an approximation of the Dirac delta function in 2D
    using a narrow Gaussian
    """
    return np.exp(-(x**2 + y**2) / (2 * epsilon**2)) / (2 * np.pi * epsilon**2)

# 2D convolution example
def convolution_2d_example():
    # Create a 2D grid
    n = 100
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, y)

    # Create a 2D signal (a square)
    signal = np.zeros((n, n))
    signal[(X >= -2) & (X <= 2) & (Y >= -2) & (Y <= 2)] = 1.0

    # Create a 2D Dirac-like impulse (narrow Gaussian)
    impulse = dirac_2d(X, Y, epsilon=0.3)

    # Normalize the impulse
    impulse = impulse / np.sum(impulse)

    # Perform 2D convolution using FFT (much faster for larger arrays)
    conv_result = np.real(np.fft.ifft2(np.fft.fft2(signal) * np.fft.fft2(impulse)))

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(signal, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
    plt.title('Original Signal (Square)')
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(impulse, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
    plt.title('2D Dirac-like Impulse')
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(conv_result, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
    plt.title('2D Convolution Result')
    plt.colorbar()

    plt.savefig("dirac2.png")
    plt.close()


if __name__ == "__main__":
    print("Running 1D convolution example...")
    convolution_1d_example()

    print("\nRunning 2D convolution example...")
    convolution_2d_example()
