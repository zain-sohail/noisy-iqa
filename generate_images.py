import numpy as np
import torch

from helpers import plot_images


def generate_synthetic_arpes(size=(256, 256), rng=None):
    """
    Generate synthetic ARPES images with random bands and features.

    Args:
        size: Image dimensions (height, width)
        rng: Random number generator

    Returns:
        Normalized synthetic ARPES image
    """
    if rng is None:
        rng = np.random.default_rng()

    kx = np.linspace(-1, 1, size[1])
    E = np.linspace(-1, 1, size[0])
    Kx, E = np.meshgrid(kx, E)

    image = np.zeros(size)

    # Random number of bands (2-5)
    n_bands = rng.integers(2, 6)

    for _ in range(n_bands):
        # Randomly choose band type and parameters
        band_type = rng.choice(["parabolic", "linear", "sine"])
        amplitude = rng.uniform(0.3, 1.0)
        offset = rng.uniform(-0.5, 0.5)
        width = rng.uniform(0.01, 0.04)

        if band_type == "parabolic":
            curvature = rng.uniform(0.3, 0.7)
            band = curvature * Kx**2 + offset
        elif band_type == "linear":
            slope = rng.uniform(-1, 1)
            band = slope * Kx + offset
        else:  # sine
            freq = rng.uniform(1, 3)
            band = amplitude * np.sin(freq * np.pi * Kx) + offset

        intensity = amplitude * np.exp(-((E - band) ** 2) / width)
        image += intensity

    # Add random background gradient
    gradient = rng.uniform(0, 0.1) * (E + rng.uniform(-1, 1))
    image += gradient

    # Normalize
    image = (image - image.min()) / (image.max() - image.min())
    return image


def generate_dataset(n_images=40, seed=42, device="cpu"):
    """Generate a dataset of synthetic ARPES images."""
    rng = np.random.default_rng(seed)
    images = np.array([generate_synthetic_arpes(rng=rng) for _ in range(n_images)])
    plot_images(images)
    return torch.tensor(images, dtype=torch.float32, device=device).unsqueeze(1)
