from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_device() -> torch.device:
    """Set the computation device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plot_images(
    images: np.ndarray,
    num: int = 4,
    figsize: Optional[tuple[int, int]] = None,
):
    """Plot a grid of ARPES images."""
    if figsize is None:
        figsize = (4 * num, num)
    fig, axs = plt.subplots(1, num, figsize=figsize)
    for i in range(num):
        axs[i].imshow(images[i], cmap="viridis")
        axs[i].axis("off")
    plt.tight_layout(pad=0.5)
    plt.show()


def normalize_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:  # Shape of tensor: [40, 1, 256, 256]
    """Normalize tensor values to [0, 1] range."""
    min_vals = tensor.amin(dim=(-1, -2), keepdim=True)  # Shape: [40, 1, 1, 1]
    max_vals = tensor.amax(dim=(-1, -2), keepdim=True)  # Shape: [40, 1, 1, 1]

    # Avoid division by zero (if max == min)
    epsilon = 1e-8
    tensor_normalized = (tensor - min_vals) / (max_vals - min_vals + epsilon)

    # Clamp to ensure values are within [0, 1] (optional, but safe)
    return torch.clamp(tensor_normalized, 0.0, 1.0)


def generate_noise_levels(
    n_levels: int = 20,
    lowest: int = -2,
    highest: int = 3,
) -> torch.Tensor:
    """Generate maximum intensity values for Poisson noise. Higher intensity means lesser noise.
    This is approximately the lambda parameter of the Poisson process"""
    return torch.tensor(
        np.logspace(lowest, highest, n_levels),
        device=set_device(),
        dtype=torch.float32,
    )


def apply_poisson_noise(
    max_ivs: torch.Tensor,
    dataset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Poisson noise at different intensity levels."""
    dataset_expanded = dataset.unsqueeze(0)
    max_ivs_expanded = max_ivs.view(-1, 1, 1, 1, 1)
    scaled_images = dataset_expanded * max_ivs_expanded

    # Compute mean values of the scaled images
    mean_ivs = torch.mean(scaled_images, dim=(2, 3, 4))  # shape [N_MIV, N_IMG]

    # add clean image mean values to the mean values tensor (use MIV of an order higher)
    MIV_MAX = max_ivs.max() * 10
    clean_images = dataset_expanded * MIV_MAX
    clean_mean_ivs = torch.mean(clean_images, dim=(2, 3, 4))  # shape [1, N_IMG]
    mean_ivs = torch.cat((mean_ivs, clean_mean_ivs), dim=0)  # shape [N_MIV+1, N_IMG]
    max_ivs = torch.cat(
        (max_ivs, torch.tensor([MIV_MAX], device=set_device(), dtype=torch.float32)),
        dim=0,
    )

    poisson_counts = torch.poisson(scaled_images)  # shape [N_MIV, N_IMG, 1, 256, 256]
    poisson_counts = normalize_tensor(poisson_counts)

    # Add clean image (dataset) to the poisson_counts tensor
    dataset = torch.cat(
        [poisson_counts, dataset_expanded],
        dim=0,
    )  # shape [N_MIV+1, N_IMG, 1, 256, 256]
    return max_ivs, mean_ivs, dataset
