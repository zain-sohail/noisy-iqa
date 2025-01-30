from dataclasses import dataclass

import pandas as pd
import piqa
from torch import Tensor


@dataclass
class MetricData:
    """Store image quality metrics and associated metadata."""

    ssim: list[float]
    ms_ssim: list[float]
    vsi: list[float]
    fsim: list[float]
    input_miv: float
    target_miv: float
    input_mean: list[float]
    target_mean: list[float]

    @classmethod
    def from_tensors(
        cls,
        ssim: Tensor,
        ms_ssim: Tensor,
        vsi: Tensor,
        fsim: Tensor,
        input_miv: Tensor,
        target_miv: Tensor,
        input_mean: Tensor,
        target_mean: Tensor,
    ) -> "MetricData":
        """Create a MetricData instance from PyTorch tensors."""
        return cls(
            ssim=ssim.cpu().numpy(),
            ms_ssim=ms_ssim.cpu().numpy(),
            vsi=vsi.cpu().numpy(),
            fsim=fsim.cpu().numpy(),
            input_miv=input_miv.cpu().numpy(),
            target_miv=target_miv.cpu().numpy(),
            input_mean=input_mean.cpu().numpy(),
            target_mean=target_mean.cpu().numpy(),
        )


class ComputeMetrics:
    """Compute image quality metrics using PIQA library."""

    def __init__(self, device):
        self.ssim = piqa.SSIM(n_channels=1, reduction="none").to(device)
        self.ms_ssim = piqa.MS_SSIM(n_channels=1, reduction="none").to(device)
        self.vsi = piqa.VSI(chromatic=False, reduction="none").to(device)
        self.fsim = piqa.FSIM(chromatic=False, reduction="none").to(device)

        self.metrics = []

    def _calculate_metrics(self, noisy_images, clean_images):
        """Calculate image quality metrics for a pair of noisy and clean images."""
        ssim_values = self.ssim(noisy_images, clean_images)
        ms_ssim_values = self.ms_ssim(noisy_images, clean_images)

        # some metrics require 3 color channel input so we repeat the single channels into 3
        noisy_images = noisy_images.repeat(1, 3, 1, 1)
        clean_images = clean_images.repeat(1, 3, 1, 1)
        vsi_values = self.vsi(noisy_images, clean_images)
        fsim_values = self.fsim(noisy_images, clean_images)
        return ssim_values, ms_ssim_values, vsi_values, fsim_values

    def run_analysis(self, max_ivs, mean_ivs, dataset):
        """Here, the analysis for metric performance with clean and noisy references is done."""
        max_idx = len(dataset) - 1

        for i in range(len(dataset)):
            noisy_batch = dataset[i]
            clean_batch = dataset[max_idx]

            # 1. Compare with the clean batch (index 20)
            metric_vals = self._calculate_metrics(noisy_batch, clean_batch)
            self.metrics.append(
                MetricData.from_tensors(
                    *metric_vals,
                    input_miv=max_ivs[i],
                    target_miv=max_ivs[max_idx],
                    input_mean=mean_ivs[i],
                    target_mean=mean_ivs[max_idx],
                ),
            )

            # 2. Compare with all less noisy batches (indices > i up to 19)
            for j in range(i + 1, max_idx):
                less_noisy_batch = dataset[j]
                metric_vals = self._calculate_metrics(noisy_batch, less_noisy_batch)

                self.metrics.append(
                    MetricData.from_tensors(
                        *metric_vals,
                        input_miv=max_ivs[i],
                        target_miv=max_ivs[j],
                        input_mean=mean_ivs[i],
                        target_mean=mean_ivs[j],
                    ),
                )

    def get_df(self):
        """Convert the metrics data to a long-form pandas DataFrame for plotting with Seaborn."""
        df = pd.DataFrame([m.__dict__ for m in self.metrics])
        df = df.explode(["ssim", "ms_ssim", "vsi", "fsim", "input_mean", "target_mean"])

        # melt the dataframe
        df = df.melt(
            id_vars=["input_miv", "target_miv", "input_mean", "target_mean"],
            var_name="metric",
            value_name="value",
        )
        df = df.astype(
            {
                "input_miv": "float32",
                "target_miv": "float32",
                "input_mean": "float32",
                "target_mean": "float32",
                "value": "float32",
            },
        )
        return df
