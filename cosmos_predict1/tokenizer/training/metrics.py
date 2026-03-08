# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The combined loss functions for continuous-space tokenizers training."""

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim

from cosmos_predict1.tokenizer.modules.utils import time2batch
from cosmos_predict1.tokenizer.training.datasets.utils import INPUT_KEY, MASK_KEY
from cosmos_predict1.utils.lazy_config import instantiate

_VALID_METRIC_NAMES = ["PSNR", "SSIM", "CodeUsage"]
_UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
_FLOAT32_EPS = torch.finfo(torch.float32).eps
_RECONSTRUCTION = "reconstructions"
_QUANT_INFO = "quant_info"


class TokenizerMetric(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.metric_modules = nn.ModuleDict()
        for key in _VALID_METRIC_NAMES:
            self.metric_modules[key] = instantiate(getattr(config, key)) if hasattr(config, key) else NULLMetric()

    def forward(
        self, inputs: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        metric = dict()
        for _, module in self.metric_modules.items():
            metric.update(module(inputs, output_batch, iteration))
        return dict(metric=metric)


class NULLMetric(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, inputs: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        return dict()


class PSNRMetric(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, inputs: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        reconstructions = output_batch[_RECONSTRUCTION]
        input_images = inputs[INPUT_KEY]
        weights = inputs[MASK_KEY]

        if input_images.ndim == 5:
            input_images, _ = time2batch(input_images)
            reconstructions, _ = time2batch(reconstructions)
            weights, _ = time2batch(weights)

        # Normalize to uint8 [0..255] range.
        true_image = (input_images.to(torch.float32) + 1) / 2
        pred_image = (reconstructions.to(torch.float32) + 1) / 2
        true_image = (true_image * _UINT8_MAX_F + 0.5).to(torch.uint8)
        pred_image = (pred_image * _UINT8_MAX_F + 0.5).to(torch.uint8)

        # Calculate PSNR using masked MSE over interest regions (mask > 0.2)
        true_image = true_image.to(torch.float32)
        pred_image = pred_image.to(torch.float32)
        weights_binary = (weights > 0.2).float()
        squared_error = (true_image - pred_image) ** 2
        mse = (squared_error * weights_binary).sum(dim=(1, 2, 3)) / (weights_binary.sum(dim=(1, 2, 3)) + _FLOAT32_EPS)
        psnr = 10 * torch.log10(_UINT8_MAX_F**2 / (mse + _FLOAT32_EPS))
        return dict(PSNR=torch.mean(psnr))


class SSIMMetric(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, inputs: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        reconstructions = output_batch[_RECONSTRUCTION]
        input_images = inputs[INPUT_KEY]
        weights = inputs[MASK_KEY]

        if input_images.ndim == 5:
            input_images, _ = time2batch(input_images)
            reconstructions, _ = time2batch(reconstructions)
            weights, _ = time2batch(weights)

        # Normalize to uint8 [0..255] range.
        true_image = (input_images.to(torch.float32) + 1) / 2
        pred_image = (reconstructions.to(torch.float32) + 1) / 2
        true_image = (true_image * _UINT8_MAX_F + 0.5).to(torch.uint8)
        pred_image = (pred_image * _UINT8_MAX_F + 0.5).to(torch.uint8)

        # Move tensors to CPU and convert to numpy arrays
        true_image_np = true_image.permute(0, 2, 3, 1).cpu().numpy()
        pred_image_np = pred_image.permute(0, 2, 3, 1).cpu().numpy()
        weights_np = weights.to(torch.float32).permute(0, 2, 3, 1).cpu().numpy()

        # Calculate masked SSIM for each image in the batch
        ssim_values = []
        for true_img, pred_img, weight_mask in zip(true_image_np, pred_image_np, weights_np):
            ssim_map = ssim(
                true_img, pred_img,
                data_range=_UINT8_MAX_F,
                multichannel=True,
                channel_axis=-1,
                full=True,
            )[1]  # full SSIM map, shape (H, W, C)
            ssim_map_2d = ssim_map.mean(axis=-1)  # (H, W)
            mask_2d = (weight_mask[..., 0] > 0.2).astype(np.float32)
            mask_sum = mask_2d.sum()
            if mask_sum > 0:
                ssim_values.append((ssim_map_2d * mask_2d).sum() / mask_sum)

        ssim_mean = np.mean(ssim_values) if ssim_values else 0.0
        return dict(SSIM=torch.tensor(ssim_mean, dtype=torch.float32, device=input_images.device))


class CodeUsageMetric(torch.nn.Module):
    """
    Calculate the perplexity of codebook usage (only for discrete tokenizers)

    :param codebook_indices: Tensor of codebook indices (quant_info)
    :param codebook_size: The total number of codebook entries
    :return: Perplexity of the codebook usage
    """

    def __init__(self, codebook_size: int) -> None:
        super().__init__()
        self.codebook_size = codebook_size

    def forward(
        self, inputs: dict[str, torch.Tensor], output_batch: dict[str, torch.Tensor], iteration: int
    ) -> dict[str, torch.Tensor]:
        code_indices = output_batch[_QUANT_INFO]
        usage_counts = torch.bincount(code_indices.flatten().int(), minlength=self.codebook_size)
        total_usage = usage_counts.sum().float()
        usage_probs = usage_counts.float() / total_usage
        entropy = -torch.sum(usage_probs * torch.log(usage_probs + _FLOAT32_EPS))
        perplexity = torch.exp(entropy)
        return dict(CodeUsage=torch.tensor(perplexity, dtype=torch.float32, device=code_indices.device))
