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

"""The network definition for discrete video tokenizer with VQ, LFQ, FSQ or ResidualFSQ. """
from collections import OrderedDict, namedtuple

import torch
import torch.nn.functional as F
from loguru import logger as logging
from torch import nn

from cosmos_predict1.tokenizer.modules import Decoder3DType, DiscreteQuantizer, Encoder3DType
from cosmos_predict1.tokenizer.modules.layers3d import CausalConv3d
from cosmos_predict1.tokenizer.modules.quantizers import InvQuantizerJit

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "quant_info", "pose_prediction"])


class CausalDiscreteVideoTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, embedding_dim: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "CausalDiscreteVideoTokenizer")
        self.embedding_dim = embedding_dim

        encoder_name = kwargs.get("encoder", Encoder3DType.BASE.name)
        self.encoder = Encoder3DType[encoder_name].value(z_channels=z_factor * z_channels, **kwargs)

        decoder_name = kwargs.get("decoder", Decoder3DType.BASE.name)
        self.decoder = Decoder3DType[decoder_name].value(z_channels=z_channels, **kwargs)

        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)

        quantizer_name = kwargs.get("quantizer", DiscreteQuantizer.RESFSQ.name)
        if quantizer_name == DiscreteQuantizer.VQ.name:
            assert "num_embeddings" in kwargs, f"`num_embeddings` must be provided for {quantizer_name}."
            kwargs.update(dict(embedding_dim=embedding_dim))
        elif quantizer_name == DiscreteQuantizer.LFQ.name:
            assert "codebook_size" in kwargs, f"`codebook_size` must be provided for {quantizer_name}."
            assert "codebook_dim" in kwargs, f"`codebook_dim` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.FSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.RESFSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}."
            assert "num_quantizers" in kwargs, f"`num_quantizers` must be provided for {quantizer_name}."
        self.quantizer = DiscreteQuantizer[quantizer_name].value(**kwargs)
        logging.info(f"{self.name} based on {quantizer_name}-VAE, with {kwargs}.")

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")

        self.predict_pose = kwargs.get("predict_pose", False)
        if self.predict_pose:
            self.num_pose_keypoints = kwargs.get("num_pose_keypoints", 176)
            self.pose_hidden_dim = kwargs.get("pose_hidden_dim", 512)
            self.temporal_compression = kwargs.get("temporal_compression", 8)

            self.spatial_attn = nn.Conv3d(embedding_dim, 1, 1)
            self.temporal_conv = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)

            self.pose_mlp = nn.Sequential(
                nn.Linear(embedding_dim, self.pose_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.pose_hidden_dim, self.pose_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.pose_hidden_dim, self.num_pose_keypoints * 3),
            )

            logging.info(f"Pose prediction enabled: {self.num_pose_keypoints} keypoints")

    def to(self, *args, **kwargs):
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(CausalDiscreteVideoTokenizer, self).to(*args, **kwargs)

    def encoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder),
                    ("quant_conv", self.quant_conv),
                    ("quantizer", self.quantizer),
                ]
            )
        )

    def decoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("inv_quant", InvQuantizerJit(self.quantizer)),
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
                ]
            )
        )

    def last_decoder_layer(self):
        return self.decoder.conv_out

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantizer(h)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def decode_code(self, code_b):
        quant_b = self.quantizer.indices_to_codes(code_b)
        quant_b = self.post_quant_conv(quant_b)
        return self.decoder(quant_b)

    def predict_pose_from_codes(self, quant_codes, original_num_frames=None):
        """Predict pose keypoints from quantized codes via spatial pooling + MLP."""
        if not self.predict_pose:
            return None

        B, C, T_compressed, H, W = quant_codes.shape

        pooled = quant_codes.mean(dim=[3, 4])  # (B, C, T')
        pooled = pooled.permute(0, 2, 1)  # (B, T', C)

        pose_flat = self.pose_mlp(pooled)  # (B, T', num_keypoints * 3)
        pose_pred = pose_flat.view(B, T_compressed, self.num_pose_keypoints, 3)

        if original_num_frames is not None and original_num_frames != T_compressed:
            pose_pred = pose_pred.view(B, T_compressed, -1).permute(0, 2, 1)
            pose_pred = F.interpolate(pose_pred, size=original_num_frames, mode='linear', align_corners=False)
            pose_pred = pose_pred.permute(0, 2, 1).view(B, original_num_frames, self.num_pose_keypoints, 3)

        return pose_pred

    def forward(self, input):
        B, C, T, H, W = input.shape
        quant_info, quant_codes, quant_loss = self.encode(input)
        reconstructions = self.decode(quant_codes)

        pose_prediction = self.predict_pose_from_codes(quant_codes, original_num_frames=T)

        if self.training:
            return dict(
                reconstructions=reconstructions,
                quant_loss=quant_loss,
                quant_info=quant_info,
                pose_prediction=pose_prediction,
            )
        return NetworkEval(
            reconstructions=reconstructions,
            quant_loss=quant_loss,
            quant_info=quant_info,
            pose_prediction=pose_prediction,
        )
