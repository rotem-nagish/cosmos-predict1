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
from loguru import logger as logging
from torch import nn
import torch.nn.functional as F

from cosmos_predict1.tokenizer.modules import Decoder3DType, DiscreteQuantizer, Encoder3DType
from cosmos_predict1.tokenizer.modules.layers3d import CausalConv3d
from cosmos_predict1.tokenizer.modules.quantizers import InvQuantizerJit

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "quant_info", "pose_prediction"])


_SINUSOIDAL_INTERNAL_DIM = 96  # fixed internal dim (divisible by 6 = 2*3 axes)


def _sinusoidal_embed(coords: torch.Tensor) -> torch.Tensor:
    """Sinusoidal positional encoding for normalised (t, y, x) coordinates.

    Args:
        coords: [..., 3] values in [0, 1].

    Returns:
        [..., _SINUSOIDAL_INTERNAL_DIM] sinusoidal embedding.
    """
    D = coords.shape[-1]  # 3
    dim_per_axis = _SINUSOIDAL_INTERNAL_DIM // D  # 32

    freqs = torch.arange(dim_per_axis // 2, device=coords.device, dtype=torch.float32)
    freqs = 1.0 / (10000 ** (2 * freqs / dim_per_axis))  # [16]

    # coords: [..., D, 1] * [16] → [..., D, 16]
    angles = coords.unsqueeze(-1) * freqs * 2 * torch.pi
    # [..., D, 32] → [..., 96]
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return emb.reshape(*coords.shape[:-1], _SINUSOIDAL_INTERNAL_DIM)


class PartFusionAttention(nn.Module):
    """Global cross-attention fusion of body-part latent codes.

    Each spatial-temporal token in the original video attends over *all* tokens
    from the part crops (face, left hand, right hand).  This lets face tokens
    in the original attend to the high-resolution face-crop encoding, hand
    tokens attend to the hand-crop encoding, etc.

    Spatial positional encoding: both Q (original) and KV (crop) tokens get
    sinusoidal positional embeddings based on their real-world (t, y, x)
    coordinates in the original frame.  For the original, these are a regular
    grid in [0,1].  For crops, each token's position is mapped through the
    bounding box: (t/T, y1 + h/H*(y2-y1), x1 + w/W*(x2-x1)).  This way
    the dot product between a face token in the original and the corresponding
    face-crop token is high because their positional encodings match.

    Cost: O(S_orig * S_parts) where S_parts = (num_parts-1) * S_orig.
    With typical bottleneck dims (S≈1792, 3 crops) this is ~10M entries per
    head — well within flash-attention budget.

    Initialised so that the output equals the original-video code at the start
    of training (gate starting at 0, out_proj keeps default init so gradient
    through gate is nonzero).
    """

    def __init__(
        self,
        channels: int,
        num_parts: int = 4,
        num_heads: int = 4,
        spatial_prior: bool = False,
        gate_init: float = 0.25,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_parts = num_parts
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.spatial_prior = spatial_prior

        # Learnable part identity embeddings (one per crop, added to KV)
        self.part_embed = nn.Parameter(torch.randn(num_parts - 1, 1, channels) * 0.02)

        # MLP to project sinusoidal pos encoding into the model's space.
        # Default (kaiming) init so positional signal is active from the start.
        # Previous zero-init of the last layer was found to prevent pos_proj
        # from ever learning (gradient starvation due to gate × zero-pos chain).
        self.pos_proj = nn.Sequential(
            nn.Linear(_SINUSOIDAL_INTERNAL_DIM, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        self.norm_q = nn.LayerNorm(channels)
        self.norm_kv = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # Learnable gate — initialised at gate_init so that gradients
        # flow more strongly through gate * out_proj(...) early in training.
        self.gate = nn.Parameter(torch.full((1,), gate_init))

        # Spatial prior: learnable scale for the bbox-overlap logit bias.
        # Starts at 1.0; the model can grow/shrink it as needed.
        if spatial_prior:
            self.prior_scale = nn.Parameter(torch.ones(1))

    def _make_grid_coords(self, T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Create normalised (t, y, x) coordinates for a regular grid.

        Returns: [T*H*W, 3] with values in [0, 1].
        """
        t = torch.linspace(0, 1, T, device=device)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        grid = torch.stack(torch.meshgrid(t, y, x, indexing="ij"), dim=-1)  # [T,H,W,3]
        return grid.reshape(-1, 3)  # [S, 3]

    def _make_crop_coords(
        self, T: int, H: int, W: int, bboxes: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Map crop grid positions to original-frame coordinates using BBs.

        Args:
            T, H, W: bottleneck spatial dims.
            bboxes: [B, T_video, num_crops, 4] normalised (x1, y1, x2, y2).
                    T_video is the *input* temporal dim; we subsample to T.
                    -1 values indicate missing detections.
        Returns:
            [B, num_crops, S, 3] — (t, y, x) in [0, 1] for each crop token.
        """
        B = bboxes.shape[0]
        num_crops = bboxes.shape[2]
        T_video = bboxes.shape[1]

        # Subsample bboxes to match bottleneck temporal dim
        # (temporal_compression typically 8, so T_video=49 → T=7)
        t_indices = torch.linspace(0, T_video - 1, T, device=device).long()
        bboxes = bboxes[:, t_indices]  # [B, T, num_crops, 4]

        # Normalised local grid for the crop
        t_lin = torch.linspace(0, 1, T, device=device)
        y_lin = torch.linspace(0, 1, H, device=device)
        x_lin = torch.linspace(0, 1, W, device=device)

        # For each crop, remap local (y, x) through the bbox
        all_coords = []
        for ci in range(num_crops):
            bb = bboxes[:, :, ci]  # [B, T, 4] — (x1, y1, x2, y2)
            x1 = bb[:, :, 0]  # [B, T]
            y1 = bb[:, :, 1]
            x2 = bb[:, :, 2]
            y2 = bb[:, :, 3]

            # Handle missing detections: use center crop (0.25..0.75)
            missing = (x1 < 0)
            x1 = torch.where(missing, torch.tensor(0.25, device=device), x1)
            y1 = torch.where(missing, torch.tensor(0.25, device=device), y1)
            x2 = torch.where(missing, torch.tensor(0.75, device=device), x2)
            y2 = torch.where(missing, torch.tensor(0.75, device=device), y2)

            # Map local coords to original frame coords
            # t_global: same as original (temporal dim is shared)
            # y_global = y1[b,t] + y_lin[h] * (y2[b,t] - y1[b,t])
            # x_global = x1[b,t] + x_lin[w] * (x2[b,t] - x1[b,t])

            # [B, T, H] and [B, T, W]
            y_global = y1[:, :, None] + y_lin[None, None, :] * (y2 - y1)[:, :, None]
            x_global = x1[:, :, None] + x_lin[None, None, :] * (x2 - x1)[:, :, None]

            # Build full coordinate grid [B, T, H, W, 3]
            t_global = t_lin[None, :, None, None].expand(B, T, H, W)
            y_global = y_global[:, :, :, None].expand(B, T, H, W)
            x_global = x_global[:, :, None, :].expand(B, T, H, W)
            coords = torch.stack([t_global, y_global, x_global], dim=-1)  # [B,T,H,W,3]
            all_coords.append(coords.reshape(B, T * H * W, 3))

        return torch.stack(all_coords, dim=1)  # [B, num_crops, S, 3]

    def _compute_spatial_prior(
        self,
        orig_coords: torch.Tensor,
        crop_coords: torch.Tensor,
        B: int,
        S: int,
        P: int,
    ) -> torch.Tensor:
        """Compute a bbox-overlap logit bias for the spatial prior.

        For each original token s and each crop p, computes a soft membership
        score (sigmoid containment) indicating whether token s lies within the
        spatial extent of crop p.  This score is broadcast uniformly over all S
        tokens of crop p, giving additive attention logit bias [B, 1, S, (P-1)*S].

        At init (prior_scale=1.0) the bias is in roughly [0, 1], which is large
        enough to meaningfully shift attention toward the spatially corresponding
        crop at the start of training.  The model can scale it up or down.

        Args:
            orig_coords: [S, 3] (t, y, x) normalised grid coords for original.
            crop_coords: [B, P-1, S, 3] (t, y, x) in original frame for crops.
            B, S, P: batch size, sequence length, num_parts (including original).
        Returns:
            [B, 1, S, (P-1)*S] float tensor of logit biases.
        """
        # Derive crop bbox extents from the min/max of mapped crop coordinates.
        cy = crop_coords[:, :, :, 1]  # [B, P-1, S]
        cx = crop_coords[:, :, :, 2]  # [B, P-1, S]
        y1 = cy.min(dim=-1).values    # [B, P-1]
        y2 = cy.max(dim=-1).values
        x1 = cx.min(dim=-1).values    # [B, P-1]
        x2 = cx.max(dim=-1).values

        oy = orig_coords[:, 1]  # [S]
        ox = orig_coords[:, 2]  # [S]

        # Soft containment via logistic: sigmoid(k*(coord - lo)) * sigmoid(k*(hi - coord))
        # → 1 inside bbox, 0 outside, smooth transition over ~1/k of the unit square.
        # With 16×16 grid the token pitch is 1/15 ≈ 0.067; k=30 gives transition ≈ 2 tokens.
        k = 30.0
        in_y = (torch.sigmoid(k * (oy[None, :, None] - y1[:, None, :])) *   # [B, S, P-1]
                torch.sigmoid(k * (y2[:, None, :] - oy[None, :, None])))
        in_x = (torch.sigmoid(k * (ox[None, :, None] - x1[:, None, :])) *   # [B, S, P-1]
                torch.sigmoid(k * (x2[:, None, :] - ox[None, :, None])))
        membership = in_y * in_x                                              # [B, S, P-1]

        # Broadcast uniformly over all S tokens within each crop:
        # [B, S, P-1, 1] → [B, S, P-1, S] → [B, S, (P-1)*S]
        prior = (membership[:, :, :, None]
                 .expand(-1, -1, -1, S)
                 .reshape(B, S, (P - 1) * S))

        return (self.prior_scale * prior).unsqueeze(1)  # [B, 1, S, (P-1)*S]

    def forward(
        self, parts: torch.Tensor, crop_bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            parts: [B, num_parts, C, T, H, W] — stacked part latents.
                   parts[:, 0] is the original video.
                   parts[:, 1:] are the crops (face, left hand, right hand).
            crop_bboxes: [B, T_video, num_crops, 4] normalised BBs per frame,
                   or None to skip spatial positional encoding and spatial prior.
        Returns:
            Fused tensor [B, C, T, H, W].
        """
        B, P, C, T, H, W = parts.shape
        S = T * H * W
        device = parts.device
        orig_dtype = parts.dtype

        # Cast bf16 input to fp32 to match this module's precision.
        # The fusion module stays fp32 so that optimizer updates (lr ≈ 1e-4)
        # are not lost to bf16 quantization.
        parts = parts.float()

        # Query: original video tokens
        q_in = parts[:, 0].reshape(B, C, S).permute(0, 2, 1)      # [B, S, C]

        # KV: all crop tokens concatenated into one long sequence
        crops = parts[:, 1:].reshape(B, P - 1, C, S)               # [B, P-1, C, S]
        crops = crops.permute(0, 1, 3, 2)                           # [B, P-1, S, C]

        # Add per-part identity embeddings
        crops = crops + self.part_embed[None, :, :, :]              # [B, P-1, S, C]

        # Spatial coordinates + positional encoding + spatial prior (when bboxes available)
        attn_bias = None
        if crop_bboxes is not None:
            orig_coords = self._make_grid_coords(T, H, W, device)             # [S, 3]
            crop_coords = self._make_crop_coords(T, H, W, crop_bboxes, device)  # [B, P-1, S, 3]

            orig_pos = self.pos_proj(_sinusoidal_embed(orig_coords))           # [S, C]
            q_in = q_in + orig_pos[None]                                       # [B, S, C]
            crop_pos = self.pos_proj(_sinusoidal_embed(crop_coords))           # [B, P-1, S, C]
            crops = crops + crop_pos                                           # [B, P-1, S, C]

            if self.spatial_prior:
                attn_bias = self._compute_spatial_prior(
                    orig_coords, crop_coords, B, S, P
                )                                                              # [B, 1, S, (P-1)*S]

        kv_in = crops.reshape(B, (P - 1) * S, C)                   # [B, (P-1)*S, C]

        # Project
        q = self.q_proj(self.norm_q(q_in))                         # [B, S, C]
        kv_normed = self.norm_kv(kv_in)                             # [B, (P-1)*S, C]
        k = self.k_proj(kv_normed)                                  # [B, (P-1)*S, C]
        v = self.v_proj(kv_normed)                                  # [B, (P-1)*S, C]

        # Reshape for multi-head attention
        hd = self.head_dim
        nH = self.num_heads
        S_kv = (P - 1) * S
        q = q.reshape(B, S, nH, hd).transpose(1, 2)               # [B, nH, S, hd]
        k = k.reshape(B, S_kv, nH, hd).transpose(1, 2)            # [B, nH, S_kv, hd]
        v = v.reshape(B, S_kv, nH, hd).transpose(1, 2)            # [B, nH, S_kv, hd]

        # Attention (attn_bias=None falls back to FlashAttention; with bias uses math/mem-efficient)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)  # [B, nH, S, hd]
        out = out.transpose(1, 2).reshape(B, S, C)                 # [B, S, C]
        out = self.out_proj(out)

        # Residual with gating
        fused = q_in + self.gate * out
        return fused.permute(0, 2, 1).reshape(B, C, T, H, W).to(orig_dtype)


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

        # Part fusion: encode body-part streams separately, fuse via cross-attention.
        # part_fusion_mode:
        #   "bottleneck" — single fusion after the full encoder (lightweight)
        #   "multilevel" — fusion at every encoder level + mid (richer but heavier)
        self.part_fusion = kwargs.get("part_fusion", False)
        self.num_parts = kwargs.get("num_parts", 4)
        self.part_fusion_mode = kwargs.get("part_fusion_mode", "bottleneck")
        if self.part_fusion:
            channels = kwargs.get("channels", 128)
            channels_mult = kwargs.get("channels_mult", [2, 4, 4])
            num_heads = kwargs.get("part_fusion_heads", 4)
            spatial_prior = kwargs.get("part_fusion_spatial_prior", False)
            gate_init = kwargs.get("part_fusion_gate", 0.25)

            # Bottleneck fusion (always created — matches existing checkpoints
            # with "part_fusion_attn" keys)
            self.part_fusion_attn = PartFusionAttention(
                z_factor * z_channels, self.num_parts, num_heads,
                spatial_prior=spatial_prior, gate_init=gate_init,
            )

            if self.part_fusion_mode == "multilevel":
                # Additional per-level fusion modules
                self.part_fusion_levels = nn.ModuleList()
                for mult in channels_mult:
                    self.part_fusion_levels.append(
                        PartFusionAttention(channels * mult, self.num_parts, num_heads,
                                            spatial_prior=spatial_prior, gate_init=gate_init)
                    )
                self.part_fusion_mid = PartFusionAttention(
                    channels * channels_mult[-1], self.num_parts, num_heads,
                    spatial_prior=spatial_prior, gate_init=gate_init,
                )
                logging.info(
                    f"Part fusion enabled (multilevel): {self.num_parts} parts, "
                    f"fusion at {len(channels_mult)} encoder levels + mid + bottleneck"
                )
            else:
                logging.info(
                    f"Part fusion enabled (bottleneck): {self.num_parts} parts, "
                    f"cross-attention over {z_factor * z_channels} channels"
                )

        # Pose prediction MLP
        self.predict_pose = kwargs.get("predict_pose", False)
        if self.predict_pose:
            self.num_pose_keypoints = kwargs.get("num_pose_keypoints", 176)  # Holistic: 33+468+21+21 -> reduce+hide_legs
            self.pose_hidden_dim = kwargs.get("pose_hidden_dim", 512)
            self.temporal_compression = kwargs.get("temporal_compression", 8)

            self.spatial_attn = nn.Conv3d(embedding_dim, 1, 1)
            self.temporal_conv = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
            self.temporal_upsampler = lambda x, T: F.interpolate(x, T, mode="linear", align_corners=False)

            # MLP to predict pose from quantized codes
            # Input: (B, embedding_dim, T', H', W') -> spatially pool -> (B, embedding_dim, T')
            # Output: (B, T', num_keypoints, 3)
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

        # Auxiliary task heads (hand location, hand orientation, etc.)
        self.aux_tasks = kwargs.get("aux_tasks", [])
        aux_hidden = 256

        if "hand_location" in self.aux_tasks:
            self.hand_loc_spatial_attn = nn.Conv3d(embedding_dim, 1, 1)
            self.hand_loc_temporal_conv = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
            self.hand_loc_mlp = nn.Sequential(
                nn.Linear(embedding_dim, aux_hidden),
                nn.ReLU(),
                nn.Linear(aux_hidden, 2 * 4),  # 2 hands × (x1, y1, x2, y2)
                nn.Sigmoid(),  # BBs are normalised to [0, 1]
            )
            logging.info("Aux task enabled: hand_location")

        if "hand_orientation" in self.aux_tasks:
            self.hand_orient_spatial_attn = nn.Conv3d(embedding_dim, 1, 1)
            self.hand_orient_temporal_conv = nn.Conv1d(embedding_dim, embedding_dim, 3, padding=1)
            self.hand_orient_mlp = nn.Sequential(
                nn.Linear(embedding_dim, aux_hidden),
                nn.ReLU(),
                nn.Linear(aux_hidden, 2 * 6),  # 2 hands × 6 orientation classes
            )
            logging.info("Aux task enabled: hand_orientation")

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")

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

    def encode(self, x, crop_bboxes=None):
        if self.part_fusion:
            return self._encode_part_fusion(x, crop_bboxes=crop_bboxes)
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantizer(h)

    def _encode_part_fusion(self, x, crop_bboxes=None):
        """Encode a multi-part input [B, C*num_parts, T, H, W]."""
        if self.part_fusion_mode == "multilevel":
            return self._encode_part_fusion_multilevel(x, crop_bboxes=crop_bboxes)
        return self._encode_part_fusion_bottleneck(x, crop_bboxes=crop_bboxes)

    def _encode_part_fusion_bottleneck(self, x, crop_bboxes=None):
        """Bottleneck mode: run the full encoder on all parts as a batch,
        then fuse once before quantisation."""
        B, C_total, T, H, W = x.shape
        P = self.num_parts
        C = C_total // P

        # Reshape to [B*P, 3, T, H, W] and encode all parts in one batched call
        parts = x.reshape(B, P, C, T, H, W).reshape(B * P, C, T, H, W)
        h = self.encoder(parts)                            # [B*P, z_ch, T', H', W']
        _, Cz, Tp, Hp, Wp = h.shape
        h = h.reshape(B, P, Cz, Tp, Hp, Wp)               # [B, P, z_ch, T', H', W']

        # Cross-attention fusion at the bottleneck (before quant_conv)
        fused = self.part_fusion_attn(h, crop_bboxes=crop_bboxes)

        fused = self.quant_conv(fused)
        return self.quantizer(fused)

    def _encode_part_fusion_multilevel(self, x, crop_bboxes=None):
        """Multilevel mode: replay the encoder's forward pass, injecting
        cross-attention fusion after every downsampling level and the mid
        bottleneck.  Only the original stream (part 0) is updated; the
        face/hand streams continue unchanged as context providers."""
        B, C_total, T, H, W = x.shape
        P = self.num_parts
        C = C_total // P
        enc = self.encoder

        # [B*P, 3, T, H, W]
        h = x.reshape(B, P, C, T, H, W).reshape(B * P, C, T, H, W)

        # Patcher + conv_in
        h = enc.patcher3d(h)
        h = enc.conv_in(h)

        # Downsampling levels with per-level fusion
        for i_level in range(enc.num_resolutions):
            for i_block in range(enc.num_res_blocks):
                h = enc.down[i_level].block[i_block](h)
                if len(enc.down[i_level].attn) > 0:
                    h = enc.down[i_level].attn[i_block](h)

            # --- Part fusion: original attends to all parts ---
            # Multilevel fusion modules don't use spatial pos encoding
            # (spatial dims at intermediate levels don't match the BB coords
            # which are normalised to the final bottleneck resolution).
            h = self._fuse_parts(h, self.part_fusion_levels[i_level], B, P)

            if i_level != enc.num_resolutions - 1:
                h = enc.down[i_level].downsample(h)

        # Middle
        h = enc.mid.block_1(h)
        h = enc.mid.attn_1(h)
        h = enc.mid.block_2(h)

        # --- Part fusion at mid bottleneck ---
        h = self._fuse_parts(h, self.part_fusion_mid, B, P)

        # End — run all parts through norm_out + conv_out so the final
        # bottleneck fusion (part_fusion_attn) operates at the same point
        # as in bottleneck-only mode, preserving checkpoint compatibility.
        h = enc.norm_out(h)
        h = h * torch.sigmoid(h)  # SiLU
        h = enc.conv_out(h)

        # Final bottleneck fusion (same as bottleneck-only mode)
        _, Cz, Tp, Hp, Wp = h.shape
        h = h.reshape(B, P, Cz, Tp, Hp, Wp)
        h = self.part_fusion_attn(h, crop_bboxes=crop_bboxes)

        h = self.quant_conv(h)
        return self.quantizer(h)

    @staticmethod
    def _fuse_parts(h, fusion_attn, B, P):
        """Apply part fusion: update part 0 (original) via cross-attention,
        keep other parts unchanged, return re-batched tensor."""
        _, Cl, Tl, Hl, Wl = h.shape
        h_parts = h.reshape(B, P, Cl, Tl, Hl, Wl)         # [B, P, C, T, H, W]
        h_fused = fusion_attn(h_parts)                      # [B, C, T, H, W]
        # Replace part 0 with fused, keep parts 1..P-1 unchanged
        h_parts = torch.cat([h_fused.unsqueeze(1), h_parts[:, 1:]], dim=1)
        return h_parts.reshape(B * P, Cl, Tl, Hl, Wl)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def decode_code(self, code_b):
        quant_b = self.quantizer.indices_to_codes(code_b)
        quant_b = self.post_quant_conv(quant_b)
        return self.decoder(quant_b)

    def predict_pose_from_codes(self, quant_codes, original_num_frames=None):
        """
        Predict pose from quantized codes.

        Args:
            quant_codes: Quantized codes of shape (B, embedding_dim, T', H', W')
            original_num_frames: Original number of frames T (before temporal compression)

        Returns:
            Predicted pose of shape (B, T, num_keypoints, 3) where T = original_num_frames or T' if not provided
        """
        if not self.predict_pose:
            return None

        B, C, T, H, W = quant_codes.shape

        # --- learned spatial attention ---
        # (B,C,T,H,W) -> (B,T,C)
        attn = self.spatial_attn(quant_codes)  # 1x1x1 conv -> (B,1,T,H,W)
        attn = attn.flatten(-2)  # (B,1,T,H*W)
        attn = attn.softmax(dim=-1)
        attn = attn.view(B, 1, T, H, W)
        pooled = (quant_codes * attn).sum(dim=(-2, -1)).permute(0, 2, 1)

        # --- temporal context ---
        # (B,T,C) -> (B,T,C)
        pooled = self.temporal_conv(pooled.transpose(1, 2)).transpose(1, 2)

        # --- pose regression ---
        pose = self.pose_mlp(pooled).view(B, T, self.num_pose_keypoints, 3)

        # --- learned temporal upsampling ---
        if original_num_frames is not None and original_num_frames != T:
            pose = pose.view(B, T, -1).transpose(1, 2)
            pose = self.temporal_upsampler(pose, original_num_frames)
            pose = pose.transpose(1, 2).view(B, original_num_frames, self.num_pose_keypoints, 3)

        return pose

    def predict_pose_from_codes_naive(self, quant_codes, original_num_frames=None):
        """
        Predict pose from quantized codes.

        Args:
            quant_codes: Quantized codes of shape (B, embedding_dim, T', H', W')
            original_num_frames: Original number of frames T (before temporal compression)

        Returns:
            Predicted pose of shape (B, T, num_keypoints, 3) where T = original_num_frames or T' if not provided
        """
        if not self.predict_pose:
            return None

        B, C, T_compressed, H, W = quant_codes.shape

        # Spatially pool: (B, C, T', H, W) -> (B, C, T')
        # Use global average pooling over spatial dimensions
        pooled = quant_codes.mean(dim=[3, 4])  # (B, C, T')
        # TODO is pooling too aggressive?
        # Transpose to (B, T', C) for MLP
        pooled = pooled.permute(0, 2, 1)  # (B, T', C)

        # Apply MLP: (B, T', C) -> (B, T', num_keypoints * 3)
        pose_flat = self.pose_mlp(pooled)  # (B, T', num_keypoints * 3)

        # Reshape to (B, T', num_keypoints, 3)
        pose_pred = pose_flat.view(B, T_compressed, self.num_pose_keypoints, 3)

        # Upsample to original temporal resolution if provided
        if original_num_frames is not None and original_num_frames != T_compressed:
            # TODO is this strategy too naive?
            # Reshape for interpolation: (B, T', num_keypoints, 3) -> (B, num_keypoints * 3, T')
            pose_pred = pose_pred.view(B, T_compressed, -1).permute(0, 2, 1)
            # Interpolate: (B, num_keypoints * 3, T') -> (B, num_keypoints * 3, T)
            pose_pred = F.interpolate(pose_pred, size=original_num_frames, mode='linear', align_corners=False)
            # Reshape back: (B, num_keypoints * 3, T) -> (B, T, num_keypoints, 3)
            pose_pred = pose_pred.permute(0, 2, 1).view(B, original_num_frames, self.num_pose_keypoints, 3)

        return pose_pred


    def _pool_and_predict(self, quant_codes, spatial_attn, temporal_conv, mlp, original_T):
        """Shared spatial-pool → temporal-conv → MLP → temporal-upsample pattern."""
        B, C, T, H, W = quant_codes.shape
        attn = spatial_attn(quant_codes).flatten(-2).softmax(dim=-1).view(B, 1, T, H, W)
        pooled = (quant_codes * attn).sum(dim=(-2, -1)).permute(0, 2, 1)  # [B, T, C]
        pooled = temporal_conv(pooled.transpose(1, 2)).transpose(1, 2)     # [B, T, C]
        out = mlp(pooled)                                                   # [B, T, D]
        if original_T is not None and original_T != T:
            out = F.interpolate(out.transpose(1, 2), original_T, mode="linear", align_corners=False).transpose(1, 2)
        return out

    def forward(self, input, mask=None, crop_bboxes=None):
        B, C, T, H, W = input.shape
        # For part_fusion the input has C = 3 * num_parts; the decoder always
        # reconstructs a 3-channel video.  Pass the full multi-channel tensor
        # to encode() which handles the split internally.
        quant_info, quant_codes, quant_loss = self.encode(input, crop_bboxes=crop_bboxes)
        reconstructions = self.decode(quant_codes)

        pose_prediction = self.predict_pose_from_codes(quant_codes, original_num_frames=T)

        # Use empty tensor instead of None for JIT compatibility (JIT traces eval mode
        # and requires all namedtuple fields to be tensors).
        if pose_prediction is None:
            pose_prediction = torch.empty(0)

        # Aux task predictions
        aux_outputs = {}
        if "hand_location" in self.aux_tasks:
            raw = self._pool_and_predict(
                quant_codes, self.hand_loc_spatial_attn,
                self.hand_loc_temporal_conv, self.hand_loc_mlp, T
            )
            aux_outputs["hand_location"] = raw.view(B, T, 2, 4)

        if "hand_orientation" in self.aux_tasks:
            raw = self._pool_and_predict(
                quant_codes, self.hand_orient_spatial_attn,
                self.hand_orient_temporal_conv, self.hand_orient_mlp, T
            )
            aux_outputs["hand_orientation"] = raw.view(B, T, 2, 6)

        if self.training:
            return dict(
                reconstructions=reconstructions,
                quant_loss=quant_loss,
                quant_info=quant_info,
                pose_prediction=pose_prediction,
                latent=quant_codes,
                **aux_outputs,
            )

        return NetworkEval(
            reconstructions=reconstructions,
            quant_loss=quant_loss,
            quant_info=quant_info,
            pose_prediction=pose_prediction,
        )
