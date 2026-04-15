# Adapted from https://github.com/KlingTeam/VFRTok/blob/main/modelling/discriminators/discriminator_dino.py

import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop

try:
    from flash_attn.ops.fused_dense import fused_mlp_func
except ImportError:
    fused_mlp_func = None

try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    flash_attn_qkvpacked_func = None


class MLPNoDrop(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=0,
                process_group=None,
            )
        else:
            return self.fc2(self.act(self.fc1(x)))

    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'


class SelfAttentionNoDrop(nn.Module):
    def __init__(
            self, block_idx, embed_dim=768, num_heads=12, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # =64
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv, self.proj = nn.Linear(embed_dim, embed_dim * 3, bias=True), nn.Linear(embed_dim, embed_dim, bias=True)
        self.using_flash_attn = flash_if_available and flash_attn_qkvpacked_func is not None

    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim)
        if self.using_flash_attn and qkv.dtype != torch.float32:
            oup = flash_attn_qkvpacked_func(qkv, softmax_scale=self.scale).view(B, L, C)
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # BHLc
            oup = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.,
            )
            oup = oup.transpose(1, 2).reshape(B, L, C)
        return self.proj(oup)

    def extra_repr(self) -> str:
        return f'using_flash_attn={self.using_flash_attn}'


class SABlockNoDrop(nn.Module):
    def __init__(self, block_idx, embed_dim, num_heads, mlp_ratio, norm_eps):
        super(SABlockNoDrop, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = SelfAttentionNoDrop(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads,
                                        flash_if_available=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLPNoDrop(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio),
                             fused_if_available=True)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-6):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int, norm_type: str, norm_eps: float, using_spec_norm: bool) -> nn.Module:
    if norm_type == 'bn':
        norm = BatchNormLocal(channels, eps=norm_eps)
    elif norm_type == 'sbn':
        norm = nn.SyncBatchNorm(channels, eps=norm_eps, process_group=None)
    elif norm_type == 'gn':
        norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=norm_eps, affine=True)
    else:
        raise NotImplementedError(f"Unsupported norm_type: {norm_type}")

    return nn.Sequential(
        (SpectralConv1d if using_spec_norm else nn.Conv1d)(channels, channels, kernel_size=kernel_size,
                                                           padding=kernel_size // 2, padding_mode='circular'),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class DinoDisc(nn.Module):
    def __init__(self, dino_ckpt_path='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
                 ks=9, depth=12, key_depths=(2, 5, 8, 11), norm_type='sbn',
                 using_spec_norm=True, norm_eps=1e-6):
        super().__init__()
        state = torch.hub.load_state_dict_from_url(dino_ckpt_path, map_location='cpu')
        for k in sorted(state.keys()):
            if '.attn.qkv.bias' in k:
                bias = state[k]
                C = bias.numel() // 3
                bias[C:2 * C].zero_()  # zero out k_bias

        key_depths = tuple(d for d in key_depths if d < depth)
        dino = FrozenDINOSmallNoDrop(depth=depth, key_depths=key_depths, norm_eps=norm_eps)
        missing, unexpected = dino.load_state_dict(state, strict=False)
        missing = [m for m in missing if all(x not in m for x in {'x_scale', 'x_shift'})]
        if torch.cuda.is_available():
            assert len(missing) == 0, f'missing keys: {missing}'
            assert len(unexpected) == 0, f'unexpected keys: {unexpected}'

        self.dino_proxy: FrozenDINOSmallNoDrop = dino
        dino_C = dino.embed_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                make_block(dino_C, kernel_size=1, norm_type=norm_type, norm_eps=norm_eps,
                           using_spec_norm=using_spec_norm),
                ResidualBlock(make_block(dino_C, kernel_size=ks, norm_type=norm_type, norm_eps=norm_eps,
                                         using_spec_norm=using_spec_norm)),
                (SpectralConv1d if using_spec_norm else nn.Conv1d)(dino_C, 1, kernel_size=1, padding=0)
            )
            for _ in range(len(key_depths) + 1)  # +1: before all attention blocks
        ])

    def forward(self, x_in_pm1, grad_ckpt=False):  # x_in_pm1: image tensor normalized to [-1, 1]
        dino_grad_ckpt = grad_ckpt and x_in_pm1.requires_grad
        activations: List[torch.Tensor] = self.dino_proxy(x_in_pm1, grad_ckpt=dino_grad_ckpt)
        activations = [x.to(dtype=x_in_pm1.dtype) for x in activations]
        B = x_in_pm1.shape[0]
        return torch.cat([
            (
                h(act) if not grad_ckpt
                else torch.utils.checkpoint.checkpoint(h, act, use_reentrant=False)
            ).view(B, -1)
            for h, act in zip(self.heads, activations)
        ], dim=1)  # cat 5 BL => B, 5L


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # BCHW => BCL => BLC
        return self.norm(x)


class FrozenDINOSmallNoDrop(nn.Module):
    """
    Frozen DINO ViT-Small without dropout/droppath (eval mode only).
    Based on timm vit_small_patch16_224 with DINO pretrained weights.
    """

    def __init__(
            self, depth=12, key_depths=(2, 5, 8, 11), norm_eps=1e-6,
            patch_size=16, in_chans=3,
            embed_dim=384, num_heads=6, mlp_ratio=4.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = 224
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim)
        self.patch_nums = self.img_size // patch_size

        # Normalization: map x in [-1, 1] to ImageNet-normalized space
        m, s = torch.tensor((0.485, 0.456, 0.406)), torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer('x_scale', (0.5 / s).reshape(1, 3, 1, 1))
        self.register_buffer('x_shift', ((0.5 - m) / s).reshape(1, 3, 1, 1))
        self.crop = RandomCrop(self.img_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_nums * self.patch_nums + 1, embed_dim))

        self.key_depths = set(d for d in key_depths if d < depth)
        self.blocks = nn.Sequential(*[
            SABlockNoDrop(block_idx=i, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_eps=norm_eps)
            for i in range(max(depth, 1 + max(self.key_depths)))
        ])

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, grad_ckpt=False):
        x = (self.x_scale * x).add_(self.x_shift)
        H, W = x.shape[-2], x.shape[-1]
        if H > self.img_size and W > self.img_size and random.random() <= 0.5:
            x = self.crop(x)
        else:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                                mode='area' if H > self.img_size else 'bicubic')

        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        activations = [(x[:, 1:] + x[:, :1]).transpose_(1, 2)]  # readout
        for i, b in enumerate(self.blocks):
            if not grad_ckpt:
                x = b(x)
            else:
                x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False)
            if i in self.key_depths:
                activations.append((x[:, 1:] + x[:, :1]).transpose_(1, 2))  # readout
        return activations


