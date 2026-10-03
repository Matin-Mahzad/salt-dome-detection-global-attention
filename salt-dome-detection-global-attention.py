# SPDX-License-Identifier: MIT
"""
Implementation accompanying:

    M. Mahzad and M. Bagheri (2026). 3D Seismic Salt Dome Segmentation Using a Hybrid UNet with Global Attention,
    Denoising Pretraining, and Discriminative Transfer Learning.

Copyright (c) 2026 Matin Mahzad. Released under the MIT License (see LICENSE).
Authors: Matin Mahzad (ORCID 0009-0000-9346-8451), Majid Bagheri.
Contact: matinmahzad@yahoo.com

Models
------
    build_cnn()         3-D U-Net without attention.
    build_cnn_swin()    The same U-Net with shifted-window 3-D self-attention.
    build_cnn_global()  The same U-Net with unfactorized global 3-D self-attention
                        (every voxel attends to every voxel).
    build_salt3dnet()   Salt3DNet-style network: fully convolutional DenseNet blocks with
                        selective-kernel attention, a dual (segmentation and reconstruction)
                        encoder, and Barlow Twins pre-training components.

The first three models are instances of one class, UNet3D. They share the convolutional backbone,
the positions at which attention is inserted, and the internals of the attention layer
(pre-normalization, joint query/key/value projection, multi-head attention, output projection,
residual connection). They differ only in the attention scope, i.e., in the set of voxels that each
voxel attends to:

    none    no token mixing
    swin    non-overlapping windows of side 8 voxels; the window grid is shifted by 4 voxels in the
            decoder blocks (windows are clipped to the feature-map size at coarse levels)
    global  all N = 64^3 = 262,144 voxels at once (fused attention kernel, O(N) memory)

Backbone: 3-D U-Net with filters (12, 24, 48, 96) and a bottleneck of width 192, two
Conv3d-BatchNorm-ReLU layers per block, max-pooling for down-sampling, transposed convolutions for
up-sampling and concatenation skip connections (3.18 M trainable parameters without attention).
All models return logits; the sigmoid is applied by the loss or metric computation.

Salt3DNet is not part of the attention-scope comparison. It is built from its own blocks (see the
Salt3DNet section) and shares with the U-Nets only the skeleton and the channel widths, so that the
input/output contract and the model size are comparable.

Example
-------
    model = build_cnn_global()   # input (B, 1, 64, 64, 64) -> logits (B, 1, 64, 64, 64)
    model = build_salt3dnet()    # same contract; forward(x, return_recon=True) -> (logits, reconstruction)

Requirements: PyTorch >= 2.1 (``scale`` argument of scaled_dot_product_attention); the fused-kernel
check additionally requires PyTorch >= 2.3.

References
----------
Cicek et al. (2016). 3D U-Net: learning dense volumetric segmentation from sparse annotation. MICCAI.
Dao et al. (2022). FlashAttention: fast and memory-efficient exact attention with IO-awareness. NeurIPS.
Jegou et al. (2017). The One Hundred Layers Tiramisu: fully convolutional DenseNets for semantic
    segmentation. CVPR Workshops.
Li et al. (2019). Selective kernel networks. CVPR.
Liu et al. (2021). Swin Transformer: hierarchical vision transformer using shifted windows. ICCV.
Liu et al. (2022). Video Swin Transformer. CVPR.
Ronneberger et al. (2015). U-Net: convolutional networks for biomedical image segmentation. MICCAI.
Vaswani et al. (2017). Attention is all you need. NeurIPS.
Yang et al. (2024). Salt3DNet: a self-supervised learning framework for 3-D salt segmentation.
    IEEE Trans. Geosci. Remote Sens. 62.
Zbontar et al. (2021). Barlow Twins: self-supervised learning via redundancy reduction. ICML.
"""
import copy
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "1.0.0"
__author__ = "Matin Mahzad"
__license__ = "MIT"

__all__ = [
    "GlobalAttention3D", "ShiftedWindowAttention3D", "ConvBlock3D", "UNet3D",
    "build_cnn", "build_cnn_swin", "build_cnn_global",
    "SelectiveKernel3D", "DenseLayer3D", "SKDenseBlock3D", "Salt3DEncoder", "Salt3DNet",
    "BarlowTwinsHead", "barlow_twins_loss", "build_salt3dnet",
]


# =============================================================================
# Attention: shared base class and the two attention scopes
# =============================================================================
class _Attention3D(nn.Module):
    """Base class of the 3-D self-attention layers.

    For a feature map ``x`` of shape (B, C, D, H, W) the layer computes
    ``x + proj(mix(qkv(LayerNorm(x))))``: channel-wise layer normalization (pre-norm), a joint linear
    projection to queries, keys and values, a token-mixing operation ``mix`` and a linear output
    projection, followed by a residual connection. Subclasses implement only ``mix``, which defines
    the attention scope, i.e., the set of voxels each voxel attends to.

    Parameters
    ----------
    dim : int
        Number of channels C.
    head_dim : int, optional
        Target number of channels per head (default: 24). The number of heads is
        ``max(1, dim // head_dim)``; for the default filters, C = 12, 24, 48, 96, 192 yield
        1, 1, 2, 4, 8 heads.
    """

    def __init__(self, dim: int, head_dim: int = 24):
        super().__init__()
        self.dim = dim
        self.heads = max(1, dim // head_dim)
        if dim % self.heads != 0:
            raise ValueError(f"dim={dim} must be divisible by the number of heads ({self.heads}).")
        self.hd = dim // self.heads
        self.scale = self.hd ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def mix(self, qkv: torch.Tensor, B: int, D: int, H: int, W: int) -> torch.Tensor:
        """Token mixing. ``qkv``: (B, D, H, W, 3C) -> (B, D, H, W, C)."""
        raise NotImplementedError

    def mix_macs(self, B: int, N: int) -> float:
        """Multiply-accumulate operations of the QK^T and AV products for N voxels."""
        raise NotImplementedError

    def forward(self, x):                                        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        t = self.norm(x.permute(0, 2, 3, 4, 1))                   # (B, D, H, W, C)
        t = self.proj(self.mix(self.qkv(t), B, D, H, W))          # (B, D, H, W, C)
        return x + t.permute(0, 4, 1, 2, 3).contiguous()


class GlobalAttention3D(_Attention3D):
    """Unfactorized global self-attention: every voxel attends to every voxel of the feature map.

    Neither windowing nor axis-wise factorization is applied. The attention is evaluated with
    ``torch.nn.functional.scaled_dot_product_attention``, which computes
    ``softmax(Q K^T / sqrt(d_h)) V`` exactly (Vaswani et al., 2017). When a fused backend
    (FlashAttention, Dao et al., 2022, or memory-efficient attention) is selected, the product is
    evaluated in tiles and the memory grows as O(N) instead of O(N^2) in the number of voxels
    N = D * H * W.

    Fused backends require a head dimension that is a multiple of 8. With the default filters the
    head dimension is 24 at every level except C = 12, where it is 12; there, queries, keys and
    values are zero-padded to 16 channels. The padding is exact: the padded channels add zero to the
    query-key products and produce zero output channels, which are discarded. The softmax scale
    always uses the unpadded head dimension.
    """

    def __init__(self, dim: int, head_dim: int = 24):
        super().__init__(dim, head_dim)
        self.pad = (-self.hd) % 8                  # channels appended to reach a multiple of 8

    def mix(self, qkv, B: int, D: int, H: int, W: int):
        N = D * H * W
        t = qkv.reshape(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)   # (3, B, heads, N, hd)
        q = t[0].contiguous()
        k = t[1].contiguous()
        v = t[2].contiguous()
        if self.pad > 0:
            q = F.pad(q, [0, self.pad])
            k = F.pad(k, [0, self.pad])
            v = F.pad(v, [0, self.pad])
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)         # unpadded head dimension
        out = out[..., : self.hd]
        return out.transpose(1, 2).reshape(B, D, H, W, self.dim)

    def mix_macs(self, B: int, N: int) -> float:              # QK^T and AV: 2 * N^2 * C multiply-adds
        return 2.0 * B * N * N * self.dim


class ShiftedWindowAttention3D(_Attention3D):
    """Shifted-window 3-D self-attention (Liu et al., 2021; 3-D extension: Liu et al., 2022).

    Each voxel attends only to the voxels of its non-overlapping window of side ``window``. For
    ``shift > 0`` the window grid is cyclically shifted by ``shift`` voxels along every axis, and the
    standard Swin attention mask prevents attention between regions that are not adjacent in the
    original volume (regions brought together by the cyclic shift). Optionally, a learned relative
    position bias of shape ((2 * window - 1)^3, heads) is added to the attention logits.
    If the feature map is not larger than the window, the window covers the entire map.

    Parameters
    ----------
    dim : int
        Number of channels.
    window : int
        Window side in voxels.
    shift : int, optional
        Cyclic shift of the window grid in voxels (default: 0, regular windows).
    rel_pos_bias : bool, optional
        Add a learned relative position bias (default: True).
    head_dim : int, optional
        Target number of channels per head (default: 24).
    """

    __constants__ = ["use_rpb"]

    def __init__(self, dim: int, window: int, shift: int = 0, rel_pos_bias: bool = True, head_dim: int = 24):
        super().__init__(dim, head_dim)
        self.ws = window
        self.shift = shift
        self.use_rpb = rel_pos_bias
        if rel_pos_bias:
            w = window
            self.rpb_table = nn.Parameter(torch.zeros((2 * w - 1) ** 3, self.heads))
            nn.init.trunc_normal_(self.rpb_table, std=0.02)
            c = torch.stack(torch.meshgrid(*([torch.arange(w)] * 3), indexing="ij")).flatten(1)   # (3, Nw) coordinates
            r = (c[:, :, None] - c[:, None, :]).permute(1, 2, 0) + (w - 1)                        # (Nw, Nw, 3) offsets >= 0
            idx = r[..., 0] * (2 * w - 1) ** 2 + r[..., 1] * (2 * w - 1) + r[..., 2]
            self.register_buffer("rel_index", idx.reshape(-1), persistent=False)

    def _region(self, n: int, ref: torch.Tensor) -> torch.Tensor:
        """Region label (0, 1 or 2) of each of the n positions along one axis, as in Swin."""
        i = torch.arange(n, device=ref.device)
        return (i >= n - self.ws).long() + (i >= n - self.shift).long()

    def _mask(self, D: int, H: int, W: int, ref: torch.Tensor) -> torch.Tensor:
        """Boolean attention mask of shape (nWin, Nw, Nw); True marks disallowed query-key pairs."""
        ws = self.ws
        r = (self._region(D, ref).view(D, 1, 1) * 9
             + self._region(H, ref).view(1, H, 1) * 3
             + self._region(W, ref).view(1, 1, W))
        r = r.view(D // ws, ws, H // ws, ws, W // ws, ws).permute(0, 2, 4, 1, 3, 5).reshape(-1, ws * ws * ws)
        return r.unsqueeze(2) != r.unsqueeze(1)

    def mix(self, qkv, B: int, D: int, H: int, W: int):
        ws, s, h, hd = self.ws, self.shift, self.heads, self.hd
        assert D % ws == 0 and H % ws == 0 and W % ws == 0, "feature map must be divisible by the window"
        nD, nH, nW = D // ws, H // ws, W // ws
        nWin, Nw = nD * nH * nW, ws * ws * ws

        if s > 0:
            qkv = torch.roll(qkv, [-s, -s, -s], [1, 2, 3])
        t = qkv.reshape(B, nD, ws, nH, ws, nW, ws, 3, h, hd)
        t = t.permute(0, 1, 3, 5, 7, 8, 2, 4, 6, 9).reshape(B, nWin, 3, h, Nw, hd)
        q = t[:, :, 0] * self.scale                               # (B, nWin, heads, Nw, hd)
        k = t[:, :, 1]
        v = t[:, :, 2]

        attn = q @ k.transpose(-2, -1)                            # (B, nWin, heads, Nw, Nw)
        if self.use_rpb:
            bias = self.rpb_table[self.rel_index].view(Nw, Nw, h).permute(2, 0, 1)
            attn += bias.to(attn.dtype).unsqueeze(0).unsqueeze(0)
        if s > 0:
            attn.masked_fill_(self._mask(D, H, W, qkv).view(1, nWin, 1, Nw, Nw), float("-inf"))
        out = attn.softmax(dim=-1) @ v                            # (B, nWin, heads, Nw, hd)

        out = out.reshape(B, nD, nH, nW, h, ws, ws, ws, hd)
        out = out.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).reshape(B, D, H, W, h * hd)
        if s > 0:
            out = torch.roll(out, [s, s, s], [1, 2, 3])
        return out

    def mix_macs(self, B: int, N: int) -> float:              # each voxel attends to Nw = ws^3 keys
        return 2.0 * B * N * (self.ws ** 3) * self.dim


# =============================================================================
# U-Net backbone (shared by CNN, CNN+Swin and CNN+Global)
# =============================================================================
class ConvBlock3D(nn.Module):
    """Conv3d-BatchNorm-ReLU, optional attention, Conv3d-BatchNorm-ReLU.

    ``attn`` is ``nn.Identity()`` in the attention-free network. All convolutions are 3x3x3 with
    padding 1, so the spatial size is preserved.
    """

    def __init__(self, cin: int, cout: int, attn: nn.Module):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(cout)
        self.attn = attn
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(cout)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.attn(x)
        return F.relu(self.bn2(self.conv2(x)))


class UNet3D(nn.Module):
    """3-D U-Net (Ronneberger et al., 2015; Cicek et al., 2016) with a selectable attention scope.

    Encoder: ``len(filters)`` x [ConvBlock3D, max-pooling]; bottleneck: ConvBlock3D of width
    ``2 * filters[-1]``; decoder: ``len(filters)`` x [transposed convolution, concatenation with the
    skip connection, ConvBlock3D]; output: 1x1x1 convolution. Attention is placed inside every
    ConvBlock3D (encoder, bottleneck and decoder), between the two convolutions of the block.
    The network returns logits.

    Parameters
    ----------
    attention : {"none", "swin", "global"}
        Attention scope: no attention, shifted-window attention, or unfactorized global attention.
    in_channels, out_channels : int
        Number of input and output channels (default: 1).
    filters : sequence of int
        Block widths of the encoder levels (default: (12, 24, 48, 96)); the bottleneck has twice
        the width of the last level.
    input_size : int
        Side of the input cube the model is built for (default: 64). It determines the window sizes
        and relative-position-bias tables of the shifted-window attention and is ignored by the
        other variants. It must be divisible by ``2 ** len(filters)``.
    window_size : int
        Window side of the shifted-window attention (default: 8). At levels whose feature map is
        not larger than the window, the window equals the feature map. Windows are shifted by
        ``window // 2`` in decoder blocks and are regular in encoder and bottleneck blocks.
    head_dim : int
        Target number of channels per attention head (default: 24).
    rel_pos_bias : bool
        Learned relative position bias of the shifted-window attention (default: True). It adds
        56,744 parameters at ``input_size=64``. With ``rel_pos_bias=False`` the shifted-window and
        global variants have the same number of parameters.
    """

    def __init__(self, attention: str = "none", in_channels: int = 1, out_channels: int = 1,
                 filters=(12, 24, 48, 96), input_size: int = 64, window_size: int = 8,
                 head_dim: int = 24, rel_pos_bias: bool = True):
        super().__init__()
        if attention not in ("none", "swin", "global"):
            raise ValueError(f"attention must be 'none', 'swin' or 'global'; got {attention!r}.")
        n = len(filters)
        if input_size % (2 ** n) != 0:
            raise ValueError(f"input_size={input_size} must be divisible by 2**len(filters)={2 ** n}.")

        def make_attn(c: int, level: int, decoder: bool) -> nn.Module:
            if attention == "none":
                return nn.Identity()
            if attention == "global":
                return GlobalAttention3D(c, head_dim)
            res = input_size >> level                       # feature-map side at this level
            win = min(window_size, res)                     # the window equals the map if res <= window
            shift = win // 2 if (decoder and res > win) else 0
            return ShiftedWindowAttention3D(c, win, shift, rel_pos_bias, head_dim)

        cin = [in_channels] + list(filters[:-1])
        self.enc = nn.ModuleList(
            [ConvBlock3D(cin[i], filters[i], make_attn(filters[i], i, False)) for i in range(n)])
        self.pool = nn.MaxPool3d(2)
        self.bridge = ConvBlock3D(filters[-1], 2 * filters[-1], make_attn(2 * filters[-1], n, False))

        rev = list(reversed(filters))
        up_in = [2 * filters[-1]] + rev[:-1]
        self.up = nn.ModuleList([nn.ConvTranspose3d(up_in[j], rev[j], 2, stride=2) for j in range(n)])
        self.dec = nn.ModuleList(
            [ConvBlock3D(2 * rev[j], rev[j], make_attn(rev[j], n - 1 - j, True)) for j in range(n)])
        self.out_conv = nn.Conv3d(filters[0], out_channels, 1)

    def forward(self, x):
        skips: List[torch.Tensor] = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bridge(x)
        for up, blk in zip(self.up, self.dec):
            x = torch.cat([up(x), skips.pop()], dim=1)
            x = blk(x)
        return self.out_conv(x)                              # logits


def build_cnn(**kw) -> UNet3D:
    """3-D U-Net without attention."""
    return UNet3D(attention="none", **kw)


def build_cnn_swin(**kw) -> UNet3D:
    """3-D U-Net with shifted-window attention."""
    return UNet3D(attention="swin", **kw)


def build_cnn_global(**kw) -> UNet3D:
    """3-D U-Net with unfactorized global attention."""
    return UNet3D(attention="global", **kw)


# =============================================================================
# Salt3DNet-style model: FC-DenseNet blocks, selective-kernel attention, Barlow Twins
# =============================================================================
class SelectiveKernel3D(nn.Module):
    """Three-dimensional selective-kernel (SK) block (Li et al., 2019), the soft-attention
    mechanism of Salt3DNet.

    split   Parallel Conv-BatchNorm-ReLU branches with different receptive fields. A branch with
            kernel size k is a 3x3x3 convolution with dilation (k - 1) / 2 (k = 5 corresponds to
            dilation 2, as in SKNet); grouped convolutions keep the branches inexpensive.
    fuse    The branch outputs are summed, averaged over (D, H, W) and passed through a fully
            connected layer.
    select  One fully connected layer per branch and a softmax across branches give per-channel
            branch weights; the output is the weighted sum of the branches.

    Each channel thus selects its receptive field from global context. Input and output have
    shape (B, C, D, H, W). There is no residual connection: the output is the weighted branch sum,
    as in SKNet.

    Parameters
    ----------
    dim : int
        Number of channels.
    kernels : sequence of int
        Odd effective kernel sizes (>= 3) of the branches (default: (3, 5)).
    groups : int
        Target number of convolution groups (default: 12); the group count used is
        ``gcd(dim, groups)``.
    reduction : int
        Reduction ratio of the fully connected bottleneck (default: 4).
    min_hidden : int
        Minimum width of the fully connected bottleneck (default: 8).
    """

    def __init__(self, dim: int, kernels=(3, 5), groups: int = 12, reduction: int = 4, min_hidden: int = 8):
        super().__init__()
        if not all(k >= 3 and k % 2 == 1 for k in kernels):
            raise ValueError("kernel sizes must be odd and >= 3.")
        g = math.gcd(dim, groups)                    # C = 12, 24, 48, 96, 192 are all divisible by 12
        self.branches = nn.ModuleList()
        for k in kernels:
            d = (k - 1) // 2                         # a 3^3 kernel with dilation d covers a k^3 neighbourhood
            self.branches.append(nn.Sequential(
                nn.Conv3d(dim, dim, 3, padding=d, dilation=d, groups=g, bias=False),
                nn.BatchNorm3d(dim), nn.ReLU()))
        hidden = max(dim // reduction, min_hidden)
        self.fc = nn.Linear(dim, hidden)             # no BatchNorm: undefined for a single sample in training mode
        self.select = nn.ModuleList([nn.Linear(hidden, dim) for _ in kernels])

    def forward(self, x):                                                        # (B, C, D, H, W)
        feats = [b(x) for b in self.branches]                                    # split
        z = F.relu(self.fc(sum(feats).mean(dim=(2, 3, 4))))                      # fuse:   (B, hidden)
        w = torch.stack([s(z) for s in self.select], dim=1).softmax(dim=1)       # select: (B, branches, C)
        return sum(w[:, i, :, None, None, None] * f for i, f in enumerate(feats))


class DenseLayer3D(nn.Module):
    """FC-DenseNet layer (Jegou et al., 2017): BatchNorm, ReLU, 3x3x3 convolution producing
    ``growth`` new feature maps."""

    def __init__(self, cin: int, growth: int):
        super().__init__()
        self.bn = nn.BatchNorm3d(cin)
        self.conv = nn.Conv3d(cin, growth, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(F.relu(self.bn(x)))


class SKDenseBlock3D(nn.Module):
    """Dense block followed by selective-kernel attention, with the same (cin -> cout) interface
    as ConvBlock3D.

    The block consists of ``n_layers`` DenseLayer3D with growth rate ``cout / n_layers``; layer l
    receives the concatenation [x, y_1, ..., y_(l-1)] of the block input and all previous layer
    outputs (dense connectivity). As in the up-path blocks of FC-DenseNet, only the newly created
    feature maps are returned; their concatenation has ``cout`` channels and is refined by a
    SelectiveKernel3D layer. The channel widths therefore match those of ConvBlock3D one-to-one,
    and the surrounding encoder-decoder skeleton is unchanged.
    """

    def __init__(self, cin: int, cout: int, n_layers: int = 4, sk_kernels=(3, 5), sk_groups: int = 12,
                 sk_reduction: int = 4):
        super().__init__()
        if cout % n_layers != 0:
            raise ValueError(f"block width {cout} must be divisible by n_layers={n_layers}.")
        g = cout // n_layers
        self.layers = nn.ModuleList([DenseLayer3D(cin + i * g, g) for i in range(n_layers)])
        self.skb = SelectiveKernel3D(cout, sk_kernels, sk_groups, sk_reduction)

    def forward(self, x):
        feats, new = x, []
        for i, layer in enumerate(self.layers):
            y = layer(feats)
            new.append(y)
            if i < len(self.layers) - 1:                       # the last concatenation is never used
                feats = torch.cat([feats, y], dim=1)
        return self.skb(torch.cat(new, dim=1))


class Salt3DEncoder(nn.Module):
    """Encoder of the Salt3DNet-style model: stem convolution, ``len(filters)`` x [SK-dense block,
    max-pooling], and an SK-dense bridge of width ``2 * filters[-1]``.

    It is instantiated twice inside Salt3DNet (segmentation and reconstruction encoders) and used
    stand-alone for Barlow Twins pre-training through ``embed``. ``forward`` returns
    ``(bottleneck, skips)``, where ``skips[i]`` has ``filters[i]`` channels.
    """

    def __init__(self, in_channels: int = 1, filters=(12, 24, 48, 96), n_layers: int = 4,
                 sk_kernels=(3, 5), sk_groups: int = 12, sk_reduction: int = 4):
        super().__init__()
        sk = dict(n_layers=n_layers, sk_kernels=sk_kernels, sk_groups=sk_groups, sk_reduction=sk_reduction)
        # As in FC-DenseNet, the first layer is a plain convolution: the signed input amplitudes are
        # not passed through BatchNorm-ReLU before any learned linear mixing.
        self.stem = nn.Conv3d(in_channels, filters[0], 3, padding=1)
        cin = [filters[0]] + list(filters[:-1])
        self.blocks = nn.ModuleList([SKDenseBlock3D(cin[i], filters[i], **sk) for i in range(len(filters))])
        self.pool = nn.MaxPool3d(2)
        self.bridge = SKDenseBlock3D(filters[-1], 2 * filters[-1], **sk)
        self.out_dim = 2 * filters[-1]

    def forward(self, x):
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            skips.append(x)
            x = self.pool(x)
        return self.bridge(x), skips

    def embed(self, x):
        """Globally averaged bottleneck features, shape (B, 2 * filters[-1]); the representation
        on which the Barlow Twins objective operates."""
        return self.forward(x)[0].mean(dim=(2, 3, 4))


class Salt3DNet(nn.Module):
    """Salt3DNet-style multitask network for 3-D salt segmentation.

    Inspired by L. Yang, S. Fomel, S. Wang, X. Chen, O. M. Saad and Y. Chen, "Salt3DNet: A
    self-supervised learning framework for 3-D salt segmentation", IEEE Trans. Geosci. Remote
    Sens. 62, 5913115 (2024), doi:10.1109/TGRS.2024.3394592. In that work, (1) stage 1 pre-trains
    the encoder with Barlow Twins; (2) stage 2 fine-tunes two encoders that reconstruct the 3-D
    seismic data and segment the salt body in a multitask, collaborative manner; and (3) encoders
    and decoder are 3-D fully convolutional DenseNets with a soft-attention mechanism, the
    selective-kernel block (SKB: several kernel sizes with learned selection). Design details
    beyond this description are choices of the present implementation, expressed in the
    conventions of this module (same widths and max-pooling / transposed-convolution /
    concatenation-skip skeleton as UNet3D; logits as output):

      * Every ConvBlock3D is replaced by an SKDenseBlock3D with the same cin -> cout widths:
        ``n_layers`` dense layers (growth rate width / n_layers) followed by one SKB (kernels 3
        and 5, grouped). No dropout is used, as in UNet3D.
      * ``dual_encoder=True``: a segmentation encoder and a reconstruction encoder, fused by
        concatenation and a 1x1x1 convolution at every skip level and at the bottleneck; one shared
        decoder; two 1x1x1 output heads (segmentation logits and reconstruction of the input).
      * ``dual_encoder=False``: single-encoder backbone without reconstruction branch.

    ``forward(x)`` returns the segmentation logits, like the U-Nets; ``forward(x, return_recon=True)``
    returns ``(logits, reconstruction)`` for a multitask loss. The corruption of ``x`` (masking or
    noise) that makes the reconstruction task non-trivial is applied by the training procedure and
    is not part of this module.

    Stage 1 (self-supervised):
        enc = Salt3DEncoder();  head = BarlowTwinsHead(enc.out_dim)
        loss = barlow_twins_loss(head(enc.embed(view_a)), head(enc.embed(view_b)))
    Stage 2 (supervised):
        model = build_salt3dnet();  model.load_pretrained_encoder(enc.state_dict())
        logits, recon = model(x, return_recon=True)

    ``input_size`` is accepted for signature parity with UNet3D; it is only used to check
    divisibility by ``2 ** len(filters)``.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, filters=(12, 24, 48, 96),
                 input_size: int = 64, n_layers: int = 4, sk_kernels=(3, 5), sk_groups: int = 12,
                 sk_reduction: int = 4, dual_encoder: bool = True):
        super().__init__()
        n = len(filters)
        if input_size % (2 ** n) != 0:
            raise ValueError(f"input_size={input_size} must be divisible by 2**len(filters)={2 ** n}.")
        sk = dict(n_layers=n_layers, sk_kernels=sk_kernels, sk_groups=sk_groups, sk_reduction=sk_reduction)

        self.enc_seg = Salt3DEncoder(in_channels, filters, **sk)       # segmentation encoder
        self.enc_rec, self.fuse, self.rec_conv = None, None, None
        if dual_encoder:
            self.enc_rec = Salt3DEncoder(in_channels, filters, **sk)   # reconstruction encoder
            widths = list(filters) + [2 * filters[-1]]                 # n skip levels, then the bottleneck
            self.fuse = nn.ModuleList([
                nn.Sequential(nn.Conv3d(2 * c, c, 1, bias=False), nn.BatchNorm3d(c), nn.ReLU())
                for c in widths])
            self.rec_conv = nn.Conv3d(filters[0], in_channels, 1)      # reconstruction head

        rev = list(reversed(filters))                                  # shared decoder, widths as in UNet3D
        up_in = [2 * filters[-1]] + rev[:-1]
        self.up = nn.ModuleList([nn.ConvTranspose3d(up_in[j], rev[j], 2, stride=2) for j in range(n)])
        self.dec = nn.ModuleList([SKDenseBlock3D(2 * rev[j], rev[j], **sk) for j in range(n)])
        self.out_conv = nn.Conv3d(filters[0], out_channels, 1)         # segmentation head (logits)

    def forward(self, x, return_recon: bool = False):
        x_b, skips = self.enc_seg(x)
        if self.enc_rec is not None:                                   # fusion of the two encoders
            r_b, r_skips = self.enc_rec(x)
            skips = [f(torch.cat([s, r], dim=1)) for f, s, r in zip(self.fuse[:-1], skips, r_skips)]
            x_b = self.fuse[-1](torch.cat([x_b, r_b], dim=1))
        x = x_b
        for up, blk in zip(self.up, self.dec):
            x = blk(torch.cat([up(x), skips.pop()], dim=1))
        logits = self.out_conv(x)
        if not return_recon:
            return logits
        assert self.rec_conv is not None, "return_recon=True requires dual_encoder=True"
        return logits, self.rec_conv(x)

    def load_pretrained_encoder(self, state_dict, strict: bool = True):
        """Stage 1 -> stage 2 transfer: copy Barlow Twins pre-trained Salt3DEncoder weights into
        both encoders."""
        self.enc_seg.load_state_dict(state_dict, strict=strict)
        if self.enc_rec is not None:
            self.enc_rec.load_state_dict(state_dict, strict=strict)


class BarlowTwinsHead(nn.Module):
    """Projection head for Barlow Twins pre-training (stage 1): MLP in_dim -> hidden -> hidden ->
    out_dim with BatchNorm and ReLU between the layers. It is discarded after pre-training and is
    therefore defined outside Salt3DNet; it does not contribute to the parameters of the model."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim, bias=False))

    def forward(self, z):
        return self.net(z)


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 5e-3) -> torch.Tensor:
    """Barlow Twins objective (Zbontar et al., 2021).

    Each embedding is standardized over the batch, the D x D cross-correlation matrix C of the two
    views is formed, and C is driven towards the identity: diagonal terms towards 1 (invariance to
    the augmentation), off-diagonal terms towards 0 (redundancy reduction, weighted by ``lambd``).

        L = sum_i (1 - C_ii)^2 + lambd * sum_{i != j} C_ij^2

    Parameters
    ----------
    z1, z2 : torch.Tensor
        Projector outputs of shape (B, D) for two augmented views of the same sub-volumes; B > 1.
    lambd : float, optional
        Weight of the off-diagonal term (default: 5e-3).
    """
    z1, z2 = z1.float(), z2.float()
    B = z1.shape[0]
    z1 = (z1 - z1.mean(0)) / (z1.std(0, correction=0) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0, correction=0) + 1e-6)
    c = z1.T @ z2 / B                                          # (D, D)
    diag = torch.diagonal(c)
    return (diag - 1).pow(2).sum() + lambd * (c.pow(2).sum() - diag.pow(2).sum())


def build_salt3dnet(**kw) -> Salt3DNet:
    """Salt3DNet-style multitask network (see Salt3DNet)."""
    return Salt3DNet(**kw)
