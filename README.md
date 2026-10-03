# 3D Salt Dome Segmentation with Hybrid U-Nets and Global Attention

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

PyTorch implementation accompanying:

> M. Mahzad and M. Bagheri (2026). **3D Seismic Salt Dome Segmentation Using a Hybrid UNet with Global Attention, Denoising Pretraining, and Discriminative Transfer Learning.**

The module defines four 3-D segmentation networks for seismic volumes:

- **CNN, CNN + Swin and CNN + Global** are one 3-D U-Net with three attention scopes: none, shifted-window, and unfactorized global attention (every voxel attends to every voxel). Only the scope changes between them, so differences in behavior can be attributed to it.
- **Salt3DNet-style network** combines fully convolutional DenseNet blocks, selective-kernel attention, a dual (segmentation and reconstruction) encoder, and Barlow Twins pre-training components.

Every model maps a `(B, 1, 64, 64, 64)` seismic volume to `(B, 1, 64, 64, 64)` logits. The only dependency is PyTorch.

The module contains the architectures, the Barlow Twins projection head, and the Barlow Twins loss. Data loading, input corruption for the reconstruction task, and the training and transfer-learning procedures are outside its scope.

## Models

| Builder | Backbone | Attention scope | Parameters |
|---|---|---|---|
| `build_cnn()` | 3-D U-Net | none | 3.18 M |
| `build_cnn_swin()` | 3-D U-Net | non-overlapping 8³ windows; window grid shifted by 4 voxels in the decoder blocks | 3.48 M |
| `build_cnn_global()` | 3-D U-Net | global: every voxel attends to all 64³ = 262,144 voxels | 3.43 M |
| `build_salt3dnet()` | FC-DenseNet blocks, dual encoder | selective-kernel (channel-wise) attention | 4.00 M |

Parameter counts are for the default configuration (filters `(12, 24, 48, 96)`, one input and one output channel, `input_size=64`) and exclude the Barlow Twins projection head.

The first three builders return instances of the same class, `UNet3D`. They share the convolutional backbone, the positions at which attention is inserted, and the internals of the attention layer (pre-normalization, joint query/key/value projection, multi-head attention, output projection, residual connection). Salt3DNet is not part of this attention-scope comparison: it is built from its own blocks and shares only the skeleton and channel widths with the U-Nets.

## Installation

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
pip install "torch>=2.1"
```

PyTorch 2.1 or newer is required (the `scale` argument of `scaled_dot_product_attention`). PyTorch 2.3 or newer is needed for the fused-kernel check described in [Running global attention at full resolution](#running-global-attention-at-full-resolution).

## Quick start

```python
import torch
from models import build_cnn, build_cnn_swin, build_cnn_global, build_salt3dnet

x = torch.randn(1, 1, 64, 64, 64)      # (batch, channel, depth, height, width)

model = build_cnn_swin()               # or build_cnn(), build_cnn_global(), build_salt3dnet()
logits = model(x)                      # (1, 1, 64, 64, 64)
probs = torch.sigmoid(logits)          # the models return logits; the sigmoid belongs to the loss or metric
```

The `build_*` functions forward keyword arguments to the underlying class:

```python
model = build_cnn_swin(input_size=32, window_size=8, rel_pos_bias=False)
model = build_salt3dnet(dual_encoder=False)
```

See the [configuration reference](#configuration-reference) for all options. For `build_cnn_global()` at 64³, read [Running global attention at full resolution](#running-global-attention-at-full-resolution) first.

## Architecture

### U-Net backbone

- Encoder widths `(12, 24, 48, 96)` and a bottleneck of width 192.
- Each block is `Conv3d → BatchNorm → ReLU → attention → Conv3d → BatchNorm → ReLU`, with 3×3×3 kernels and padding 1.
- Max-pooling for down-sampling, transposed convolutions for up-sampling, concatenation skip connections, and a 1×1×1 output convolution.
- Attention is inserted in all nine blocks (four encoder blocks, the bottleneck, four decoder blocks). The plain `CNN` uses `nn.Identity()` in its place.

### Attention layer

Every attention layer computes

```
x + proj( mix( qkv( LayerNorm(x) ) ) )
```

that is, channel-wise pre-normalization, a joint query/key/value projection, a token-mixing step `mix`, an output projection, and a residual connection. `mix` is the only part that differs between variants, and it defines the attention scope. The number of heads is `max(1, C // 24)`, which gives 1, 1, 2, 4 and 8 heads for C = 12, 24, 48, 96 and 192 channels.

### Attention scopes

| Scope | Each voxel attends to | Notes |
|---|---|---|
| `none` | nothing | No token mixing. |
| `swin` | its own 8×8×8 window | Shifted-window attention (Liu et al., 2021; 3-D form: Liu et al., 2022). In decoder blocks the window grid is cyclically shifted by 4 voxels along every axis, and the standard Swin mask prevents attention between regions that are not adjacent in the volume. Learned relative position bias (optional). |
| `global` | all N = 64³ voxels | Exact `softmax(Q K^T / sqrt(d_h)) V` through `torch.nn.functional.scaled_dot_product_attention`, with no windowing or axis-wise factorization. |

Configuration of the attention layers at each resolution:

| Feature map | N (voxels) | C | Heads | Head dim | Attention in | `swin` window |
|---|---|---|---|---|---|---|
| 64³ | 262,144 | 12 | 1 | 12 (padded to 16 in `global`) | encoder, decoder | 8 |
| 32³ | 32,768 | 24 | 1 | 24 | encoder, decoder | 8 |
| 16³ | 4,096 | 48 | 2 | 24 | encoder, decoder | 8 |
| 8³ | 512 | 96 | 4 | 24 | encoder, decoder | 8 (whole map) |
| 4³ | 64 | 192 | 8 | 24 | bottleneck | 4 (whole map) |

Windows are regular in encoder and bottleneck blocks and shifted by 4 voxels in decoder blocks at 64³, 32³ and 16³. At 8³ and 4³ the window covers the whole feature map, so no shift is applied and `swin` and `global` differ in scope only at the three finest levels.

Fused attention kernels require a head dimension that is a multiple of 8. At C = 12 the head dimension is 12, so queries, keys and values are zero-padded to 16 channels. The padding is exact: the padded channels add zero to the query-key products and produce zero output channels, which are discarded, and the softmax scale uses the unpadded head dimension.

The attention layers add 248,688 parameters to the 3,177,877 of the plain U-Net. With `rel_pos_bias=True` (the default) the shifted-window variant has another 56,744 parameters for the relative position bias; with `rel_pos_bias=False` both attention variants have the same parameter count.

### Running global attention at full resolution

At 64³ the first encoder block and the last decoder block attend over N = 262,144 voxels.

**Memory.** With a fused `scaled_dot_product_attention` backend (FlashAttention or memory-efficient attention) the attention memory grows as O(N). If no fused backend is eligible, PyTorch falls back to the math implementation, which materializes the full N × N score matrix (about 275 GB per head and sample in fp32) and runs out of memory. A CUDA GPU with fp16/bf16 autocast is the most reliable way to get a fused kernel, since FlashAttention requires half precision:

```python
model = build_cnn_global().cuda()
x = torch.randn(1, 1, 64, 64, 64, device="cuda")

with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(x)
```

**Checking the backend** (PyTorch 2.3 or newer). Restrict attention to the fused backends so that an ineligible setup raises an error instead of silently falling back:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
```

**Compute.** Memory is linear in N, but compute is not: the `QK^T` and `AV` products cost `2·B·N²·C` multiply-accumulates per layer. For the default model at 64³ this is about 3.4 × 10¹² MACs per sample (forward pass, attention products only), roughly 97% of it in the two full-resolution blocks. The shifted-window variant needs about 8.6 × 10⁹ MACs.

## Salt3DNet-style network and Barlow Twins pre-training

`build_salt3dnet()` is inspired by Salt3DNet (Yang et al., 2024). Design details beyond the published description are choices of this implementation; see the `Salt3DNet` docstring. The network keeps the U-Net skeleton (max-pooling, transposed convolutions, concatenation skips) and channel widths, so the input/output contract and the model size stay comparable:

- Every `ConvBlock3D` is replaced by an `SKDenseBlock3D`: four dense layers (growth rate = block width / 4) followed by a selective-kernel block (Li et al., 2019) with kernels 3 and 5 and grouped convolutions.
- With `dual_encoder=True` (the default), a segmentation encoder and a reconstruction encoder are fused by concatenation and a 1×1×1 convolution at every skip level and at the bottleneck. One shared decoder feeds two 1×1×1 heads: segmentation logits and a reconstruction of the input.
- With `dual_encoder=False` the network has a single encoder and no reconstruction branch (2.48 M instead of 4.00 M parameters).

```python
model = build_salt3dnet()
logits = model(x)                              # segmentation logits, as for the U-Nets
logits, recon = model(x, return_recon=True)    # logits and reconstruction, for a multitask loss
```

The corruption of `x` (masking or noise) that makes the reconstruction task non-trivial is applied by the training procedure, not by the model.

### Stage 1: self-supervised pre-training (Barlow Twins)

```python
from models import Salt3DEncoder, BarlowTwinsHead, barlow_twins_loss

enc = Salt3DEncoder()                    # enc.embed(x) -> (B, 192): globally averaged bottleneck features
head = BarlowTwinsHead(enc.out_dim)      # projector, discarded after pre-training

# view_a, view_b: two augmented views of the same sub-volumes, shape (B, 1, 64, 64, 64), B > 1
loss = barlow_twins_loss(head(enc.embed(view_a)), head(enc.embed(view_b)))
loss.backward()
```

### Stage 2: supervised multitask fine-tuning

```python
model = build_salt3dnet()
model.load_pretrained_encoder(enc.state_dict())    # copies the weights into both encoders
logits, recon = model(x, return_recon=True)
```

## Configuration reference

### `UNet3D`

| Argument | Default | Description |
|---|---|---|
| `attention` | `"none"` | `"none"`, `"swin"` or `"global"`. |
| `in_channels`, `out_channels` | `1`, `1` | Number of input and output channels. |
| `filters` | `(12, 24, 48, 96)` | Block widths of the encoder levels; the bottleneck has twice the width of the last level. |
| `input_size` | `64` | Side of the input cube the model is built for. Must be divisible by `2 ** len(filters)`. Sets the window sizes and bias tables of `swin`; ignored by the other variants. |
| `window_size` | `8` | Window side of `swin`. Clipped to the feature-map size at coarse levels. Windows are shifted by `window_size // 2` in decoder blocks. |
| `head_dim` | `24` | Target channels per head; the number of heads is `max(1, C // head_dim)`. |
| `rel_pos_bias` | `True` | Learned relative position bias in `swin` (56,744 parameters at `input_size=64`). |

Notes:

- Input sides must be divisible by `2 ** len(filters)` (16 for the default filters).
- For `swin`, build the model with the `input_size` you will feed: window sizes and relative-position-bias tables are fixed at construction, and every feature map larger than the window must be divisible by it.
- The number of heads must divide the channel count of every attention layer, otherwise a `ValueError` is raised.

### `Salt3DNet`

`in_channels`, `out_channels`, `filters`, `input_size` (only used to check divisibility by `2 ** len(filters)`), `n_layers` (dense layers per block, default 4), `sk_kernels` (default `(3, 5)`), `sk_groups` (default 12), `sk_reduction` (default 4) and `dual_encoder` (default `True`). Each block width must be divisible by `n_layers`.

## Module contents

- **Attention layers:** `GlobalAttention3D`, `ShiftedWindowAttention3D`
- **U-Net:** `ConvBlock3D`, `UNet3D`, `build_cnn`, `build_cnn_swin`, `build_cnn_global`
- **Salt3DNet:** `SelectiveKernel3D`, `DenseLayer3D`, `SKDenseBlock3D`, `Salt3DEncoder`, `Salt3DNet`, `build_salt3dnet`
- **Barlow Twins:** `BarlowTwinsHead`, `barlow_twins_loss`

## Citation

If you use this code, please cite:

```bibtex
@misc{mahzad2026salt,
  title  = {3D Seismic Salt Dome Segmentation Using a Hybrid UNet with Global Attention, Denoising Pretraining, and Discriminative Transfer Learning},
  author = {Mahzad, Matin and Bagheri, Majid},
  year   = {2026}
}
```

If you use the Salt3DNet-style model, please also cite Yang et al. (2024).

## References

- Cicek et al. (2016). 3D U-Net: learning dense volumetric segmentation from sparse annotation. MICCAI.
- Dao et al. (2022). FlashAttention: fast and memory-efficient exact attention with IO-awareness. NeurIPS.
- Jegou et al. (2017). The One Hundred Layers Tiramisu: fully convolutional DenseNets for semantic segmentation. CVPR Workshops.
- Li et al. (2019). Selective kernel networks. CVPR.
- Liu et al. (2021). Swin Transformer: hierarchical vision transformer using shifted windows. ICCV.
- Liu et al. (2022). Video Swin Transformer. CVPR.
- Ronneberger et al. (2015). U-Net: convolutional networks for biomedical image segmentation. MICCAI.
- Vaswani et al. (2017). Attention is all you need. NeurIPS.
- Yang, L., Fomel, S., Wang, S., Chen, X., Saad, O. M., and Chen, Y. (2024). Salt3DNet: a self-supervised learning framework for 3-D salt segmentation. IEEE Trans. Geosci. Remote Sens., 62, 5913115. [doi:10.1109/TGRS.2024.3394592](https://doi.org/10.1109/TGRS.2024.3394592)
- Zbontar et al. (2021). Barlow Twins: self-supervised learning via redundancy reduction. ICML.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Matin Mahzad.

## Contact

Matin Mahzad ([ORCID 0009-0000-9346-8451](https://orcid.org/0009-0000-9346-8451)), matinmahzad@yahoo.com
Majid Bagheri
