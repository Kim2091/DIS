"""
DIS2: Direct Image Supersampling 2

A faster successor to DIS built around structural reparameterization
(RepVGG / ECBSR / ABPN lineage):

- TRAINING form: every 3x3 conv is an Edge-oriented Convolution Block (ECB)
  with parallel branches (plain 3x3, 1x1, 1x1->3x3 expansion, and
  1x1->Sobel-x / Sobel-y / Laplacian edge extractors, plus identity where
  shapes allow). The extra branches only exist during training and buy the
  quality that DIS got from being twice as deep.

- INFERENCE form: every ECB collapses algebraically into a single plain
  3x3 conv (switch_to_deploy()), giving a pure conv+PReLU stack:

      conv3x3(in->C) PReLU [conv3x3(C->C) PReLU] x num_convs
      conv3x3(C->r^2*out) PixelShuffle(r) + bilinear(x)

Why it is faster than DIS at matched quality:
- All convolutions run at LR; DIS ran its tail conv at HR (r^2 x the pixels).
- The upsampler is conv(C -> r^2*out_ch) + PixelShuffle; DIS used
  conv(C -> r^2*C) + PixelShuffle + PReLU + HR tail, which alone was ~19%
  of DIS Fast's per-pixel MACs.
- No fusion conv, no long skip: the plain stack is what TensorRT/GLSL
  actually execute, and reparameterized training recovers the quality.

Per-LR-pixel MACs at 2x (body dominates):
  DIS  Fast  (32f, 8 blocks):  ~198K
  DIS2 Fast  (32f, 6 convs):   ~60K   (3.3x fewer, 8 conv layers vs 20)
  DIS2 Balanced (48f, 8 convs): ~172K vs DIS Balanced ~272K

The deploy graph uses only Conv / PReLU / Add / DepthToSpace / Resize, so
tools/export_onnx-style export, TensorRT, and tools/export_glsl.py all work
unchanged. FP16-safe: no batchnorm, PReLU activations, bounded magnitudes.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Registries are only needed inside traiNNer-redux; keep the module
# importable standalone (export tools, tests).
try:
    from traiNNer.utils.registry import ARCH_REGISTRY, SPANDREL_REGISTRY
except ImportError:  # standalone use
    class _NoOpRegistry:
        def register(self, obj=None, **kwargs):
            if obj is None:
                return lambda f: f
            return obj

    ARCH_REGISTRY = _NoOpRegistry()
    SPANDREL_REGISTRY = _NoOpRegistry()


# =============================================================================
# Reparameterizable branches (ECBSR-style)
# =============================================================================

class SeqConv3x3(nn.Module):
    """A 1x1 conv followed by either a 3x3 conv or a fixed edge filter.

    Collapses to a single 3x3 conv via rep_params(). The training-time
    forward pads the intermediate activation with its bias value so that
    training and fused inference are numerically identical.
    """

    def __init__(self, seq_type: str, in_channels: int, out_channels: int,
                 depth_multiplier: float = 2.0):
        super().__init__()
        self.seq_type = seq_type
        self.in_channels = in_channels
        self.out_channels = out_channels

        if seq_type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_channels * depth_multiplier)
            conv0 = nn.Conv2d(in_channels, self.mid_planes, 1)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = nn.Conv2d(self.mid_planes, out_channels, 3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
        else:
            conv0 = nn.Conv2d(in_channels, out_channels, 1)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # learnable per-channel scale on a fixed edge kernel
            scale = torch.randn(out_channels, 1, 1, 1) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(out_channels) * 1e-3
            self.bias = nn.Parameter(bias)

            mask = torch.zeros(out_channels, 1, 3, 3)
            if seq_type == 'conv1x1-sobelx':
                mask[:, 0, 0, 0] = 1.0
                mask[:, 0, 1, 0] = 2.0
                mask[:, 0, 2, 0] = 1.0
                mask[:, 0, 0, 2] = -1.0
                mask[:, 0, 1, 2] = -2.0
                mask[:, 0, 2, 2] = -1.0
            elif seq_type == 'conv1x1-sobely':
                mask[:, 0, 0, 0] = 1.0
                mask[:, 0, 0, 1] = 2.0
                mask[:, 0, 0, 2] = 1.0
                mask[:, 0, 2, 0] = -1.0
                mask[:, 0, 2, 1] = -2.0
                mask[:, 0, 2, 2] = -1.0
            elif seq_type == 'conv1x1-laplacian':
                mask[:, 0, 0, 1] = 1.0
                mask[:, 0, 1, 0] = 1.0
                mask[:, 0, 1, 2] = 1.0
                mask[:, 0, 2, 1] = 1.0
                mask[:, 0, 1, 1] = -4.0
            else:
                raise ValueError(f'Unknown seq_type: {seq_type}')
            self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = F.conv2d(x, self.k0, self.b0)
        # zero-pad, then overwrite the border with the bias value: this makes
        # the two-conv sequence exactly equal to the fused zero-padded 3x3
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0 = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0
        y0[:, :, -1:, :] = b0
        y0[:, :, :, 0:1] = b0
        y0[:, :, :, -1:] = b0

        if self.seq_type == 'conv1x1-conv3x3':
            return F.conv2d(y0, self.k1, self.b1)
        return F.conv2d(y0, self.scale * self.mask, self.bias,
                        groups=self.out_channels)

    def rep_params(self):
        device = self.k0.device
        if self.seq_type == 'conv1x1-conv3x3':
            # fold the 1x1 into the 3x3: contract over mid channels
            rk = F.conv2d(self.k1, self.k0.permute(1, 0, 2, 3))
            rb = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rb = F.conv2d(rb, self.k1).view(-1) + self.b1
        else:
            tmp = self.scale * self.mask  # [out, 1, 3, 3] depthwise
            k1 = torch.zeros(self.out_channels, self.out_channels, 3, 3, device=device)
            for i in range(self.out_channels):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            rk = F.conv2d(k1, self.k0.permute(1, 0, 2, 3))
            rb = torch.ones(1, self.out_channels, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rb = F.conv2d(rb, k1).view(-1) + self.bias
        return rk, rb


class ECB(nn.Module):
    """Edge-oriented Convolution Block: multi-branch during training,
    a single plain 3x3 conv after switch_to_deploy()/rep_params()."""

    def __init__(self, in_channels: int, out_channels: int,
                 depth_multiplier: float = 2.0, with_idt: bool = True,
                 deploy: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        self.with_idt = with_idt and (in_channels == out_channels)

        if deploy:
            self.rep_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
            self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', in_channels, out_channels,
                                          depth_multiplier)
            self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', in_channels, out_channels)
            self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', in_channels, out_channels)
            self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.rep_conv(x)
        y = (self.conv3x3(x)
             + F.conv2d(x, F.pad(self.conv1x1.weight, (1, 1, 1, 1)), self.conv1x1.bias,
                        padding=1)
             + self.conv1x1_3x3(x)
             + self.conv1x1_sbx(x)
             + self.conv1x1_sby(x)
             + self.conv1x1_lpl(x))
        if self.with_idt:
            y = y + x
        return y

    def rep_params(self):
        k = self.conv3x3.weight.clone()
        b = self.conv3x3.bias.clone()

        k += F.pad(self.conv1x1.weight, (1, 1, 1, 1))
        b += self.conv1x1.bias

        for branch in (self.conv1x1_3x3, self.conv1x1_sbx, self.conv1x1_sby, self.conv1x1_lpl):
            rk, rb = branch.rep_params()
            k += rk
            b += rb

        if self.with_idt:
            for i in range(self.out_channels):
                k[i, i, 1, 1] += 1.0
        return k, b

    def switch_to_deploy(self):
        if self.deploy:
            return
        k, b = self.rep_params()
        self.rep_conv = nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1)
        self.rep_conv.weight.data = k
        self.rep_conv.bias.data = b
        for attr in ('conv3x3', 'conv1x1', 'conv1x1_3x3', 'conv1x1_sbx',
                     'conv1x1_sby', 'conv1x1_lpl'):
            self.__delattr__(attr)
        self.deploy = True


# =============================================================================
# DIS2
# =============================================================================

class DIS2(nn.Module):
    """
    DIS2: reparameterized plain-conv SR network.

    Args:
        in_channels: Input image channels (3 for RGB, 1 for luma)
        out_channels: Output image channels
        num_features: Feature width C
        num_convs: Number of C->C body convs (network depth = num_convs + 2)
        scale: 1, 2, 3 or 4 (single-shot PixelShuffle)
        depth_multiplier: Expansion in the 1x1->3x3 rep branch (training only)
        deploy: Build directly in fused inference form
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 32,
        num_convs: int = 6,
        scale: int = 2,
        depth_multiplier: float = 2.0,
        deploy: bool = False,
    ):
        super().__init__()
        if scale not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported scale factor: {scale}")

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.head = ECB(in_channels, num_features, depth_multiplier, deploy=deploy)
        self.head_act = nn.PReLU(num_features)

        self.body = nn.ModuleList(
            [ECB(num_features, num_features, depth_multiplier, deploy=deploy)
             for _ in range(num_convs)])
        self.body_acts = nn.ModuleList(
            [nn.PReLU(num_features) for _ in range(num_convs)])

        self.tail = ECB(num_features, out_channels * scale * scale,
                        depth_multiplier, deploy=deploy)
        self.upsampler = nn.PixelShuffle(scale) if scale > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            base = x
        else:
            base = F.interpolate(x, scale_factor=self.scale, mode='bilinear',
                                 align_corners=False)

        y = self.head_act(self.head(x))
        for conv, act in zip(self.body, self.body_acts):
            y = act(conv(y))
        y = self.upsampler(self.tail(y))
        return y + base

    def switch_to_deploy(self):
        """Collapse all rep branches into plain 3x3 convs (in place)."""
        for m in self.modules():
            if isinstance(m, ECB):
                m.switch_to_deploy()
        return self


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def dis2_fast(
    in_channels: int = 3,
    out_channels: int = 3,
    num_features: int = 32,
    num_convs: int = 6,
    scale: int = 2,
    deploy: bool = False,
) -> DIS2:
    return DIS2(
        in_channels=in_channels,
        out_channels=out_channels,
        num_features=num_features,
        num_convs=num_convs,
        scale=scale,
        deploy=deploy,
    )


@ARCH_REGISTRY.register()
@SPANDREL_REGISTRY.register()
def dis2_balanced(
    in_channels: int = 3,
    out_channels: int = 3,
    num_features: int = 48,
    num_convs: int = 8,
    scale: int = 2,
    deploy: bool = False,
) -> DIS2:
    return DIS2(
        in_channels=in_channels,
        out_channels=out_channels,
        num_features=num_features,
        num_convs=num_convs,
        scale=scale,
        deploy=deploy,
    )
