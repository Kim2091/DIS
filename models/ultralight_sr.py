
"""DIS: Direct Image Supersampling

A stupidly fast super-resolution architecture.
DIS - because we DIS-regard complexity, DIS-card unnecessary layers,
and DIS-patch images at blazing speed.

Optimized for speed, FP16, TensorRT, and dynamic ONNX compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv - much faster than regular conv"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FastResBlock(nn.Module):
    """Ultra-fast residual block with minimal operations"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        # Using PReLU - single param per channel, FP16 safe
        self.act = nn.PReLU(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class LightBlock(nn.Module):
    """Lightweight block using depthwise separable convolutions"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv = DepthwiseSeparableConv(channels, channels, 3)
        self.act = nn.PReLU(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.dw_conv(x))


class PixelShuffleUpsampler(nn.Module):
    """Efficient upsampling using pixel shuffle (ESPCN style)"""
    
    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pixel_shuffle(self.conv(x)))


class DIS(nn.Module):
    """
    DIS: Direct Image Supersampling
    
    Why "DIS"?
    - DIS-regard complexity (minimal architecture)
    - DIS-card batch norm (faster, FP16 stable)
    - DIS-patch images fast (blazing inference)
    - DIS-tilled efficiency (pure, concentrated speed)
    - DIS-tinctly simple (no attention, no transformers, just convs)
    
    Design principles:
    - Minimal depth (fewer layers = faster inference)
    - No batch normalization (better for inference, FP16 stable)
    - PReLU activation (FP16 safe, learnable)
    - Pixel shuffle upsampling (efficient and high quality)
    - Global residual learning (image + learned residual)
    - Mix of regular conv and depthwise separable for speed
    
    Args:
        in_channels: Input image channels (default: 3 for RGB)
        out_channels: Output image channels (default: 3 for RGB)
        num_features: Number of feature channels (default: 32)
        num_blocks: Number of residual blocks (default: 4)
        scale: Upscaling factor (2, 3, or 4)
        use_depthwise: Use depthwise separable convs for extra speed
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 4,
        scale: int = 4,
        use_depthwise: bool = False
    ):
        super().__init__()
        
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Shallow feature extraction
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.head_act = nn.PReLU(num_features)
        
        # Feature extraction body
        if use_depthwise:
            self.body = nn.Sequential(*[LightBlock(num_features) for _ in range(num_blocks)])
        else:
            self.body = nn.Sequential(*[FastResBlock(num_features) for _ in range(num_blocks)])
        
        # Feature fusion
        self.fusion = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Upsampling - handle different scales
        if scale == 4:
            self.upsampler = nn.Sequential(
                PixelShuffleUpsampler(num_features, num_features, 2),
                PixelShuffleUpsampler(num_features, num_features, 2),
            )
        elif scale == 3:
            self.upsampler = PixelShuffleUpsampler(num_features, num_features, 3)
        elif scale == 2:
            self.upsampler = PixelShuffleUpsampler(num_features, num_features, 2)
        elif scale == 1:
            self.upsampler = nn.Identity()
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")
        
        # Final reconstruction
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bilinear upscale for global residual (fast, TensorRT friendly)
        # Using align_corners=False for ONNX compatibility
        if self.scale == 1:
            base = x
        else:
            base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        # Feature extraction
        feat = self.head_act(self.head(x))
        
        # Body with residual
        body_out = self.body(feat)
        body_out = self.fusion(body_out) + feat
        
        # Upsample and reconstruct
        out = self.upsampler(body_out)
        out = self.tail(out)
        
        # Global residual learning
        return out + base


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module, input_size: Tuple[int, int] = (64, 64)) -> dict:
    """Get model information including parameter count and FLOPs estimate"""
    params = count_parameters(model)
    
    # Rough FLOPs estimate
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, *input_size)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    except ImportError:
        flops = None
    
    return {
        "parameters": params,
        "parameters_human": f"{params / 1e6:.3f}M" if params > 1e6 else f"{params / 1e3:.2f}K",
        "flops": flops,
        "flops_human": f"{flops / 1e9:.3f}G" if flops else "N/A (install thop)"
    }


if __name__ == "__main__":
    # Test the models
    print("=" * 60)
    print("DIS: Direct Image Supersampling - Architecture Test")
    print("=" * 60)
    
    # Test standard model
    model = DIS(scale=4, num_features=32, num_blocks=4)
    info = get_model_info(model)
    print(f"\nDIS (standard):")
    print(f"  Parameters: {info['parameters_human']}")
    print(f"  FLOPs (64x64 input): {info['flops_human']}")
    
    # Test depthwise variant
    model_dw = DIS(scale=4, num_features=32, num_blocks=4, use_depthwise=True)
    info_dw = get_model_info(model_dw)
    print(f"\nDIS (depthwise):")
    print(f"  Parameters: {info_dw['parameters_human']}")
    print(f"  FLOPs (64x64 input): {info_dw['flops_human']}")
    
    # Test tiny model (DIS with num_blocks=2, num_features=16)
    model_tiny = DIS(scale=4, num_features=16, num_blocks=2)
    info_tiny = get_model_info(model_tiny)
    print(f"\nDIS_Tiny (num_blocks=2, num_features=16):")
    print(f"  Parameters: {info_tiny['parameters_human']}")
    print(f"  FLOPs (64x64 input): {info_tiny['flops_human']}")
    
    # Test forward pass
    print("\n" + "=" * 60)
    print("Forward pass test (FP16)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = model.to(device).half()
    x = torch.randn(1, 3, 64, 64, device=device, dtype=torch.float16)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Scale verified: {y.shape[-1] / x.shape[-1]}x")
    print("\n✓ FP16 forward pass successful!")
