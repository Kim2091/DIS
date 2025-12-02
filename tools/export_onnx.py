"""
ONNX Export Utilities for DIS (Direct Image Supersampling)

Features:
- Dynamic shape support (batch, height, width)
- FP16 export support
- TensorRT optimization hints
- Validation of exported model
"""

import torch
import torch.onnx
import torch.nn as nn
import torch.nn.functional as F
import argparse
import ast
import re
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx/onnxruntime not installed. Install with: pip install onnx onnxruntime-gpu")

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# ============================================================================
# Inline DIS model definition (to avoid importing dis_arch.py which has 
# external dependencies)
# ============================================================================

class DepthwiseSeparableConv(nn.Module):
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
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.act = nn.PReLU(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class LightBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dw_conv = DepthwiseSeparableConv(channels, channels, 3)
        self.act = nn.PReLU(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.dw_conv(x))


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.act = nn.PReLU(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pixel_shuffle(self.conv(x)))


class DIS(nn.Module):
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
        
        self.head = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.head_act = nn.PReLU(num_features)
        
        if use_depthwise:
            self.body = nn.Sequential(*[LightBlock(num_features) for _ in range(num_blocks)])
        else:
            self.body = nn.Sequential(*[FastResBlock(num_features) for _ in range(num_blocks)])
        
        self.fusion = nn.Conv2d(num_features, num_features, 3, padding=1)
        
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
        
        self.tail = nn.Conv2d(num_features, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1:
            base = x
        else:
            base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        feat = self.head_act(self.head(x))
        body_out = self.body(feat)
        body_out = self.fusion(body_out) + feat
        out = self.upsampler(body_out)
        out = self.tail(out)
        
        return out + base


# ============================================================================
# Parse model variants from dis_arch.py without importing
# ============================================================================

def parse_model_variants(arch_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse dis_arch.py to extract model variant definitions without importing.
    Returns dict mapping variant name to default parameters.
    """
    variants = {}
    
    if not arch_file.exists():
        return variants
    
    source = arch_file.read_text(encoding='utf-8')
    
    # Parse the source as AST
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return variants
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('dis_'):
            variant_name = node.name[4:]  # Strip 'dis_' prefix
            
            # Extract default arguments
            defaults = {}
            args = node.args
            
            # Get argument names and defaults
            num_defaults = len(args.defaults)
            num_args = len(args.args)
            
            for i, arg in enumerate(args.args):
                arg_name = arg.arg
                default_index = i - (num_args - num_defaults)
                
                if default_index >= 0:
                    default_node = args.defaults[default_index]
                    # Extract literal values
                    if isinstance(default_node, ast.Constant):
                        defaults[arg_name] = default_node.value
                    elif isinstance(default_node, ast.Num):  # Python 3.7 compat
                        defaults[arg_name] = default_node.n
                    elif isinstance(default_node, ast.NameConstant):  # Python 3.7 compat
                        defaults[arg_name] = default_node.value
            
            variants[variant_name] = defaults
    
    return variants


# Find dis_arch.py in the models directory (parent of tools)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ARCH_FILE = PROJECT_ROOT / "models" / "dis_arch.py"
MODEL_VARIANTS = parse_model_variants(ARCH_FILE)


def visualize_architecture(
    model: nn.Module,
    model_name: str = "DIS",
    scale: int = 4
) -> str:
    """
    Generate an ASCII visualization of the DIS model architecture.
    
    Args:
        model: The DIS model instance
        model_name: Name to display in the header
        scale: Upscaling factor
        
    Returns:
        ASCII art string representing the architecture
    """
    # Extract model parameters
    in_channels = getattr(model, 'in_channels', 3)
    out_channels = getattr(model, 'out_channels', 3)
    
    # Get number of features from head conv
    num_features = model.head.out_channels if hasattr(model.head, 'out_channels') else 32
    
    # Count body blocks
    num_blocks = len(model.body) if hasattr(model.body, '__len__') else 0
    
    # Detect block type
    if num_blocks > 0:
        first_block = model.body[0]
        if hasattr(first_block, 'dw_conv'):
            block_type = "LightBlock (Depthwise)"
            block_symbol = "◇"
        else:
            block_type = "FastResBlock"
            block_symbol = "■"
    else:
        block_type = "None"
        block_symbol = "○"
    
    # Count upsampler stages
    if isinstance(model.upsampler, nn.Identity):
        upsample_stages = 0
    elif isinstance(model.upsampler, nn.Sequential):
        upsample_stages = len(model.upsampler)
    else:
        upsample_stages = 1
    
    # Calculate widths for the diagram
    block_width = max(20, num_features // 2)
    total_width = max(50, block_width + 20)
    
    # Build the visualization
    lines = []
    
    # Header
    lines.append("┌" + "─" * (total_width - 2) + "┐")
    title = f" {model_name} Architecture (×{scale}) "
    padding = (total_width - 2 - len(title)) // 2
    lines.append("│" + " " * padding + title + " " * (total_width - 2 - padding - len(title)) + "│")
    lines.append("├" + "─" * (total_width - 2) + "┤")
    
    # Input
    input_str = f"Input: {in_channels}ch"
    lines.append("│" + input_str.center(total_width - 2) + "│")
    lines.append("│" + "▼".center(total_width - 2) + "│")
    
    # Head
    head_box = f"┌{'─' * (block_width - 2)}┐"
    head_label = f"│{'Head Conv3x3'.center(block_width - 2)}│"
    head_feat = f"│{f'{in_channels}→{num_features}ch'.center(block_width - 2)}│"
    head_box_end = f"└{'─' * (block_width - 2)}┘"
    
    pad = (total_width - 2 - block_width) // 2
    lines.append("│" + " " * pad + head_box + " " * (total_width - 2 - pad - block_width) + "│")
    lines.append("│" + " " * pad + head_label + " " * (total_width - 2 - pad - block_width) + "│")
    lines.append("│" + " " * pad + head_feat + " " * (total_width - 2 - pad - block_width) + "│")
    lines.append("│" + " " * pad + head_box_end + " " * (total_width - 2 - pad - block_width) + "│")
    
    # PReLU
    lines.append("│" + "▼".center(total_width - 2) + "│")
    lines.append("│" + "PReLU".center(total_width - 2) + "│")
    lines.append("│" + "▼".center(total_width - 2) + "│")
    
    # Skip connection start - build content then pad to width
    def make_line(content: str) -> str:
        """Pad content to fit within the outer box."""
        # Account for the outer │ characters
        inner_width = total_width - 2
        # Content might have unicode chars, so calculate display width
        return "│" + content.ljust(inner_width) + "│"
    
    # Body section with skip connection
    skip_prefix = " " * (pad - 4) + "───┐"
    body_top = f"├{'─' * (block_width - 2)}┤"
    lines.append(make_line(skip_prefix + body_top))
    
    # Body blocks
    if num_blocks > 0:
        # Show blocks with visual representation
        blocks_per_row = min(8, num_blocks)
        num_rows = (num_blocks + blocks_per_row - 1) // blocks_per_row
        
        for row in range(num_rows):
            start_idx = row * blocks_per_row
            end_idx = min(start_idx + blocks_per_row, num_blocks)
            blocks_in_row = end_idx - start_idx
            
            # Block symbols
            block_symbols = " ".join([block_symbol] * blocks_in_row)
            block_content = block_symbols.center(block_width - 2)
            line_content = " " * (pad - 4) + "│   " + "│" + block_content + "│"
            lines.append(make_line(line_content))
        
        # Block type label
        type_content = block_type[:block_width-4].center(block_width - 2)
        line_content = " " * (pad - 4) + "│   " + "│" + type_content + "│"
        lines.append(make_line(line_content))
        
        # Blocks count label
        blocks_content = f"×{num_blocks} blocks".center(block_width - 2)
        line_content = " " * (pad - 4) + "│   " + "│" + blocks_content + "│"
        lines.append(make_line(line_content))
    
    # Fusion
    fusion_top = f"├{'─' * (block_width - 2)}┤"
    lines.append(make_line(" " * (pad - 4) + "│   " + fusion_top))
    
    fusion_content = "Fusion Conv3x3".center(block_width - 2)
    lines.append(make_line(" " * (pad - 4) + "│   " + "│" + fusion_content + "│"))
    
    fusion_bottom = f"└{'─' * (block_width - 2)}┘"
    lines.append(make_line(" " * (pad - 4) + "│   " + fusion_bottom))
    
    # Skip connection add
    lines.append(make_line(" " * (pad - 4) + "└──────►(+)"))
    lines.append("│" + "▼".center(total_width - 2) + "│")
    
    # Upsampler
    if upsample_stages > 0:
        up_box = f"┌{'─' * (block_width - 2)}┐"
        lines.append("│" + " " * pad + up_box + " " * (total_width - 2 - pad - block_width) + "│")
        
        if scale == 4:
            up_label = f"│{'PixelShuffle ×2'.center(block_width - 2)}│"
            lines.append("│" + " " * pad + up_label + " " * (total_width - 2 - pad - block_width) + "│")
            lines.append("│" + " " * pad + f"│{'      ▼      '.center(block_width - 2)}│" + " " * (total_width - 2 - pad - block_width) + "│")
            lines.append("│" + " " * pad + up_label + " " * (total_width - 2 - pad - block_width) + "│")
        elif scale == 3:
            up_label = f"│{'PixelShuffle ×3'.center(block_width - 2)}│"
            lines.append("│" + " " * pad + up_label + " " * (total_width - 2 - pad - block_width) + "│")
        elif scale == 2:
            up_label = f"│{'PixelShuffle ×2'.center(block_width - 2)}│"
            lines.append("│" + " " * pad + up_label + " " * (total_width - 2 - pad - block_width) + "│")
        
        up_box_end = f"└{'─' * (block_width - 2)}┘"
        lines.append("│" + " " * pad + up_box_end + " " * (total_width - 2 - pad - block_width) + "│")
        lines.append("│" + "▼".center(total_width - 2) + "│")
    
    # Tail
    tail_box = f"┌{'─' * (block_width - 2)}┐"
    tail_label = f"│{'Tail Conv3x3'.center(block_width - 2)}│"
    tail_feat = f"│{f'{num_features}→{out_channels}ch'.center(block_width - 2)}│"
    tail_box_end = f"└{'─' * (block_width - 2)}┘"
    
    lines.append("│" + " " * pad + tail_box + " " * (total_width - 2 - pad - block_width) + "│")
    lines.append("│" + " " * pad + tail_label + " " * (total_width - 2 - pad - block_width) + "│")
    lines.append("│" + " " * pad + tail_feat + " " * (total_width - 2 - pad - block_width) + "│")
    lines.append("│" + " " * pad + tail_box_end + " " * (total_width - 2 - pad - block_width) + "│")
    
    # Global residual
    lines.append("│" + "▼".center(total_width - 2) + "│")
    lines.append("│" + "(+) ◄── Bilinear ×{} (global skip)".format(scale).center(total_width - 2) + "│")
    lines.append("│" + "▼".center(total_width - 2) + "│")
    
    # Output
    output_str = f"Output: {out_channels}ch (×{scale} resolution)"
    lines.append("│" + output_str.center(total_width - 2) + "│")
    
    # Footer with stats
    lines.append("├" + "─" * (total_width - 2) + "┤")
    stats = f" Features: {num_features} | Blocks: {num_blocks} | Scale: {scale}× "
    lines.append("│" + stats.center(total_width - 2) + "│")
    lines.append("└" + "─" * (total_width - 2) + "┘")
    
    return "\n".join(lines)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    scale: int = 4,
    input_shape: Tuple[int, int, int, int] = (1, 3, 64, 64),
    dynamic_axes: bool = True,
    fp16: bool = False,
    opset_version: int = 17,
    simplify: bool = True,
    verbose: bool = True
) -> str:
    """
    Export DIS model to ONNX format.
    
    Args:
        model: The PyTorch model to export
        output_path: Path to save the ONNX model
        scale: Upscaling factor (for naming)
        input_shape: Example input shape (B, C, H, W)
        dynamic_axes: Enable dynamic batch/height/width
        fp16: Export in FP16 precision
        opset_version: ONNX opset version (17+ recommended for TensorRT)
        simplify: Simplify the ONNX graph (requires onnxsim)
        verbose: Print export information
        
    Returns:
        Path to the exported ONNX model
    """
    model.eval()
    
    # Determine device and dtype
    device = next(model.parameters()).device
    dtype = torch.float16 if fp16 else torch.float32
    
    # Move model to appropriate dtype
    if fp16:
        model = model.half()
    else:
        model = model.float()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device, dtype=dtype)
    
    # Setup dynamic axes for dynamic shapes
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height_out', 3: 'width_out'}
        }
    else:
        dynamic_axes_dict = None
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Exporting model to ONNX...")
        print(f"  Input shape: {input_shape}")
        print(f"  Dynamic axes: {dynamic_axes}")
        print(f"  FP16: {fp16}")
        print(f"  Opset version: {opset_version}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_dict,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
        verbose=False
    )
    
    if verbose:
        print(f"  Exported to: {output_path}")
    
    # Simplify the model if requested
    if simplify:
        try:
            import onnxsim
            if verbose:
                print("  Simplifying ONNX model...")
            
            onnx_model = onnx.load(str(output_path))
            
            # For dynamic shapes, we need to specify input shape hints
            if dynamic_axes:
                # Use the input shape as a hint for simplification
                input_shapes = {'input': list(input_shape)}
                onnx_model, check = onnxsim.simplify(
                    onnx_model,
                    test_input_shapes=input_shapes
                )
            else:
                onnx_model, check = onnxsim.simplify(onnx_model)
            
            if check:
                onnx.save(onnx_model, str(output_path))
                if verbose:
                    print("  ✓ Model simplified successfully")
            else:
                if verbose:
                    print("  ⚠ Simplification check failed, using original")
        except ImportError:
            if verbose:
                print("  ⚠ onnxsim not installed, skipping simplification")
        except Exception as e:
            if verbose:
                print(f"  ⚠ Simplification failed: {e}")
    
    # Validate the exported model
    if ONNX_AVAILABLE:
        try:
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            if verbose:
                print("  ✓ ONNX model validation passed")
        except Exception as e:
            print(f"  ✗ ONNX validation failed: {e}")
    
    return str(output_path)


def validate_onnx_model(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    test_shapes: List[Tuple[int, int, int, int]] = None,
    fp16: bool = False,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    verbose: bool = True
) -> bool:
    """
    Validate ONNX model output matches PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        test_shapes: List of input shapes to test
        fp16: Use FP16 precision
        atol: Absolute tolerance for comparison
        rtol: Relative tolerance for comparison
        verbose: Print validation information
        
    Returns:
        True if validation passes
    """
    if not ONNX_AVAILABLE:
        print("ONNX Runtime not available for validation")
        return False
    
    if test_shapes is None:
        test_shapes = [
            (1, 3, 64, 64),
            (1, 3, 128, 128),
            (2, 3, 64, 64),
            (1, 3, 100, 150),  # Non-square test
        ]
    
    pytorch_model.eval()
    dtype = torch.float16 if fp16 else torch.float32
    device = next(pytorch_model.parameters()).device
    
    if fp16:
        pytorch_model = pytorch_model.half()
    else:
        pytorch_model = pytorch_model.float()
    
    # Setup ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return False
    
    if verbose:
        print(f"\nValidating ONNX model: {onnx_path}")
        print(f"  Providers: {session.get_providers()}")
    
    all_passed = True
    
    for shape in test_shapes:
        # Create test input
        test_input = torch.randn(*shape, device=device, dtype=dtype)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input)
        
        # ONNX inference
        input_np = test_input.cpu().numpy()
        onnx_output = session.run(None, {'input': input_np})[0]
        
        # Compare outputs
        pytorch_output_np = pytorch_output.cpu().numpy()
        
        try:
            import numpy as np
            np.testing.assert_allclose(
                onnx_output, pytorch_output_np, 
                atol=atol if not fp16 else atol * 10,  # Looser tolerance for FP16
                rtol=rtol if not fp16 else rtol * 10
            )
            status = "✓ PASS"
        except AssertionError as e:
            status = "✗ FAIL"
            all_passed = False
            if verbose:
                max_diff = np.max(np.abs(onnx_output - pytorch_output_np))
                print(f"    Max difference: {max_diff}")
        
        if verbose:
            print(f"  Shape {shape}: {status} -> output {onnx_output.shape}")
    
    return all_passed


def export_for_tensorrt(
    model: torch.nn.Module,
    output_dir: str,
    model_name: str = "ultralight_sr",
    scale: int = 4,
    fp16: bool = True,
    verbose: bool = True
) -> dict:
    """
    Export model optimized for TensorRT conversion.
    
    Creates both FP32 and FP16 ONNX files with TensorRT-friendly settings.
    
    Args:
        model: PyTorch model to export
        output_dir: Directory to save exports
        model_name: Base name for the model files
        scale: Upscaling factor
        fp16: Also export FP16 version
        verbose: Print export information
        
    Returns:
        Dictionary with paths to exported models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exports = {}
    
    # Always export FP32 version (reference)
    fp32_path = output_dir / f"{model_name}_x{scale}_fp32_dynamic.onnx"
    export_to_onnx(
        model, 
        str(fp32_path), 
        scale=scale,
        fp16=False, 
        dynamic_axes=True,
        opset_version=17,
        verbose=verbose
    )
    exports['fp32'] = str(fp32_path)
    
    # Export FP16 version
    if fp16:
        fp16_path = output_dir / f"{model_name}_x{scale}_fp16_dynamic.onnx"
        export_to_onnx(
            model, 
            str(fp16_path), 
            scale=scale,
            fp16=True, 
            dynamic_axes=True,
            opset_version=17,
            verbose=verbose
        )
        exports['fp16'] = str(fp16_path)
    
    if verbose:
        print(f"\n✓ TensorRT-ready exports saved to: {output_dir}")
        print("\nTo convert to TensorRT engine:")
        print(f"  trtexec --onnx={fp32_path} --saveEngine=model.engine --fp16")
        print("\nFor dynamic shapes with TensorRT:")
        print(f"  trtexec --onnx={fp32_path} \\")
        print(f"    --minShapes=input:1x3x64x64 \\")
        print(f"    --optShapes=input:1x3x256x256 \\")
        print(f"    --maxShapes=input:1x3x1024x1024 \\")
        print(f"    --saveEngine=model_dynamic.engine --fp16")
    
    return exports


def main():
    # Get available model variants
    variant_names = list(MODEL_VARIANTS.keys())
    
    parser = argparse.ArgumentParser(
        description='Export DIS model to ONNX',
        usage='%(prog)s input.safetensors output.onnx [options]'
    )
    parser.add_argument('input', type=str,
                        help='Path to input weights (.safetensors or .pth)')
    parser.add_argument('output', type=str,
                        help='Path to output ONNX file')
    parser.add_argument('--model', type=str, default=variant_names[0] if variant_names else 'good',
                        choices=variant_names if variant_names else None,
                        help=f'Model variant (available: {", ".join(variant_names) if variant_names else "none found"})')
    parser.add_argument('--scale', type=int, default=4, choices=[1, 2, 3, 4],
                        help='Upscaling factor (default: 4)')
    parser.add_argument('--fp16', action='store_true',
                        help='Export in FP16 precision')
    parser.add_argument('--validate', action='store_true',
                        help='Validate exported model against PyTorch')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX simplification')
    
    args = parser.parse_args()
    
    # Create model using parsed variant parameters
    print("=" * 60)
    print("DIS: Direct Image Supersampling - ONNX Export")
    print("=" * 60)
    
    if not MODEL_VARIANTS:
        raise RuntimeError(f"No model variants found in {ARCH_FILE}")
    
    if args.model not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {args.model}. Available: {', '.join(variant_names)}")
    
    # Get variant parameters and override scale
    variant_params = MODEL_VARIANTS[args.model].copy()
    variant_params['scale'] = args.scale
    
    # Create model with parsed parameters
    model = DIS(**variant_params)
    
    # Load weights
    print(f"Loading weights from: {args.input}")
    weights_path = Path(args.input)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {args.input}")
    
    if weights_path.suffix == '.safetensors':
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors not installed. Install with: pip install safetensors")
        state_dict = load_safetensors(args.input)
    else:
        state_dict = torch.load(args.input, map_location='cpu')
        # Handle nested state dict (e.g., from training checkpoints)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
    model.load_state_dict(state_dict)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}")
    print(f"Parameters: {params:,} ({params/1e3:.2f}K)")
    print(f"Scale: {args.scale}x")
    print(f"Device: {device}")
    
    # Print architecture visualization
    print("\n" + "=" * 60)
    print("Architecture Visualization")
    print("=" * 60)
    print(visualize_architecture(model, model_name=f"DIS-{args.model}", scale=args.scale))
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    export_to_onnx(
        model,
        str(output_path),
        scale=args.scale,
        fp16=args.fp16,
        dynamic_axes=True,
        opset_version=17,
        simplify=not args.no_simplify,
        verbose=True
    )
    
    # Validate if requested
    if args.validate and ONNX_AVAILABLE:
        print("\n" + "=" * 60)
        print("Validation")
        print("=" * 60)
        validate_onnx_model(str(output_path), model, fp16=args.fp16)
    
    print(f"\n✓ Export complete: {output_path}")


if __name__ == "__main__":
    main()
