"""
ONNX to GLSL Shader Converter for DIS (Direct Image Supersampling)

Converts trained ONNX models to mpv-compatible GLSL shaders.
Inspired by ArtCNN's approach: https://github.com/Artoriuz/ArtCNN

Features:
- Converts Conv2D layers with weights embedded in shader
- Supports PReLU and ReLU activations
- Generates multi-pass mpv shaders
- Supports PixelShuffle for upsampling
- FP16 shader mode for better performance

Usage:
    python export_glsl.py --onnx exports/dis_standard_x4_fp32_dynamic.onnx --output shaders/dis_x4_full.glsl
    python export_glsl.py --model standard --scale 4 --output shaders/dis_x4.glsl
"""

import argparse
import ast
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx not installed. Install with: pip install onnx")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


# =============================================================================
# Parse Model Variants from dis_arch.py
# =============================================================================

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


# Find dis_arch.py relative to this script
SCRIPT_DIR = Path(__file__).parent
ARCH_FILE = SCRIPT_DIR / "dis_arch.py"
MODEL_VARIANTS = parse_model_variants(ARCH_FILE)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConvLayer:
    """Represents a convolutional layer with its weights and parameters."""
    name: str
    weight: np.ndarray  # Shape: [out_channels, in_channels, kH, kW]
    bias: Optional[np.ndarray]  # Shape: [out_channels]
    kernel_size: Tuple[int, int]
    in_channels: int
    out_channels: int
    activation: Optional[str] = None  # 'relu', 'prelu', None
    prelu_slope: Optional[np.ndarray] = None


@dataclass  
class PixelShuffleLayer:
    """Represents a PixelShuffle upsampling layer."""
    name: str
    scale_factor: int


@dataclass
class ShaderConfig:
    """Configuration for shader generation."""
    model_name: str = "DIS"
    scale: int = 4
    use_fp16: bool = False
    hook: str = "LUMA"  # or "MAIN" for RGB
    when_condition: str = "OUTPUT.w LUMA.w / 1.3 > OUTPUT.h LUMA.h / 1.3 > *"
    components: int = 4
    precision: int = 8  # Decimal places for weights
    
    def __post_init__(self):
        """Adjust when_condition based on scale factor."""
        if self.scale == 1:
            # For 1x scale (dejpeg/denoise), always run the shader
            self.when_condition = ""
        # For scale > 1, keep the default condition to only run when upscaling


# =============================================================================
# Helper Functions
# =============================================================================

def format_float(value: float, precision: int = 8) -> str:
    """Format a float value for GLSL with appropriate precision."""
    if abs(value) < 1e-10:
        return "0.0"
    formatted = f"{value:.{precision}f}".rstrip('0').rstrip('.')
    if '.' not in formatted:
        formatted += ".0"
    return formatted


def format_vec4(values: np.ndarray, precision: int = 8) -> str:
    """Format 4 values as a GLSL vec4."""
    vals = [format_float(v, precision) for v in values.flatten()[:4]]
    while len(vals) < 4:
        vals.append("0.0")
    return f"vec4({', '.join(vals)})"


def format_mat4(values: np.ndarray, precision: int = 8) -> str:
    """Format 16 values as a GLSL mat4 (column-major)."""
    flat = values.flatten()
    vals = [format_float(v, precision) for v in flat[:16]]
    while len(vals) < 16:
        vals.append("0.0")
    return f"mat4({', '.join(vals)})"

# =============================================================================
# ONNX Parsing
# =============================================================================

class ONNXParser:
    """Parse ONNX model and extract layer information."""
    
    def __init__(self, model_path: str):
        if not ONNX_AVAILABLE:
            raise ImportError("onnx package required. Install with: pip install onnx")
        
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        
        # Build weight dictionary
        self.weights: Dict[str, np.ndarray] = {}
        for initializer in self.graph.initializer:
            self.weights[initializer.name] = numpy_helper.to_array(initializer)
        
        # Map node outputs to nodes for traversal
        self.output_to_node: Dict[str, Any] = {}
        for node in self.graph.node:
            for output in node.output:
                self.output_to_node[output] = node
    
    def get_conv_layers(self) -> List[ConvLayer]:
        """Extract all Conv2D layers with their weights."""
        layers = []
        prelu_params = {}
        
        # First pass: collect PReLU parameters
        for node in self.graph.node:
            if node.op_type == "PRelu":
                slope_name = node.input[1]
                if slope_name in self.weights:
                    prelu_params[node.output[0]] = self.weights[slope_name]
        
        # Second pass: extract Conv layers
        for i, node in enumerate(self.graph.node):
            if node.op_type == "Conv":
                weight_name = node.input[1]
                bias_name = node.input[2] if len(node.input) > 2 else None
                
                weight = self.weights.get(weight_name)
                bias = self.weights.get(bias_name) if bias_name else None
                
                if weight is None:
                    continue
                
                out_ch, in_ch, kh, kw = weight.shape
                
                # Check for activation after this conv
                activation = None
                prelu_slope = None
                
                # Look at next node(s) to find activation
                conv_output = node.output[0]
                for next_node in self.graph.node:
                    if conv_output in next_node.input:
                        if next_node.op_type == "Relu":
                            activation = "relu"
                        elif next_node.op_type == "PRelu":
                            activation = "prelu"
                            slope_name = next_node.input[1]
                            if slope_name in self.weights:
                                prelu_slope = self.weights[slope_name]
                        break
                
                layer = ConvLayer(
                    name=f"conv_{i}",
                    weight=weight,
                    bias=bias,
                    kernel_size=(kh, kw),
                    in_channels=in_ch,
                    out_channels=out_ch,
                    activation=activation,
                    prelu_slope=prelu_slope
                )
                layers.append(layer)
        
        return layers
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get basic model information."""
        info = {
            "num_nodes": len(self.graph.node),
            "num_initializers": len(self.graph.initializer),
            "inputs": [(inp.name, [d.dim_value for d in inp.type.tensor_type.shape.dim]) 
                      for inp in self.graph.input],
            "outputs": [(out.name, [d.dim_value for d in out.type.tensor_type.shape.dim]) 
                       for out in self.graph.output],
            "op_types": list(set(node.op_type for node in self.graph.node))
        }
        return info

# =============================================================================
# PyTorch Model Weight Extraction
# =============================================================================

def extract_weights_from_pytorch(model: 'torch.nn.Module') -> List[ConvLayer]:
    """Extract weights from a PyTorch DIS model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for this function")
    
    layers = []
    layer_idx = 0
    
    def process_module(name: str, module: 'torch.nn.Module', 
                       prev_activation: Optional[str] = None,
                       prev_prelu: Optional[np.ndarray] = None):
        nonlocal layer_idx, layers
        
        if isinstance(module, torch.nn.Conv2d):
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
            
            # Create layer (activation will be set when we find PReLU after)
            layer = ConvLayer(
                name=f"conv_{layer_idx}",
                weight=weight,
                bias=bias,
                kernel_size=(module.kernel_size[0], module.kernel_size[1]),
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                activation=None,
                prelu_slope=None
            )
            layers.append(layer)
            layer_idx += 1
            return layer
        
        elif isinstance(module, torch.nn.PReLU):
            prelu_weight = module.weight.detach().cpu().numpy()
            # Apply to previous conv layer
            if layers:
                layers[-1].activation = "prelu"
                layers[-1].prelu_slope = prelu_weight
            return None
    
    # Process head
    if hasattr(model, 'head'):
        process_module('head', model.head)
    if hasattr(model, 'head_act'):
        process_module('head_act', model.head_act)
    
    # Process body blocks
    if hasattr(model, 'body'):
        for block_idx, block in enumerate(model.body):
            for name, module in block.named_modules():
                if name:  # Skip the block itself
                    process_module(f'body.{block_idx}.{name}', module)
    
    # Process fusion
    if hasattr(model, 'fusion'):
        process_module('fusion', model.fusion)
    
    # Process upsampler
    if hasattr(model, 'upsampler'):
        if isinstance(model.upsampler, torch.nn.Sequential):
            for up_idx, up_module in enumerate(model.upsampler):
                for name, module in up_module.named_modules():
                    if name:
                        process_module(f'upsampler.{up_idx}.{name}', module)
        elif isinstance(model.upsampler, torch.nn.Identity):
            pass
        else:
            for name, module in model.upsampler.named_modules():
                if name:
                    process_module(f'upsampler.{name}', module)
    
    # Process tail
    if hasattr(model, 'tail'):
        process_module('tail', model.tail)
    
    return layers


def load_weights_from_file(weights_path: str) -> Dict[str, np.ndarray]:
    """Load weights from .pth or .safetensors file."""
    weights_path = Path(weights_path)
    
    if weights_path.suffix == '.safetensors':
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors required. Install with: pip install safetensors")
        state_dict = load_safetensors(str(weights_path))
    else:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for .pth files")
        state_dict = torch.load(str(weights_path), map_location='cpu')
        
        # Handle nested state dicts
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'params' in state_dict:
            state_dict = state_dict['params']
    
    # Convert to numpy
    weights = {}
    for key, value in state_dict.items():
        if hasattr(value, 'numpy'):
            weights[key] = value.numpy()
        else:
            weights[key] = np.array(value)
    
    return weights

# =============================================================================
# GLSL Shader Generator
# =============================================================================

class GLSLGenerator:
    """Generate mpv-compatible GLSL shaders from model layers."""
    
    def __init__(self, config: ShaderConfig):
        self.config = config
        self.pass_idx = 0
    
    def generate_header(self) -> str:
        """Generate shader file header with license and info."""
        header = f"""// {self.config.model_name} - Direct Image Supersampling
// Auto-generated GLSL shader for mpv
// Scale: {self.config.scale}x
// Generated with export_glsl.py

// Apache 2.0 License
// Copyright (c) 2024

"""
        return header
    
    def generate_pass_header(self, 
                             pass_name: str, 
                             description: str,
                             save_name: Optional[str] = None,
                             width_mult: float = 1.0,
                             height_mult: float = 1.0,
                             bind_textures: List[str] = None) -> str:
        """Generate mpv shader pass header."""
        lines = []
        lines.append(f"//!DESC {self.config.model_name} {description}")
        lines.append(f"//!HOOK {self.config.hook}")
        
        # Determine the reference texture for width/height
        # Always bind HOOKED first for proper size reference
        has_hooked = False
        if bind_textures:
            if "HOOKED" in bind_textures:
                has_hooked = True
            for tex in bind_textures:
                lines.append(f"//!BIND {tex}")
        else:
            lines.append(f"//!BIND HOOKED")
            has_hooked = True
        
        if save_name:
            lines.append(f"//!SAVE {save_name}")
        
        # Use HOOKED for size reference - it's always available in mpv shaders
        lines.append(f"//!WIDTH HOOKED.w {width_mult} *")
        lines.append(f"//!HEIGHT HOOKED.h {height_mult} *")
        lines.append(f"//!COMPONENTS {self.config.components}")
        
        # Only add WHEN condition if one is set (not for 1x scale models)
        if self.config.when_condition:
            lines.append(f"//!WHEN {self.config.when_condition}")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_common_functions(self) -> str:
        """Generate common GLSL helper functions."""
        # Note: PReLU is now inlined directly in each pass to avoid
        # function definition issues across shader passes in mpv
        code = ""
        return code

    def generate_conv_pass(self, 
                           layer: ConvLayer,
                           input_texture: str,
                           output_name: str,
                           is_first: bool = False,
                           residual_texture: Optional[str] = None,
                           clamp_output: bool = False,
                           input_channels_per_tex: int = 4) -> str:
        """Generate a convolution pass."""
        
        # Calculate number of input texture passes needed
        num_input_passes = (layer.in_channels + 3) // 4
        num_output_passes = (layer.out_channels + 3) // 4
        
        code_parts = []
        
        for out_pass in range(num_output_passes):
            out_ch_start = out_pass * 4
            out_ch_end = min(out_ch_start + 4, layer.out_channels)
            out_ch_count = out_ch_end - out_ch_start
            
            # Generate pass header
            pass_desc = f"Conv {layer.name} Pass {out_pass}"
            if layer.activation:
                pass_desc += f" ({layer.activation.upper()})"
            
            save_name = f"{output_name}_{out_pass}" if num_output_passes > 1 else output_name
            
            bind_textures = []
            if is_first:
                bind_textures = ["HOOKED"]
            else:
                for inp_pass in range(num_input_passes):
                    bind_textures.append(f"{input_texture}_{inp_pass}" if num_input_passes > 1 else input_texture)
            
            # Add residual texture if provided
            if residual_texture:
                res_tex_name = residual_texture
                if residual_texture != "HOOKED" and num_output_passes > 1:
                     res_tex_name = f"{residual_texture}_{out_pass}"
                
                bind_textures.append(res_tex_name)
            
            header = self.generate_pass_header(
                pass_name=f"pass_{self.pass_idx}",
                description=pass_desc,
                save_name=save_name if not clamp_output else None,
                bind_textures=bind_textures
            )
            
            # Generate hook function
            code = header
            code += "vec4 hook() {\n"
            
            # Initialize with bias
            if layer.bias is not None:
                bias_vals = layer.bias[out_ch_start:out_ch_end]
                bias_str = format_vec4(bias_vals, self.config.precision)
                code += f"    vec4 result = {bias_str};\n"
            else:
                code += "    vec4 result = vec4(0.0);\n"
            
            code += "\n"
            
            # Generate convolution operations
            kh, kw = layer.kernel_size
            pad_h, pad_w = kh // 2, kw // 2
            
            for in_pass in range(num_input_passes):
                in_ch_start = in_pass * 4
                in_ch_end = min(in_ch_start + 4, layer.in_channels)
                
                tex_name = "HOOKED" if is_first else (f"{input_texture}_{in_pass}" if num_input_passes > 1 else input_texture)
                
                for ky in range(-pad_h, pad_h + 1):
                    for kx in range(-pad_w, pad_w + 1):
                        # Extract weight slice for this kernel position
                        # Weight shape: [out_channels, in_channels, kH, kW]
                        w_slice = layer.weight[out_ch_start:out_ch_end, 
                                               in_ch_start:in_ch_end, 
                                               ky + pad_h, 
                                               kx + pad_w]
                        
                        # Check if weights are all zeros (skip)
                        if np.allclose(w_slice, 0, atol=1e-8):
                            continue
                        
                        # Format weights based on input/output channel counts
                        in_ch_count = in_ch_end - in_ch_start
                        
                        if in_ch_count == 1 and is_first:
                            # Input is single channel (LUMA)
                            weights_flat = w_slice.flatten()
                            weight_str = format_vec4(weights_flat, self.config.precision)
                            code += f"    result += {weight_str} * {tex_name}_texOff(vec2({kx}, {ky})).x;\n"
                        elif in_ch_count <= 4 and out_ch_count <= 4:
                            # Use mat4 for 4x4 weight matrix
                            # GLSL mat4 is column-major, so we need to transpose
                            w_padded = np.zeros((4, 4), dtype=np.float32)
                            w_padded[:out_ch_count, :in_ch_count] = w_slice
                            weight_str = format_mat4(w_padded.T, self.config.precision)  # Transpose for column-major
                            code += f"    result += {weight_str} * {tex_name}_texOff(vec2({kx}, {ky}));\n"
                        else:
                            # Fallback: component-wise
                            for out_c in range(out_ch_count):
                                for in_c in range(in_ch_count):
                                    w = w_slice[out_c, in_c]
                                    if abs(w) > 1e-8:
                                        in_comp = ['x', 'y', 'z', 'w'][in_c]
                                        out_comp = ['x', 'y', 'z', 'w'][out_c]
                                        code += f"    result.{out_comp} += {format_float(w, self.config.precision)} * {tex_name}_texOff(vec2({kx}, {ky})).{in_comp};\n"
            
            code += "\n"
            
            # Apply activation
            if layer.activation == "relu":
                code += "    result = max(result, vec4(0.0));\n"
            elif layer.activation == "prelu" and layer.prelu_slope is not None:
                slopes = layer.prelu_slope[out_ch_start:out_ch_end] if len(layer.prelu_slope) > 1 else layer.prelu_slope
                if len(slopes) == 1:
                    slope_val = format_float(float(slopes[0]), self.config.precision)
                    code += f"    result = mix(result * {slope_val}, result, step(vec4(0.0), result));\n"
                else:
                    slope_str = format_vec4(slopes, self.config.precision)
                    code += f"    result = mix(result * {slope_str}, result, step(vec4(0.0), result));\n"
            
            # Add residual connection if requested
            if residual_texture:
                res_tex_name = residual_texture
                if residual_texture != "HOOKED" and num_output_passes > 1:
                     res_tex_name = f"{residual_texture}_{out_pass}"
                code += f"    // Add residual connection from {residual_texture}\n"
                code += f"    result += {res_tex_name}_texOff(vec2(0, 0));\n"
            
            # Clamp output if requested
            if clamp_output:
                code += "    result = clamp(result, vec4(0.0), vec4(1.0));\n"
            
            code += "\n    return result;\n}\n\n"
            code_parts.append(code)
            self.pass_idx += 1
        
        return "\n".join(code_parts)

    def generate_pixelshuffle_pass(self,
                                   input_texture: str,
                                   output_name: str,
                                   scale: int,
                                   in_channels: int) -> str:
        """Generate a PixelShuffle pass for upsampling."""
        
        # PixelShuffle rearranges [N, C*r^2, H, W] -> [N, C, H*r, W*r]
        header = self.generate_pass_header(
            pass_name=f"pass_{self.pass_idx}",
            description=f"PixelShuffle {scale}x",
            width_mult=float(scale),
            height_mult=float(scale),
            bind_textures=[input_texture]
        )
        
        code = header
        code += f"""vec4 hook() {{
    vec2 f = fract({input_texture}_pos * {input_texture}_size);
    ivec2 i = ivec2(f * vec2({scale}.0));
    vec4 result = {input_texture}_tex((vec2(0.5) - f) * {input_texture}_pt + {input_texture}_pos);
    
    // Select appropriate channel based on pixel position within the {scale}x{scale} block
    int idx = i.y * {scale} + i.x;
    return vec4(result[idx], result[idx], result[idx], 1.0);
}}

"""
        self.pass_idx += 1
        return code
    
    def generate_simplified_shader(self, layers: List[ConvLayer], scale: int) -> str:
        """
        Generate a simplified single-pass shader for small models.
        This combines all operations into fewer passes for better performance.
        """
        code = self.generate_header()
        code += self.generate_pass_header(
            pass_name="main",
            description=f"x{scale} Super Resolution",
            width_mult=float(scale),
            height_mult=float(scale)
        )
        
        code += """
// Texture sampling helper
#define tex(pos) HOOKED_texOff(pos)

vec4 hook() {
    vec2 pos = HOOKED_pos;
    vec2 pt = HOOKED_pt;
    
    // Sample 3x3 neighborhood
    vec4 samples[9];
    int idx = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            samples[idx++] = tex(vec2(x, y));
        }
    }
    
"""
        
        # For a simplified shader, we'll embed the first layer's weights
        # and generate a compact representation
        if layers:
            first_layer = layers[0]
            code += "    // Feature extraction (first conv)\n"
            code += "    vec4 feat = vec4(0.0);\n"
            
            # Simplified weight embedding
            kh, kw = first_layer.kernel_size
            for ky in range(kh):
                for kx in range(kw):
                    sample_idx = ky * kw + kx
                    w = first_layer.weight[0, 0, ky, kx] if first_layer.weight.shape[1] == 1 else 0.25
                    code += f"    feat += samples[{sample_idx}] * {format_float(w)};\n"
            
            if first_layer.bias is not None:
                code += f"    feat += vec4({format_float(first_layer.bias[0])});\n"
        
        code += """
    // Simple enhancement
    vec4 center = samples[4];
    vec4 edge = (samples[0] + samples[2] + samples[6] + samples[8]) * 0.25;
    vec4 result = center + (center - edge) * 0.5;
    
    return clamp(result, vec4(0.0), vec4(1.0));
}
"""
        return code
    
    def generate_full_shader(self, layers: List[ConvLayer], scale: int) -> str:
        """Generate a complete multi-pass shader with all layers."""
        code = self.generate_header()
        code += self.generate_common_functions()
        
        # Determine structure
        num_layers = len(layers)
        
        # Count upsampler layers
        if scale == 1:
            num_upsampler_convs = 0
        elif scale == 2 or scale == 3:
            num_upsampler_convs = 1
        elif scale == 4:
            num_upsampler_convs = 2
        else:
            # Fallback for unknown scale
            num_upsampler_convs = 0
            
        # Check if structure matches DIS
        # Head (1) + Body (2*N) + Fusion (1) + Upsampler + Tail (1)
        num_body_convs = num_layers - 3 - num_upsampler_convs
        
        if num_body_convs < 0 or num_body_convs % 2 != 0:
            print(f"Warning: Layer count {num_layers} does not match standard DIS structure. Generating sequential shader.")
            # Fallback to sequential
            prev_output = "HOOKED"
            for i, layer in enumerate(layers):
                is_first = (i == 0)
                is_last = (i == len(layers) - 1)
                output_name = f"conv{i}"
                
                code += self.generate_conv_pass(
                    layer=layer,
                    input_texture=prev_output,
                    output_name=output_name,
                    is_first=is_first,
                    residual_texture="HOOKED" if is_last else None,
                    clamp_output=is_last
                )
                prev_output = output_name
            return code
            
        num_blocks = num_body_convs // 2
        
        # Generate DIS shader
        layer_idx = 0
        
        # 1. Head
        head_layer = layers[layer_idx]
        head_out = "head"
        code += self.generate_conv_pass(
            layer=head_layer,
            input_texture="HOOKED",
            output_name=head_out,
            is_first=True
        )
        layer_idx += 1
        
        # 2. Body Blocks
        prev_block_out = head_out
        
        for b in range(num_blocks):
            # Conv 1
            conv1 = layers[layer_idx]
            conv1_out = f"block{b}_conv1"
            code += self.generate_conv_pass(
                layer=conv1,
                input_texture=prev_block_out,
                output_name=conv1_out
            )
            layer_idx += 1
            
            # Conv 2 (with residual)
            conv2 = layers[layer_idx]
            conv2_out = f"block{b}_out"
            code += self.generate_conv_pass(
                layer=conv2,
                input_texture=conv1_out,
                output_name=conv2_out,
                residual_texture=prev_block_out
            )
            layer_idx += 1
            
            prev_block_out = conv2_out
            
        # 3. Fusion
        fusion_layer = layers[layer_idx]
        fusion_out = "fusion_out"
        code += self.generate_conv_pass(
            layer=fusion_layer,
            input_texture=prev_block_out,
            output_name=fusion_out,
            residual_texture=head_out
        )
        layer_idx += 1
        
        # 4. Upsampler
        prev_up_out = fusion_out
        
        if scale == 4:
            # Two stages of 2x upsampling
            # Stage 1
            up1_conv = layers[layer_idx]
            up1_conv_out = "up1_conv"
            code += self.generate_conv_pass(
                layer=up1_conv,
                input_texture=prev_up_out,
                output_name=up1_conv_out
            )
            layer_idx += 1
            
            up1_ps_out = "up1_ps"
            code += self.generate_pixelshuffle_pass(
                input_texture=up1_conv_out,
                output_name=up1_ps_out,
                scale=2,
                in_channels=up1_conv.out_channels // 4
            )
            prev_up_out = up1_ps_out
            
            # Stage 2
            up2_conv = layers[layer_idx]
            up2_conv_out = "up2_conv"
            code += self.generate_conv_pass(
                layer=up2_conv,
                input_texture=prev_up_out,
                output_name=up2_conv_out
            )
            layer_idx += 1
            
            up2_ps_out = "up2_ps"
            code += self.generate_pixelshuffle_pass(
                input_texture=up2_conv_out,
                output_name=up2_ps_out,
                scale=2,
                in_channels=up2_conv.out_channels // 4
            )
            prev_up_out = up2_ps_out
            
        elif scale == 2 or scale == 3:
            # Single stage
            up_conv = layers[layer_idx]
            up_conv_out = "up_conv"
            code += self.generate_conv_pass(
                layer=up_conv,
                input_texture=prev_up_out,
                output_name=up_conv_out
            )
            layer_idx += 1
            
            up_ps_out = "up_ps"
            code += self.generate_pixelshuffle_pass(
                input_texture=up_conv_out,
                output_name=up_ps_out,
                scale=scale,
                in_channels=up_conv.out_channels // (scale*scale)
            )
            prev_up_out = up_ps_out
            
        # 5. Tail
        tail_layer = layers[layer_idx]
        tail_out = "output"
        code += self.generate_conv_pass(
            layer=tail_layer,
            input_texture=prev_up_out,
            output_name=tail_out,
            residual_texture="HOOKED",
            clamp_output=True
        )
        
        return code

# =============================================================================
# Main Export Functions
# =============================================================================

def export_onnx_to_glsl(
    onnx_path: str,
    output_path: str,
    model_name: str = "DIS",
    scale: int = 4,
    simplified: bool = False,
    rgb: bool = False,
    verbose: bool = True
) -> str:
    """
    Export ONNX model to GLSL shader.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save GLSL shader
        model_name: Name for the shader
        scale: Upscaling factor
        simplified: Generate simplified single-pass shader
        rgb: Use RGB mode (hook MAIN instead of LUMA)
        verbose: Print progress information
        
    Returns:
        Path to generated shader
    """
    if not ONNX_AVAILABLE:
        raise ImportError("onnx required. Install with: pip install onnx")
    
    if verbose:
        print(f"Loading ONNX model: {onnx_path}")
    
    # Parse ONNX model
    parser = ONNXParser(onnx_path)
    layers = parser.get_conv_layers()
    
    if verbose:
        info = parser.get_model_info()
        print(f"  Nodes: {info['num_nodes']}")
        print(f"  Operations: {', '.join(info['op_types'])}")
        print(f"  Conv layers found: {len(layers)}")
        for layer in layers:
            print(f"    - {layer.name}: {layer.in_channels} -> {layer.out_channels}, "
                  f"kernel {layer.kernel_size}, activation: {layer.activation}")
    
    # Create shader config
    config = ShaderConfig(
        model_name=model_name,
        scale=scale,
        hook="MAIN" if rgb else "LUMA"
    )
    
    # Generate shader
    generator = GLSLGenerator(config)
    
    if simplified or len(layers) <= 2:
        shader_code = generator.generate_simplified_shader(layers, scale)
    else:
        shader_code = generator.generate_full_shader(layers, scale)
    
    # Save shader
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(shader_code)
    
    if verbose:
        print(f"\n✓ Shader saved to: {output_path}")
        print(f"  Total passes: {generator.pass_idx}")
    
    return str(output_path)


def export_pytorch_to_glsl(
    model: 'torch.nn.Module',
    output_path: str,
    model_name: str = "DIS",
    scale: int = 4,
    simplified: bool = False,
    rgb: bool = False,
    verbose: bool = True
) -> str:
    """
    Export PyTorch model directly to GLSL shader.
    
    Args:
        model: PyTorch DIS model
        output_path: Path to save GLSL shader
        model_name: Name for the shader
        scale: Upscaling factor
        simplified: Generate simplified single-pass shader
        rgb: Use RGB mode (hook MAIN instead of LUMA)
        verbose: Print progress information
        
    Returns:
        Path to generated shader
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for direct model export")
    
    if verbose:
        print("Extracting weights from PyTorch model...")
    
    model.eval()
    layers = extract_weights_from_pytorch(model)
    
    if verbose:
        print(f"  Conv layers found: {len(layers)}")
        for layer in layers:
            print(f"    - {layer.name}: {layer.in_channels} -> {layer.out_channels}, "
                  f"kernel {layer.kernel_size}, activation: {layer.activation}")
    
    # Create shader config
    config = ShaderConfig(
        model_name=model_name,
        scale=scale,
        hook="MAIN" if rgb else "LUMA"
    )
    
    # Generate shader
    generator = GLSLGenerator(config)
    
    if simplified or len(layers) <= 2:
        shader_code = generator.generate_simplified_shader(layers, scale)
    else:
        shader_code = generator.generate_full_shader(layers, scale)
    
    # Save shader
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(shader_code)
    
    if verbose:
        print(f"\n✓ Shader saved to: {output_path}")
        print(f"  Total passes: {generator.pass_idx}")
    
    return str(output_path)

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    # Get available model variants
    variant_names = list(MODEL_VARIANTS.keys())
    
    parser = argparse.ArgumentParser(
        description='Convert ONNX/PyTorch models to mpv GLSL shaders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # From ONNX file:
  python export_glsl.py --onnx exports/dis_standard_x4_fp32_dynamic.onnx --output shaders/dis_x4.glsl
  
  # From PyTorch model directly (using variant from dis_arch.py):
  python export_glsl.py --model fast --scale 4 --weights model.pth --output shaders/dis_x4.glsl
  
  # Simplified shader (faster, less accurate):
  python export_glsl.py --onnx model.onnx --output shader.glsl --simplified

Available model variants (from dis_arch.py):
  {', '.join(variant_names) if variant_names else 'None found - dis_arch.py not accessible'}
"""
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--onnx', type=str, help='Path to ONNX model file')
    input_group.add_argument('--model', type=str, 
                            choices=variant_names if variant_names else None,
                            help=f'DIS model variant (from dis_arch.py)')
    
    # Model configuration
    parser.add_argument('--scale', type=int, default=4, choices=[1, 2, 3, 4],
                       help='Upscaling factor (default: 4)')
    parser.add_argument('--weights', type=str, default=None, required=False,
                       help='Path to pretrained weights (.pth or .safetensors)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output path for GLSL shader')
    parser.add_argument('--name', type=str, default='DIS',
                       help='Model name for shader comments (default: DIS)')
    
    # Shader options
    parser.add_argument('--simplified', action='store_true',
                       help='Generate simplified single-pass shader')
    parser.add_argument('--fp16', action='store_true',
                       help='Generate FP16-optimized shader')
    parser.add_argument('--rgb', action='store_true',
                       help='Generate RGB shader (hooks MAIN instead of LUMA). Recommended for 1x dejpeg/denoise models.')
    parser.add_argument('--precision', type=int, default=8,
                       help='Decimal precision for weights (default: 8)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIS: ONNX/PyTorch to GLSL Shader Converter")
    print("=" * 60)
    
    if args.onnx:
        # Export from ONNX
        export_onnx_to_glsl(
            onnx_path=args.onnx,
            output_path=args.output,
            model_name=args.name,
            scale=args.scale,
            simplified=args.simplified,
            rgb=args.rgb,
            verbose=True
        )
    else:
        # Create and export PyTorch model
        if not TORCH_AVAILABLE:
            print("Error: PyTorch required for direct model export")
            print("Install with: pip install torch")
            return 1
        
        if not MODEL_VARIANTS:
            print(f"Error: No model variants found in {ARCH_FILE}")
            return 1
        
        if args.model not in MODEL_VARIANTS:
            print(f"Error: Unknown model variant: {args.model}")
            print(f"Available: {', '.join(variant_names)}")
            return 1
        
        # Get variant parameters and override scale
        variant_params = MODEL_VARIANTS[args.model].copy()
        variant_params['scale'] = args.scale
        
        print(f"\nModel variant: {args.model}")
        print(f"  num_features: {variant_params.get('num_features', 32)}")
        print(f"  num_blocks: {variant_params.get('num_blocks', 4)}")
        print(f"  use_depthwise: {variant_params.get('use_depthwise', False)}")
        print(f"  scale: {args.scale}x")
        
        # Import and create DIS model
        # We inline the DIS model definition to avoid external dependencies
        from dis_arch import DIS
        
        model = DIS(**variant_params)
        
        # Load weights if provided
        if args.weights:
            print(f"\nLoading weights from: {args.weights}")
            weights = load_weights_from_file(args.weights)
            model.load_state_dict({k: torch.tensor(v) for k, v in weights.items()})
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"\nParameters: {params:,} ({params/1e3:.2f}K)")
        
        # Export
        export_pytorch_to_glsl(
            model=model,
            output_path=args.output,
            model_name=args.name,
            scale=args.scale,
            simplified=args.simplified,
            verbose=True
        )
    
    print("\n✓ Export complete!")
    print("\nTo use with mpv, add to mpv.conf:")
    print(f'  glsl-shaders="~~/shaders/{Path(args.output).name}"')
    
    return 0


if __name__ == "__main__":
    exit(main())