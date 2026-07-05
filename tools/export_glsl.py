"""
ONNX to GLSL Shader Converter for DIS (Direct Image Supersampling)

Converts trained ONNX models to mpv-compatible GLSL shaders.
Inspired by ArtCNN's approach: https://github.com/Artoriuz/ArtCNN

Unlike the previous converter, this one walks the actual ONNX graph instead
of assuming a fixed layer layout, so it supports:
- All DIS scales (1x, 2x, 3x, 4x) including PixelShuffle (DepthToSpace) upsampling
- Regular and depthwise-separable (LightBlock) variants
- PReLU / ReLU activations, including the PReLU that follows PixelShuffle
- The bilinear global residual (Resize) via GPU bilinear sampling of HOOKED
- FP16 and FP32 ONNX weights
- LUMA (1-channel) and RGB/MAIN (3-channel) models, auto-detected

Supported ONNX ops: Conv (groups=1 or depthwise), PRelu, Relu, Add,
DepthToSpace (CRD/DCR), Resize (bilinear global residual), Constant, Identity.
Export models with tools/export_onnx.py (opset 17); older opsets that emit
Reshape/Transpose chains for PixelShuffle are not supported.

NOTE: intermediate feature maps contain negative values, so the shader needs
float FBOs. Use vo=gpu-next (default float16 FBOs), or vo=gpu with
--fbo-format=rgba16f. With mpv's default unorm FBOs on vo=gpu the features
get clamped and the output degrades subtly.

Usage:
    python tools/export_glsl.py --onnx model.onnx --output shader.glsl --name MyModel
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnx not installed. Install with: pip install onnx")


# =============================================================================
# Formatting helpers
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
    """Format up to 4 values as a GLSL vec4, zero-padded."""
    vals = [format_float(float(v), precision) for v in np.asarray(values, dtype=np.float64).flatten()[:4]]
    while len(vals) < 4:
        vals.append("0.0")
    return f"vec4({', '.join(vals)})"


def format_mat4(values: np.ndarray, precision: int = 8) -> str:
    """Format 16 values as a GLSL mat4 (column-major order expected)."""
    flat = np.asarray(values, dtype=np.float64).flatten()
    vals = [format_float(float(v), precision) for v in flat[:16]]
    while len(vals) < 16:
        vals.append("0.0")
    return f"mat4({', '.join(vals)})"


def sanitize(name: str) -> str:
    """Turn an ONNX node/tensor name into a valid GLSL identifier."""
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() else '_')
    s = ''.join(out).strip('_')
    while '__' in s:
        s = s.replace('__', '_')
    if not s or s[0].isdigit():
        s = 'x' + s
    return s


# =============================================================================
# Intermediate representation
# =============================================================================

@dataclass
class Tensor:
    """A value flowing through the graph, materialized as mpv textures.

    kind:
      'hooked'   - the graph input (HOOKED texture)
      'base'     - bilinear-upscaled graph input (virtual; sampled from HOOKED)
      'textures' - feature maps stored in ceil(channels/4) saved textures
    """
    kind: str
    channels: int
    scale: int  # spatial size relative to HOOKED
    textures: List[str] = field(default_factory=list)


@dataclass
class ConvInfo:
    weight: np.ndarray            # [out_ch, in_ch/groups, kh, kw], float32
    bias: Optional[np.ndarray]    # [out_ch] or None
    groups: int
    name: str


class ConversionError(Exception):
    pass


# =============================================================================
# Shader emission
# =============================================================================

class ShaderBuilder:
    def __init__(self, model_name: str, hook: str, scale: int, precision: int, source: str):
        self.model_name = model_name
        self.hook = hook            # 'LUMA' or 'MAIN'
        self.scale = scale
        self.precision = precision
        self.source = source
        self.passes: List[str] = []

        if scale > 1:
            # Only run when actually upscaling.
            self.when = f"OUTPUT.w {hook}.w / 1.2 > OUTPUT.h {hook}.h / 1.2 > *"
        else:
            self.when = ""

    # -- header ---------------------------------------------------------------

    def file_header(self) -> str:
        return (
            f"// {self.model_name} - DIS (Direct Image Supersampling)\n"
            f"// Auto-generated mpv GLSL shader ({self.scale}x, {self.hook} hook)\n"
            f"// Source model: {self.source}\n"
            f"// Generated by tools/export_glsl.py\n"
            f"//\n"
            f"// Requires float FBOs for intermediate feature maps:\n"
            f"//   vo=gpu-next (recommended), or vo=gpu with --fbo-format=rgba16f\n"
            f"\n"
        )

    def pass_header(self, desc: str, binds: List[str], save: Optional[str], out_scale: int) -> str:
        lines = [f"//!DESC {self.model_name} {desc}", f"//!HOOK {self.hook}"]
        seen = set()
        bind_list = []
        for tex in ["HOOKED"] + binds:
            if tex not in seen:
                seen.add(tex)
                bind_list.append(tex)
        for tex in bind_list:
            lines.append(f"//!BIND {tex}")
        if save:
            lines.append(f"//!SAVE {save}")
        if out_scale != 1:
            lines.append(f"//!WIDTH HOOKED.w {out_scale} *")
            lines.append(f"//!HEIGHT HOOKED.h {out_scale} *")
        lines.append("//!COMPONENTS 4")
        if self.when:
            lines.append(f"//!WHEN {self.when}")
        lines.append("")
        return "\n".join(lines)

    # -- activation -----------------------------------------------------------

    def _activation_code(self, act: Optional[Tuple[str, Optional[np.ndarray]]],
                         ch_start: int, ch_count: int) -> str:
        if act is None:
            return ""
        kind, slopes = act
        if kind == "relu":
            return "    result = max(result, vec4(0.0));\n"
        if kind == "prelu":
            s = np.asarray(slopes, dtype=np.float64).flatten()
            if s.size == 1:
                s = np.repeat(s, ch_start + ch_count)
            vals = s[ch_start:ch_start + ch_count]
            slope_str = format_vec4(vals, self.precision)
            return f"    result = mix(result * {slope_str}, result, step(vec4(0.0), result));\n"
        raise ConversionError(f"Unknown activation: {kind}")

    # -- conv pass ------------------------------------------------------------

    def emit_conv(self,
                  conv: ConvInfo,
                  inp: Tensor,
                  out_textures: List[str],
                  act: Optional[Tuple[str, Optional[np.ndarray]]],
                  residual: Optional[Tensor],
                  is_output: bool) -> None:
        """Emit ceil(out_ch/4) passes computing a convolution (+ fused
        activation and residual add)."""
        out_ch, w_in_ch, kh, kw = conv.weight.shape
        pad_h, pad_w = kh // 2, kw // 2
        depthwise = conv.groups > 1

        if depthwise:
            if conv.groups != out_ch or w_in_ch != 1 or inp.channels != out_ch:
                raise ConversionError(
                    f"Conv '{conv.name}': only groups=1 or full depthwise convs are supported "
                    f"(groups={conv.groups}, weight shape {conv.weight.shape})")

        num_out_groups = (out_ch + 3) // 4
        num_in_groups = (inp.channels + 3) // 4
        assert len(out_textures) == num_out_groups

        for g in range(num_out_groups):
            oc0 = g * 4
            oc1 = min(oc0 + 4, out_ch)
            oc_count = oc1 - oc0

            binds = list(inp.textures)
            if depthwise:
                binds = [inp.textures[g]]
            if residual is not None and residual.kind == "textures":
                binds.append(residual.textures[g])

            desc = f"{conv.name} ({oc_count}ch) {g + 1}/{num_out_groups}"
            save = None if (is_output and num_out_groups == 1) else out_textures[g]
            code = self.pass_header(desc, binds, save, inp.scale)
            code += "vec4 hook() {\n"

            if conv.bias is not None:
                code += f"    vec4 result = {format_vec4(conv.bias[oc0:oc1], self.precision)};\n"
            else:
                code += "    vec4 result = vec4(0.0);\n"

            if depthwise:
                tex = inp.textures[g]
                for ky in range(-pad_h, pad_h + 1):
                    for kx in range(-pad_w, pad_w + 1):
                        w = conv.weight[oc0:oc1, 0, ky + pad_h, kx + pad_w]
                        if np.allclose(w, 0, atol=1e-8):
                            continue
                        code += (f"    result += {format_vec4(w, self.precision)}"
                                 f" * {tex}_texOff(vec2({kx}, {ky}));\n")
            else:
                for ig in range(num_in_groups):
                    ic0 = ig * 4
                    ic1 = min(ic0 + 4, inp.channels)
                    ic_count = ic1 - ic0
                    tex = "HOOKED" if inp.kind == "hooked" else inp.textures[ig]

                    for ky in range(-pad_h, pad_h + 1):
                        for kx in range(-pad_w, pad_w + 1):
                            w = conv.weight[oc0:oc1, ic0:ic1, ky + pad_h, kx + pad_w]
                            if np.allclose(w, 0, atol=1e-8):
                                continue
                            if inp.channels == 1 and inp.kind == "hooked":
                                # LUMA input: single channel in .x
                                code += (f"    result += {format_vec4(w.flatten(), self.precision)}"
                                         f" * {tex}_texOff(vec2({kx}, {ky})).x;\n")
                            else:
                                # mat4 * vec4; GLSL mat4 is column-major so
                                # column i must hold weights for input channel i.
                                w_pad = np.zeros((4, 4), dtype=np.float64)
                                w_pad[:oc_count, :ic_count] = w
                                code += (f"    result += {format_mat4(w_pad.T, self.precision)}"
                                         f" * {tex}_texOff(vec2({kx}, {ky}));\n")

            code += self._activation_code(act, oc0, oc_count)

            if residual is not None:
                if residual.kind == "textures":
                    code += f"    result += {residual.textures[g]}_texOff(vec2(0, 0));\n"
                else:
                    # 'hooked'/'base': GPU bilinear sampling of HOOKED at the
                    # current position matches F.interpolate(align_corners=False)
                    # (and is an exact copy when this pass runs at 1x).
                    code += "    result += HOOKED_tex(HOOKED_pos);\n"

            if is_output:
                code += "    result = clamp(result, vec4(0.0), vec4(1.0));\n"
                if self.hook == "LUMA":
                    code += "    return vec4(result.x, 0.0, 0.0, 1.0);\n}\n\n"
                else:
                    code += "    return vec4(result.rgb, 1.0);\n}\n\n"
            else:
                code += "    return result;\n}\n\n"

            self.passes.append(code)

    # -- pixelshuffle pass ----------------------------------------------------

    def emit_pixelshuffle(self,
                          inp: Tensor,
                          out_textures: List[str],
                          r: int,
                          mode: str,
                          act: Optional[Tuple[str, Optional[np.ndarray]]],
                          residual_base: bool = False,
                          is_output: bool = False) -> None:
        """Emit passes performing DepthToSpace [C*r^2, H, W] -> [C, H*r, W*r].

        For each output pixel, idx = (y % r) * r + (x % r) selects which input
        channel supplies the value:
          CRD (PyTorch PixelShuffle): in_ch = out_ch * r^2 + idx
          DCR:                        in_ch = idx * C_out + out_ch
        """
        c_out = inp.channels // (r * r)
        num_out_groups = (c_out + 3) // 4
        out_scale = inp.scale * r
        assert len(out_textures) == num_out_groups
        if is_output and num_out_groups > 1:
            raise ConversionError("PixelShuffle output with more than 4 channels")

        def in_channel(oc: int, idx: int) -> int:
            if mode == "CRD":
                return oc * r * r + idx
            return idx * c_out + oc  # DCR

        for g in range(num_out_groups):
            oc0 = g * 4
            oc1 = min(oc0 + 4, c_out)
            oc_count = oc1 - oc0

            # All (texture, component) pairs this group needs.
            needed_texs: List[int] = []
            for idx in range(r * r):
                for oc in range(oc0, oc1):
                    t = in_channel(oc, idx) // 4
                    if t not in needed_texs:
                        needed_texs.append(t)
            needed_texs.sort()
            binds = [inp.textures[t] for t in needed_texs]
            svar = {t: f"s{si}" for si, t in enumerate(needed_texs)}

            desc = f"PixelShuffle {r}x ({oc_count}ch) {g + 1}/{num_out_groups}"
            save = None if is_output else out_textures[g]
            code = self.pass_header(desc, binds, save, out_scale)
            code += "vec4 hook() {\n"

            ref = inp.textures[needed_texs[0]]
            code += f"    vec2 f = fract({ref}_pos * {ref}_size);\n"
            code += f"    ivec2 sp = ivec2(f * vec2({r}.0));\n"
            code += f"    int idx = sp.y * {r} + sp.x;\n"
            for t in needed_texs:
                tex = inp.textures[t]
                code += (f"    vec4 {svar[t]} = {tex}_tex((vec2(0.5) - f)"
                         f" * {tex}_pt + {tex}_pos);\n")

            comps = "xyzw"

            def selector(oc: int, idx: int) -> str:
                ch = in_channel(oc, idx)
                return f"{svar[ch // 4]}.{comps[ch % 4]}"

            # If every output component reads component `idx` of a fixed
            # texture (true for CRD when C_out is a multiple of 4 and r == 2),
            # use dynamic component indexing instead of an if-chain.
            dynamic_ok = (r * r == 4)
            if dynamic_ok:
                for oc in range(oc0, oc1):
                    for idx in range(4):
                        ch = in_channel(oc, idx)
                        if ch // 4 != in_channel(oc, 0) // 4 or ch % 4 != idx:
                            dynamic_ok = False

            code += "    vec4 result;\n"
            if dynamic_ok:
                parts = []
                for oc in range(oc0, oc1):
                    parts.append(f"{svar[in_channel(oc, 0) // 4]}[idx]")
                while len(parts) < 4:
                    parts.append("0.0")
                code += f"    result = vec4({', '.join(parts)});\n"
            else:
                for idx in range(r * r):
                    parts = [selector(oc, idx) for oc in range(oc0, oc1)]
                    while len(parts) < 4:
                        parts.append("0.0")
                    kw = "if" if idx == 0 else "else if"
                    if idx == r * r - 1:
                        code += f"    else result = vec4({', '.join(parts)});\n"
                    else:
                        code += f"    {kw} (idx == {idx}) result = vec4({', '.join(parts)});\n"

            code += self._activation_code(act, oc0, oc_count)
            if residual_base:
                # GPU bilinear sample of HOOKED at the output pixel center
                # (matches F.interpolate(align_corners=False)).
                code += "    result += HOOKED_tex(HOOKED_pos);\n"
            if is_output:
                code += "    result = clamp(result, vec4(0.0), vec4(1.0));\n"
                if self.hook == "LUMA":
                    code += "    return vec4(result.x, 0.0, 0.0, 1.0);\n}\n\n"
                else:
                    code += "    return vec4(result.rgb, 1.0);\n}\n\n"
            else:
                code += "    return result;\n}\n\n"
            self.passes.append(code)

    # -- generic add pass (fallback) -------------------------------------------

    def emit_add(self, a: Tensor, b: Tensor, out_textures: List[str]) -> None:
        num_groups = (a.channels + 3) // 4
        for g in range(num_groups):
            binds = [a.textures[g], b.textures[g]]
            code = self.pass_header(f"Add {g + 1}/{num_groups}", binds, out_textures[g], a.scale)
            code += "vec4 hook() {\n"
            code += f"    vec4 result = {a.textures[g]}_texOff(vec2(0, 0));\n"
            code += f"    result += {b.textures[g]}_texOff(vec2(0, 0));\n"
            code += "    return result;\n}\n\n"
            self.passes.append(code)

    def build(self) -> str:
        return self.file_header() + "".join(self.passes)


# =============================================================================
# ONNX graph walking
# =============================================================================

class GraphConverter:
    def __init__(self, onnx_path: str, model_name: str, precision: int = 8,
                 expected_scale: Optional[int] = None, verbose: bool = True):
        if not ONNX_AVAILABLE:
            raise ImportError("onnx package required. Install with: pip install onnx")

        self.model = onnx.load(onnx_path)
        self.graph = self.model.graph
        self.model_name = model_name
        self.precision = precision
        self.expected_scale = expected_scale
        self.verbose = verbose
        self.source = Path(onnx_path).name

        self.weights: Dict[str, np.ndarray] = {}
        for init in self.graph.initializer:
            self.weights[init.name] = np.asarray(numpy_helper.to_array(init), dtype=np.float32)

        # Fold Constant nodes into the weights dict.
        self.const_nodes = set()
        for node in self.graph.node:
            if node.op_type == "Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        self.weights[node.output[0]] = np.asarray(
                            numpy_helper.to_array(attr.t), dtype=np.float32)
                self.const_nodes.add(node.output[0])

        self.consumers: Dict[str, List] = {}
        for node in self.graph.node:
            for name in node.input:
                self.consumers.setdefault(name, []).append(node)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _single_consumer(self, tensor_name: str):
        cons = self.consumers.get(tensor_name, [])
        return cons[0] if len(cons) == 1 else None

    def _input_channels(self) -> int:
        inp = self.graph.input[0]
        dims = inp.type.tensor_type.shape.dim
        if len(dims) != 4:
            raise ConversionError("Expected NCHW input")
        ch = dims[1].dim_value
        if ch not in (1, 3):
            raise ConversionError(f"Unsupported input channels: {ch} (expected 1 or 3)")
        return ch

    def _detect_scale(self) -> int:
        scale = 1
        for node in self.graph.node:
            if node.op_type == "DepthToSpace":
                for attr in node.attribute:
                    if attr.name == "blocksize":
                        scale *= attr.i
        return scale

    def _attr(self, node, name, default=None):
        for attr in node.attribute:
            if attr.name == name:
                return onnx.helper.get_attribute_value(attr)
        return default

    def _conv_info(self, node) -> ConvInfo:
        weight = self.weights.get(node.input[1])
        if weight is None:
            raise ConversionError(f"Conv '{node.name}': weight is not a constant initializer")
        bias = self.weights.get(node.input[2]) if len(node.input) > 2 else None

        groups = self._attr(node, "group", 1)
        strides = self._attr(node, "strides", [1, 1])
        dilations = self._attr(node, "dilations", [1, 1])
        pads = self._attr(node, "pads", [0, 0, 0, 0])
        kh, kw = weight.shape[2], weight.shape[3]

        if list(strides) != [1, 1] or list(dilations) != [1, 1]:
            raise ConversionError(f"Conv '{node.name}': only stride 1 / dilation 1 supported")
        if list(pads) != [kh // 2, kw // 2, kh // 2, kw // 2]:
            raise ConversionError(f"Conv '{node.name}': only 'same' padding supported (pads={pads})")

        return ConvInfo(weight=weight, bias=bias, groups=groups,
                        name=sanitize(node.name) or "conv")

    def _slopes(self, node) -> np.ndarray:
        s = self.weights.get(node.input[1])
        if s is None:
            raise ConversionError(f"PRelu '{node.name}': slope is not a constant initializer")
        return s.flatten()

    # -- main -------------------------------------------------------------------

    def convert(self) -> str:
        in_ch = self._input_channels()
        hook = "LUMA" if in_ch == 1 else "MAIN"
        scale = self._detect_scale()
        if self.expected_scale is not None and self.expected_scale != scale:
            raise ConversionError(
                f"--scale {self.expected_scale} does not match the model "
                f"(detected {scale}x from the ONNX graph)")

        self._log(f"  Input: {in_ch} channel(s) -> hooking {hook}")
        self._log(f"  Detected scale: {scale}x")

        builder = ShaderBuilder(self.model_name, hook, scale, self.precision, self.source)

        input_name = self.graph.input[0].name
        output_name = self.graph.output[0].name
        tensors: Dict[str, Tensor] = {
            input_name: Tensor(kind="hooked", channels=in_ch, scale=1)
        }
        consumed = set()  # node ids fused into another pass
        tex_counter = [0]
        used_names = set()

        def new_textures(base: str, channels: int) -> List[str]:
            n = (channels + 3) // 4
            tex_counter[0] += 1
            base = sanitize(base)[:40]
            if base in used_names:
                base = f"{base}_t{tex_counter[0]}"
            used_names.add(base)
            if n == 1:
                return [base]
            return [f"{base}_{i}" for i in range(n)]

        for node in self.graph.node:
            if id(node) in consumed or node.op_type == "Constant":
                continue

            if node.op_type in ("Identity", "Cast"):
                src = node.input[0]
                if src in self.weights:
                    self.weights[node.output[0]] = self.weights[src]
                elif src in tensors:
                    tensors[node.output[0]] = tensors[src]
                continue

            if node.op_type == "Resize":
                src = node.input[0]
                if src not in tensors or tensors[src].kind != "hooked":
                    raise ConversionError(
                        "Resize is only supported as the bilinear global residual on the input")
                mode = self._attr(node, "mode", b"nearest")
                mode = mode.decode() if isinstance(mode, bytes) else mode
                if mode not in ("linear", "cubic"):
                    raise ConversionError(f"Resize mode '{mode}' not supported (expected linear)")
                if mode == "cubic":
                    self._log("  Warning: cubic Resize approximated with GPU bilinear sampling")
                tensors[node.output[0]] = Tensor(kind="base", channels=in_ch, scale=scale)
                continue

            if node.op_type == "Conv":
                conv = self._conv_info(node)
                inp = tensors.get(node.input[0])
                if inp is None:
                    raise ConversionError(f"Conv '{node.name}': input tensor not available")
                if inp.kind == "base":
                    raise ConversionError(f"Conv '{node.name}': convolving the bilinear base is unsupported")

                out_ch = conv.weight.shape[0]
                cur_out = node.output[0]

                # Fuse a following PRelu/Relu (unless a DepthToSpace follows;
                # DIS applies the upsampler activation after the shuffle).
                act = None
                nxt = self._single_consumer(cur_out)
                if nxt is not None and nxt.op_type == "Relu":
                    act = ("relu", None)
                    consumed.add(id(nxt))
                    cur_out = nxt.output[0]
                elif nxt is not None and nxt.op_type == "PRelu":
                    act = ("prelu", self._slopes(nxt))
                    consumed.add(id(nxt))
                    cur_out = nxt.output[0]

                # Fuse a following Add with an already-computed tensor (residual).
                residual = None
                nxt = self._single_consumer(cur_out)
                if nxt is not None and nxt.op_type == "Add":
                    other = nxt.input[1] if nxt.input[0] == cur_out else nxt.input[0]
                    other_t = tensors.get(other)
                    if other_t is not None:
                        ok_tex = (other_t.kind == "textures"
                                  and other_t.channels == out_ch
                                  and other_t.scale == inp.scale)
                        ok_base = (other_t.kind in ("hooked", "base")
                                   and other_t.channels == out_ch
                                   and (other_t.kind == "base" or other_t.scale == inp.scale
                                        or inp.scale == 1))
                        if ok_tex or ok_base:
                            residual = other_t
                            consumed.add(id(nxt))
                            cur_out = nxt.output[0]

                is_output = (cur_out == output_name)
                out_tex = new_textures(conv.name, out_ch)
                builder.emit_conv(conv, inp, out_tex, act, residual, is_output)
                tensors[cur_out] = Tensor(kind="textures", channels=out_ch,
                                          scale=inp.scale, textures=out_tex)
                continue

            if node.op_type == "DepthToSpace":
                inp = tensors.get(node.input[0])
                if inp is None or inp.kind != "textures":
                    raise ConversionError(f"DepthToSpace '{node.name}': input tensor not available")
                r = self._attr(node, "blocksize")
                mode = self._attr(node, "mode", b"DCR")
                mode = mode.decode() if isinstance(mode, bytes) else mode
                if inp.channels % (r * r) != 0:
                    raise ConversionError(f"DepthToSpace '{node.name}': channels not divisible by r^2")
                cur_out = node.output[0]

                # Fuse the post-shuffle activation (PixelShuffleUpsampler.act).
                act = None
                nxt = self._single_consumer(cur_out)
                if nxt is not None and nxt.op_type == "Relu":
                    act = ("relu", None)
                    consumed.add(id(nxt))
                    cur_out = nxt.output[0]
                elif nxt is not None and nxt.op_type == "PRelu":
                    act = ("prelu", self._slopes(nxt))
                    consumed.add(id(nxt))
                    cur_out = nxt.output[0]

                c_out = inp.channels // (r * r)

                # Fuse a following Add with the bilinear global residual
                # (DIS2-style graphs: ... -> DepthToSpace -> Add(Resize(input))).
                residual_base = False
                nxt = self._single_consumer(cur_out)
                if nxt is not None and nxt.op_type == "Add":
                    other = nxt.input[1] if nxt.input[0] == cur_out else nxt.input[0]
                    other_t = tensors.get(other)
                    if (other_t is not None and other_t.kind in ("hooked", "base")
                            and other_t.channels == c_out
                            and (other_t.kind == "base" or inp.scale * r == 1)):
                        residual_base = True
                        consumed.add(id(nxt))
                        cur_out = nxt.output[0]

                is_output = (cur_out == output_name)
                out_tex = new_textures(f"ps{tex_counter[0]}", c_out)
                builder.emit_pixelshuffle(inp, out_tex, r, mode, act,
                                          residual_base=residual_base, is_output=is_output)
                tensors[cur_out] = Tensor(kind="textures", channels=c_out,
                                          scale=inp.scale * r, textures=out_tex)
                continue

            if node.op_type == "Add":
                a = tensors.get(node.input[0])
                b = tensors.get(node.input[1])
                if a is None or b is None:
                    raise ConversionError(f"Add '{node.name}': inputs not available")
                if a.kind != "textures" or b.kind != "textures" or a.channels != b.channels:
                    raise ConversionError(f"Add '{node.name}': unsupported operand combination")
                out_tex = new_textures(f"add{tex_counter[0]}", a.channels)
                builder.emit_add(a, b, out_tex)
                tensors[node.output[0]] = Tensor(kind="textures", channels=a.channels,
                                                 scale=a.scale, textures=out_tex)
                continue

            raise ConversionError(
                f"Unsupported op '{node.op_type}' ({node.name}). "
                f"Re-export the model with tools/export_onnx.py (opset 17).")

        out_t = tensors.get(output_name)
        if out_t is None:
            raise ConversionError("Graph output was never produced")
        if out_t.kind == "textures" and builder.passes and f"//!SAVE {out_t.textures[0]}" in builder.passes[-1]:
            # Output ended up in a saved texture (e.g. the graph ends with a
            # bare Add): emit a final copy pass with clamp + alpha fixup.
            code = builder.pass_header("Output", [out_t.textures[0]], None, out_t.scale)
            code += "vec4 hook() {\n"
            code += f"    vec4 result = {out_t.textures[0]}_texOff(vec2(0, 0));\n"
            code += "    result = clamp(result, vec4(0.0), vec4(1.0));\n"
            if hook == "LUMA":
                code += "    return vec4(result.x, 0.0, 0.0, 1.0);\n}\n\n"
            else:
                code += "    return vec4(result.rgb, 1.0);\n}\n\n"
            builder.passes.append(code)

        self._log(f"  Generated {len(builder.passes)} passes")
        return builder.build()


# =============================================================================
# Entry points
# =============================================================================

def export_onnx_to_glsl(onnx_path: str,
                        output_path: str,
                        model_name: str = "DIS",
                        precision: int = 8,
                        expected_scale: Optional[int] = None,
                        verbose: bool = True) -> str:
    if verbose:
        print(f"Loading ONNX model: {onnx_path}")
    converter = GraphConverter(onnx_path, model_name, precision, expected_scale, verbose)
    shader = converter.convert()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(shader, encoding="utf-8", newline="\n")
    if verbose:
        print(f"\n[OK] Shader saved to: {out}")
    return str(out)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DIS ONNX models to mpv GLSL shaders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/export_glsl.py --onnx model_x2.onnx --output shaders/model_x2.glsl --name MyModel

The scale (1x/2x/3x/4x) and hook (LUMA for 1-channel models, MAIN for RGB
models) are detected automatically from the ONNX graph.
""")
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output path for GLSL shader')
    parser.add_argument('--name', type=str, default='DIS', help='Model name for shader comments')
    parser.add_argument('--scale', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Optional: assert the model scale (auto-detected otherwise)')
    parser.add_argument('--precision', type=int, default=8,
                        help='Decimal precision for weights (default: 8)')
    args = parser.parse_args()

    print("=" * 60)
    print("DIS: ONNX to GLSL Shader Converter")
    print("=" * 60)

    try:
        out = export_onnx_to_glsl(
            onnx_path=args.onnx,
            output_path=args.output,
            model_name=args.name,
            precision=args.precision,
            expected_scale=args.scale,
        )
    except ConversionError as e:
        print(f"\nError: {e}")
        return 1

    print("\nTo use with mpv, add to mpv.conf:")
    print(f'  glsl-shaders="~~/shaders/{Path(out).name}"')
    print("\nNote: use vo=gpu-next, or vo=gpu with fbo-format=rgba16f")
    return 0


if __name__ == "__main__":
    exit(main())
