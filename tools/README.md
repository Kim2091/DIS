# Tools

Utility scripts for DIS model conversion and export.

## export_onnx.py

Converts PyTorch models to ONNX format.

```bash
python tools/export_onnx.py pretrained_models/model.safetensors model.onnx --model balanced --scale 4
```

Arguments:
- `input`: Path to input weights (.safetensors or .pth)
- `output`: Path to output ONNX file
- `--model`: Model variant (e.g., `fast`, `balanced`)
- `--scale`: Upscaling factor (1, 2, 3, or 4)
- `--fp16`: Export in FP16 precision
- `--validate`: Validate exported model against PyTorch
- `--no-simplify`: Skip ONNX graph simplification

## export_glsl.py

Converts ONNX models to mpv-compatible GLSL shaders. Works for all DIS scales
(1x, 2x, 3x, 4x), both the regular and depthwise (`use_depthwise`) variants,
and FP16 or FP32 exports. The converter walks the ONNX graph directly, so the
scale and hook point are detected automatically:

- 3-channel (RGB) models hook `MAIN`
- 1-channel models hook `LUMA`

```bash
python tools/export_glsl.py --onnx 1x-SwatKats_DIS_Balanced_fp16.onnx --output exports/1x-SwatKats_DIS_Balanced_fp16.glsl --name SwatKats
```

Arguments:
- `--onnx`: Path to ONNX model file (export with `export_onnx.py`, opset 17)
- `--output`: Output path for GLSL shader
- `--name`: Model name for shader comments
- `--scale`: Optional; asserts the expected scale against what the graph contains
- `--precision`: Decimal precision for embedded weights (default: 8)
- `--compute`: Experimental; emits compute-shader passes for the dense 3x3 convs.
  Each 16x8 workgroup cooperatively loads the input tile (with halo) into shared
  memory once instead of every pixel fetching its 3x3 neighborhood from every
  input texture. Output is pixel-identical to the fragment version (verified
  within 1/255 in mpv). Measured on an RTX 5080 laptop with vo=gpu-next +
  Vulkan: ~5% faster than the fragment shader (the workload is ALU-bound on
  modern GPUs, so the fetch savings are mostly absorbed by the texture cache;
  bandwidth-limited GPUs may see more). On Windows, use `--gpu-api=vulkan`
  with this: the d3d11 backend's HLSL translation takes ~1s per pass on the
  first (uncached) load.

**mpv requirements:** the intermediate feature maps contain negative values,
so the shader needs float FBOs. Use `vo=gpu-next` (recommended), or `vo=gpu`
with `fbo-format=rgba16f`. With the default unorm FBOs on `vo=gpu`, features
get clamped and quality silently degrades.

Known limitation: convolution borders use clamp-to-edge sampling instead of
the zero padding ONNX uses, so the outermost few pixels differ slightly from
ONNX inference. This is inherent to mpv hook shaders (ArtCNN/FSRCNNX behave
the same way).

