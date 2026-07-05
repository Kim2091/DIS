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

## fuse_dis2.py

Fuses a trained DIS2 checkpoint (training form, with reparameterization
branches) into the plain-conv deploy form, and optionally exports ONNX.

```bash
# Fuse to a deploy checkpoint
python tools/fuse_dis2.py trained.safetensors fused.safetensors --model fast --scale 2

# Fuse and export ONNX (TensorRT / export_glsl.py ready)
python tools/fuse_dis2.py trained.safetensors fused.onnx --model fast --scale 2 --fp16
```

The fused model computes exactly the same function as the training form
(verified to float precision) with ~4x fewer parameters.

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

**mpv requirements:** the intermediate feature maps contain negative values,
so the shader needs float FBOs. Use `vo=gpu-next` (recommended), or `vo=gpu`
with `fbo-format=rgba16f`. With the default unorm FBOs on `vo=gpu`, features
get clamped and quality silently degrades.

Known limitation: convolution borders use clamp-to-edge sampling instead of
the zero padding ONNX uses, so the outermost few pixels differ slightly from
ONNX inference. This is inherent to mpv hook shaders (ArtCNN/FSRCNNX behave
the same way).

