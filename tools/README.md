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
- `--scale`: Upscaling factor (1, 2, or 4)
- `--fp16`: Export in FP16 precision
- `--validate`: Validate exported model against PyTorch
- `--no-simplify`: Skip ONNX graph simplification

## export_glsl_1x_only.py
**__NOTE:__** This is experimental. It is currently broken for any model scale above 1x.

Converts ONNX models to mpv-compatible RGB GLSL shaders.

```bash
python .\export_glsl.py --onnx .\1x-SwatKats_DIS_Balanced_fp16.onnx --output .\exports\1x-SwatKats_DIS_Balanced_fp16.glsl --scale 1 --rgb
```

Arguments:
- `--onnx`: Path to ONNX model file
- `--output`: Output path for GLSL shader
- `--scale`: Must be set to `1`
- `--rgb`: Generate RGB shader (hooks MAIN instead of LUMA)
- `--name`: Model name for shader comments
- `--simplified`: Generate simplified single-pass shader

