# DIS

DIS (Direct Image Supersampling) is a lightweight image super-resolution architecture optimized for speed and real-time inference. It has support for PyTorch, ONNX, and TensorRT.

This is the inference and ONNX conversion code. To train a model, you'll want to use [traiNNer-redux](https://github.com/Kim2091/traiNNer-redux-1).

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Kim2091/DIS
    ```

2.  **Install PyTorch with CUDA**:
    Follow the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

3.  **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Model Variants

| Variant | Parameters (2x) | Description |
|---------|-----------|-------------|
| `DIS_Balanced` | ~269K | Balance of speed and quality |
| `DIS_Fast` | ~195K | Fastest, recommended |

## DIS2 (experimental)

DIS2 is a reparameterized successor to DIS: it trains with multi-branch
ECB blocks (RepVGG/ECBSR-style) that collapse into a **plain stack of 3x3
convs** for inference, and keeps all convolutions at low resolution
(DIS ran its tail conv at output resolution). Same op set as DIS
(Conv/PReLU/Add/PixelShuffle/Resize), so TensorRT, ONNX, and the GLSL
converter work unchanged on the fused model.

| Variant (2x) | MACs / LR pixel | Deploy params | vs DIS |
|---------|-----------------|--------|--------|
| `DIS2_Fast` (32f, 6 convs) | ~60K | ~60K | 3.3x fewer MACs than DIS_Fast |
| `DIS2_Balanced` (48f, 8 convs) | ~172K | ~173K | 1.6x fewer MACs than DIS_Balanced |

Measured as mpv GLSL shaders (960x540 -> 1080p): DIS2_Fast renders ~4-5x
faster than DIS_Fast. Quality parity with DIS relies on the rep branches
during training and needs to be confirmed with a full training run.

Train with `type: dis2_fast` / `type: dis2_balanced` in traiNNer-redux
(drop `models/dis2_arch.py` into its arch folder), then fuse for release:

```bash
python tools/fuse_dis2.py trained.safetensors fused.onnx --model fast --scale 2 --fp16
```

The training-form checkpoint is ~4x larger than the fused one; always ship
the fused version.

## Benchmarks

**Configuration**: 2x upscale, 720p, FP16 with TensorRT, 2 streams

| Model | FPS | PSNR (BHI100) | SSIM (BHI100) | Notes |
|-------|-----|---------------|---------------|-------|
| DIS_Balanced | 100 | 27.44 | 0.898 | Slightly behind Compact, faster |
| DIS_Fast | 137 | 27.27 | 0.895 | On par with ArtCNN R8F48, 2x faster |
| ArtCNN R8F48 | 86 | 27.25 | 0.897 | Reference model |
| Compact | 78 | 27.59 | 0.90 | Reference model |


## Usage

### Command-Line Usage

**Image upscaling (PyTorch)**:
```bash
python inference.py --input lr.png --output sr.png --scale 4 --fp16
```

**ONNX inference**:
```bash
python inference.py --input lr.png --output sr.png --model model.onnx --backend onnx
```

**Benchmark**:
```bash
python inference.py --benchmark --scale 4 --fp16
```

## Tools

Utility scripts are located in the `tools/` directory.

**Convert PyTorch model to ONNX**:
```bash
python tools/export_onnx.py --model pretrained_models/model.pth --output model.onnx
```
-   `--dynamic`: Create a model that supports various input sizes.
-   `--fp16`: Convert the model to FP16 for a speed boost.

**Convert ONNX model to an mpv GLSL shader**:
```bash
python tools/export_glsl.py --onnx model.onnx --output model.glsl --name MyModel
```
Supports all scales (1x-4x); the scale and hook point (RGB → `MAIN`, single-channel → `LUMA`) are detected automatically from the ONNX graph. Requires float FBOs in mpv: use `vo=gpu-next`, or `vo=gpu` with `fbo-format=rgba16f`. See [tools/README.md](tools/README.md) for details.

## TensorRT

The easiest way to use this model with TensorRT is through [Vapourkit](https://github.com/Kim2091/vapourkit) or [VideoJaNai](https://github.com/the-database/VideoJaNai).

Alternatively you can convert to TensorRT manually:

```bash
# Dynamic shapes
trtexec --onnx=model_fp16.onnx \
    --minShapes=input:1x3x64x64 \
    --optShapes=input:1x3x256x256 \
    --maxShapes=input:1x3x1024x1024 \
    --saveEngine=model_dynamic.engine \
    --fp16
```


