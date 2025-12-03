# DIS

DIS (Direct Image Supersampling) is a lightweight image super-resolution architecture optimized for speed and real-time inference. It has support for PyTorch, ONNX, and TensorRT.

This is the inference and ONNX conversion code. To train a model, you'll want to use [traiNNer-redux](https://github.com/the-database/traiNNer-redux).

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

## Benchmarks

**Configuration**: 2x upscale, 720p, FP16 with TensorRT, 2 streams

| Model | FPS | PSNR (BHI100) | SSIM (BHI100) | Notes |
|-------|-----|---------------|---------------|-------|
| DIS_Balanced | 100 | 27.44 | 0.898 | Slightly behind Compact, faster |
| DIS_Fast | 137 | 27.27 | 0.895 | On par with ArtCNN R8F48, 2x faster |
| Compact | 78 | 27.59 | 0.90 | Reference model |
| SPAN 48 | 81 | 27.53 | 0.90 | Reference model |

| ArtCNN R8F48 | 86 | 27.25 | 0.897 | Reference model |

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
