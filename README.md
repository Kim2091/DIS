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
| `DIS_XL` | ~343K | Max quality |
| `DIS_Balanced` | ~269K | Balance of speed and quality |
| `DIS_Fast` | ~195K | Fastest, recommended |

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

## Benchmarks:

2x, 720p FP16 tensorrt, 2 streams

__DIS_XL__ (Somewhere between Span 48 and Span 52)
51 FPS
BHI100 PSNR:
`'mean': np.float64(27.55756875991821)}}`

BHI100 SSIMC:
`'mean': np.float64(0.9000162765941646)}}`

__DIS_Balanced__ (Slightly behind full size Compact, but faster)
100 FPS
BHI100 PSNR:
`'mean': np.float64(27.442623882293702)}}`

BHI100 SSIMC:
`'mean': np.float64(0.8978795604670543)}}`

__DIS_Fast__ (Slightly behind ArtCNN R8F48, but almost twice as fast)
137 FPS
BHI100 PSNR:
`'mean': np.float64(27.27130084991455)}}`

BHI100 SSIMC:
`'mean': np.float64(0.8946407187861588)}}`

__SuperUltraCompact__
340 FPS

__SPAN__
81 FPS

__ArtCNNR8F48__
86 FPS

__Compact__
78 FPS