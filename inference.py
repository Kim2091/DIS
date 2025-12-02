"""
Inference utilities for DIS (Direct Image Supersampling)

Supports:
- PyTorch inference (FP32/FP16)
- ONNX Runtime inference
- TensorRT inference
- Tiled inference for large images
"""

import torch
import numpy as np
from typing import TYPE_CHECKING, Union, Optional, Tuple
import time

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ort = None
    ORT_AVAILABLE = False

try:
    from safetensors.torch import load_file as load_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    load_safetensors = None
    SAFETENSORS_AVAILABLE = False

if TYPE_CHECKING:
    from PIL import Image as PILImage

from models.ultralight_sr import DIS

# Type alias for image inputs
ImageInput = Union[np.ndarray, "PILImage.Image", torch.Tensor]

# Model size presets (matching dis_arch.py registry entries)
MODEL_PRESETS = {
    "nano": {"num_features": 10, "num_blocks": 6},
    "fast": {"num_features": 32, "num_blocks": 8},
    "balanced": {"num_features": 32, "num_blocks": 12},
    "xl": {"num_features": 32, "num_blocks": 16},
}


class DISInference:
    """
    Unified inference class for DIS models.
    
    Supports PyTorch, ONNX, and TensorRT backends.
    """
    
    SUPPORTED_BACKENDS = ("pytorch", "onnx", "tensorrt")
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "pytorch",
        device: str = "cuda",
        fp16: bool = True,
        scale: int = 4,
        model_type: str = "standard",
        num_features: int = 32,
        num_blocks: int = 4
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model weights (.pth) or ONNX file (.onnx)
            backend: One of 'pytorch', 'onnx', 'tensorrt'
            device: 'cuda' or 'cpu'
            fp16: Use FP16 precision (only applies to CUDA)
            scale: Upscaling factor
            model_type: Model variant - 'standard', 'depthwise', or preset name 
                        ('nano', 'fast', 'balanced', 'xl')
            num_features: Number of feature channels (ignored if using preset)
            num_blocks: Number of residual blocks (ignored if using preset)
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Supported: {self.SUPPORTED_BACKENDS}")
        
        self.backend = backend
        self.device = device
        self.fp16 = fp16 and (device == "cuda")  # FP16 only makes sense on CUDA
        self.scale = scale
        self.model = None
        self.session = None
        self.engine = None
        
        init_methods = {
            "pytorch": lambda: self._init_pytorch(model_path, model_type, num_features, num_blocks),
            "onnx": lambda: self._init_onnx(model_path),
            "tensorrt": lambda: self._init_tensorrt(model_path),
        }
        init_methods[backend]()
    
    def _init_pytorch(self, model_path, model_type, num_features, num_blocks):
        """Initialize PyTorch model"""
        use_depthwise = False
        
        # Check if model_type is a preset name
        if model_type in MODEL_PRESETS:
            preset = MODEL_PRESETS[model_type]
            num_features = preset["num_features"]
            num_blocks = preset["num_blocks"]
        elif model_type == "depthwise":
            use_depthwise = True
        elif model_type != "standard":
            raise ValueError(f"Unknown model_type: {model_type}. "
                           f"Use 'standard', 'depthwise', or one of: {list(MODEL_PRESETS.keys())}")
        
        self.model = DIS(
            scale=self.scale,
            num_features=num_features,
            num_blocks=num_blocks,
            use_depthwise=use_depthwise
        )
        
        if model_path:
            state_dict = self._load_weights(model_path)
            self.model.load_state_dict(state_dict)
        
        self.model = self.model.to(self.device)
        if self.fp16 and self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()
    
    def _load_weights(self, model_path: str) -> dict:
        """Load model weights from .pth, .pt, or .safetensors file."""
        path_lower = model_path.lower()
        
        if path_lower.endswith('.safetensors'):
            if not SAFETENSORS_AVAILABLE:
                raise RuntimeError(
                    "safetensors not installed. Install with: pip install safetensors"
                )
            return load_safetensors(model_path)
        else:
            # .pth or .pt file
            return torch.load(model_path, map_location='cpu', weights_only=True)
    
    def _init_onnx(self, model_path):
        """Initialize ONNX Runtime session"""
        if not ORT_AVAILABLE:
            raise RuntimeError("ONNX Runtime not installed")
        
        if self.device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def _init_tensorrt(self, model_path):
        """Initialize TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise RuntimeError("TensorRT or PyCUDA not installed")
        
        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Setup I/O bindings
        self.cuda = cuda
    
    def _preprocess(self, image: ImageInput) -> torch.Tensor:
        """
        Preprocess input image to tensor.
        
        Args:
            image: Input image (HWC numpy, PIL Image, or CHW tensor)
            
        Returns:
            Preprocessed tensor (BCHW, normalized to [0, 1])
        """
        if isinstance(image, torch.Tensor):
            x = image
        elif PIL_AVAILABLE and Image is not None and isinstance(image, Image.Image):
            x = torch.from_numpy(np.array(image)).float()
            x = x.permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(image, np.ndarray):
            x = torch.from_numpy(image).float()
            if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
                x = x.permute(2, 0, 1)  # HWC -> CHW
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Add batch dim if needed
        if x.ndim == 3:
            x = x.unsqueeze(0)
        
        # Normalize to [0, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0
        
        # Move to device and dtype
        x = x.to(self.device)
        if self.fp16 and self.device == "cuda":
            x = x.half()
        
        return x
    
    def _postprocess(self, tensor: torch.Tensor, to_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        Postprocess output tensor to image.
        
        Args:
            tensor: Output tensor (BCHW)
            to_numpy: Convert to numpy array
            
        Returns:
            Image as numpy (HWC, uint8) or tensor
        """
        x = tensor.clamp(0, 1)
        
        if to_numpy:
            x = x.squeeze(0).permute(1, 2, 0)  # CHW -> HWC
            x = (x.float().cpu().numpy() * 255).astype(np.uint8)
        
        return x
    
    @torch.no_grad()
    def __call__(
        self, 
        image: ImageInput,
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Run super-resolution on an image.
        
        Args:
            image: Input image
            return_tensor: Return tensor instead of numpy array
            
        Returns:
            Super-resolved image
        """
        x = self._preprocess(image)
        
        if self.backend == "pytorch":
            y = self.model(x)
        elif self.backend == "onnx":
            x_np = x.cpu().numpy()
            y_np = self.session.run([self.output_name], {self.input_name: x_np})[0]
            y = torch.from_numpy(y_np).to(self.device)
        elif self.backend == "tensorrt":
            y = self._infer_tensorrt(x)
        
        return y if return_tensor else self._postprocess(y)
    
    def _infer_tensorrt(self, x: torch.Tensor) -> torch.Tensor:
        """TensorRT inference"""
        # Allocate output buffer
        batch, c, h, w = x.shape
        out_shape = (batch, c, h * self.scale, w * self.scale)
        output = torch.empty(out_shape, dtype=x.dtype, device=self.device)
        
        # Set input shape for dynamic engines
        self.context.set_input_shape("input", x.shape)
        
        # Run inference
        bindings = [x.data_ptr(), output.data_ptr()]
        self.context.execute_v2(bindings)
        
        return output
    
    def benchmark(
        self, 
        input_size: Tuple[int, int] = (64, 64),
        batch_size: int = 1,
        num_warmup: int = 10,
        num_runs: int = 100
    ) -> dict:
        """
        Benchmark inference speed.
        
        Args:
            input_size: (height, width) of test input
            batch_size: Batch size
            num_warmup: Number of warmup runs
            num_runs: Number of timed runs
            
        Returns:
            Dictionary with timing statistics
        """
        h, w = input_size
        dtype = torch.float16 if self.fp16 else torch.float32
        x = torch.randn(batch_size, 3, h, w, device=self.device, dtype=dtype)
        
        # Warmup
        for _ in range(num_warmup):
            _ = self(x, return_tensor=True)
        
        if self.device == "cuda":
            torch.cuda.synchronize()
        
        # Timed runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self(x, return_tensor=True)
            if self.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            "input_size": f"{batch_size}x3x{h}x{w}",
            "output_size": f"{batch_size}x3x{h*self.scale}x{w*self.scale}",
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "fps": float(1000 / np.mean(times) * batch_size)
        }


def tiled_inference(
    model: DISInference,
    image: np.ndarray,
    tile_size: int = 256,
    overlap: int = 32
) -> np.ndarray:
    """
    Process large images using tiled inference with blending.
    
    Useful for images that don't fit in GPU memory.
    
    Args:
        model: DIS inference model
        image: Input image (HWC, uint8)
        tile_size: Size of each tile (must be > overlap)
        overlap: Overlap between tiles for seamless blending
        
    Returns:
        Super-resolved image (HWC, uint8)
    """
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")
    
    h, w, c = image.shape
    scale = model.scale
    
    # Handle images smaller than tile_size
    if h <= tile_size and w <= tile_size:
        return model(image)
    
    # Output buffers
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
    
    # Create smooth blending weights (raised cosine)
    def create_blend_weight(size: int) -> np.ndarray:
        if size <= 1:
            return np.ones(size)
        t = np.linspace(0, 1, size)
        # Smooth blend: stronger in center, weaker at edges
        return np.clip(np.minimum(t, 1 - t) * 2, 0.01, 1.0)
    
    stride = tile_size - overlap
    scaled_tile = tile_size * scale
    
    # Precompute blend weights for this tile size
    wy = create_blend_weight(scaled_tile)
    wx = create_blend_weight(scaled_tile)
    base_blend = (wy[:, None] * wx[None, :])[:, :, None]
    
    # Process tiles
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Clamp tile to image bounds
            y1 = min(y, max(0, h - tile_size))
            x1 = min(x, max(0, w - tile_size))
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)
            
            tile = image[y1:y2, x1:x2]
            
            # Handle edge tiles that may be smaller
            actual_h, actual_w = tile.shape[:2]
            if actual_h < tile_size or actual_w < tile_size:
                # Pad small tiles
                padded = np.zeros((tile_size, tile_size, c), dtype=tile.dtype)
                padded[:actual_h, :actual_w] = tile
                sr_tile = model(padded).astype(np.float32)
                sr_tile = sr_tile[:actual_h * scale, :actual_w * scale]
                tile_weight = base_blend[:actual_h * scale, :actual_w * scale]
            else:
                sr_tile = model(tile).astype(np.float32)
                tile_weight = base_blend
            
            # Accumulate weighted output
            oy1, oy2 = y1 * scale, y1 * scale + sr_tile.shape[0]
            ox1, ox2 = x1 * scale, x1 * scale + sr_tile.shape[1]
            
            output[oy1:oy2, ox1:ox2] += sr_tile * tile_weight
            weight[oy1:oy2, ox1:ox2] += tile_weight
    
    # Normalize and convert back to uint8
    output = output / np.maximum(weight, 1e-8)
    return np.clip(output, 0, 255).astype(np.uint8)


def main():
    """CLI for DIS inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='DIS: Direct Image Supersampling Inference')
    parser.add_argument('--input', '-i', type=str, help='Input image path')
    parser.add_argument('--output', '-o', type=str, help='Output image path')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model weights path (.pth or .onnx)')
    parser.add_argument('--backend', '-b', type=str, default='pytorch', 
                        choices=['pytorch', 'onnx'], help='Inference backend')
    parser.add_argument('--scale', '-s', type=int, default=4, choices=[2, 3, 4], 
                        help='Upscaling factor')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'depthwise', 'nano', 'fast', 'balanced', 'xl'],
                        help='Model variant (presets: nano, fast, balanced, xl)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 precision')
    parser.add_argument('--tile-size', '-t', type=int, default=0, 
                        help='Tile size for large images (0 = disabled)')
    parser.add_argument('--overlap', type=int, default=32, 
                        help='Tile overlap for blending')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark only')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.benchmark and (not args.input or not args.output):
        parser.error("--input and --output are required unless --benchmark is specified")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine backend from model extension if not specified
    backend = args.backend
    if args.model and args.model.endswith('.onnx'):
        backend = "onnx"
    
    # Create inference engine
    engine = DISInference(
        model_path=args.model,
        backend=backend,
        device=device,
        fp16=args.fp16,
        scale=args.scale,
        model_type=args.model_type
    )
    
    # Run benchmark if requested
    if args.benchmark:
        print(f"\nBenchmark ({backend}, {'FP16' if args.fp16 else 'FP32'}, {device}):")
        print("-" * 50)
        for size in [(64, 64), (128, 128), (256, 256), (512, 512)]:
            results = engine.benchmark(input_size=size)
            print(f"  {results['input_size']:>15} -> {results['output_size']:>15}  "
                  f"{results['mean_ms']:6.2f}ms  ({results['fps']:6.1f} FPS)")
        return
    
    # Load image
    if not PIL_AVAILABLE:
        print("Error: PIL/Pillow is required for image loading")
        return 1
    
    print(f"Loading: {args.input}")
    image = Image.open(args.input).convert('RGB')
    image_np = np.array(image)
    
    print(f"Processing ({image.size[0]}x{image.size[1]}) with {backend}...")
    start = time.perf_counter()
    
    if args.tile_size > 0:
        output = tiled_inference(
            engine, image_np, 
            tile_size=args.tile_size, 
            overlap=args.overlap
        )
    else:
        output = engine(image_np)
    
    elapsed = (time.perf_counter() - start) * 1000
    
    # Save output
    output_image = Image.fromarray(output)
    output_image.save(args.output)
    
    print(f"Saved: {args.output}")
    print(f"Size: {image.size[0]}x{image.size[1]} -> {output_image.size[0]}x{output_image.size[1]}")
    print(f"Time: {elapsed:.2f}ms")


if __name__ == "__main__":
    main()
