"""
Fuse a trained DIS2 checkpoint (training form, with rep branches) into the
plain-conv deploy form, and optionally export it to ONNX.

Usage:
    # Fuse a training checkpoint to a deploy checkpoint:
    python tools/fuse_dis2.py model.safetensors fused.safetensors --model fast --scale 2

    # Fuse and export ONNX in one go:
    python tools/fuse_dis2.py model.safetensors fused.onnx --model fast --scale 2 --fp16

The deploy graph uses only Conv / PReLU / Add / DepthToSpace / Resize, so the
resulting ONNX works with TensorRT and tools/export_glsl.py unchanged.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.dis2_arch import dis2_fast, dis2_balanced  # noqa: E402

try:
    from safetensors.torch import load_file as load_safetensors
    from safetensors.torch import save_file as save_safetensors
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

VARIANTS = {"fast": dis2_fast, "balanced": dis2_balanced}


def load_state_dict(path: Path):
    if path.suffix == ".safetensors":
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors required: pip install safetensors")
        return load_safetensors(str(path))
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    for key in ("state_dict", "params_ema", "params"):
        if key in sd:
            return sd[key]
    return sd


def export_onnx(model, out_path: Path, fp16: bool, opset: int = 17):
    import onnx
    model = model.half() if fp16 else model.float()
    dummy = torch.randn(1, model.in_channels, 64, 64,
                        dtype=torch.float16 if fp16 else torch.float32)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height_out", 3: "width_out"}},
        opset_version=opset, do_constant_folding=True, dynamo=False)
    try:
        import onnxsim
        m, ok = onnxsim.simplify(onnx.load(str(out_path)),
                                 test_input_shapes={"input": [1, model.in_channels, 64, 64]})
        if ok:
            onnx.save(m, str(out_path))
            print("  ONNX simplified")
    except ImportError:
        pass
    onnx.checker.check_model(onnx.load(str(out_path)))
    print(f"  ONNX validated: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fuse DIS2 rep branches for deployment")
    parser.add_argument("input", help="Training checkpoint (.safetensors or .pth)")
    parser.add_argument("output", help="Output path (.safetensors, .pth, or .onnx)")
    parser.add_argument("--model", default="fast", choices=list(VARIANTS.keys()))
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--fp16", action="store_true", help="FP16 ONNX export")
    args = parser.parse_args()

    model = VARIANTS[args.model](scale=args.scale)
    sd = load_state_dict(Path(args.input))
    model.load_state_dict(sd)
    model.eval()

    params_before = sum(p.numel() for p in model.parameters())
    model.switch_to_deploy()
    params_after = sum(p.numel() for p in model.parameters())
    print(f"Fused {args.model} x{args.scale}: {params_before:,} -> {params_after:,} params")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if out.suffix == ".onnx":
            export_onnx(model, out, args.fp16)
        elif out.suffix == ".safetensors":
            if not SAFETENSORS_AVAILABLE:
                raise ImportError("safetensors required: pip install safetensors")
            save_safetensors(model.state_dict(), str(out))
            print(f"Saved deploy checkpoint: {out}")
        else:
            torch.save(model.state_dict(), str(out))
            print(f"Saved deploy checkpoint: {out}")
    return 0


if __name__ == "__main__":
    exit(main())
