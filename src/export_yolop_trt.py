"""Export YOLOP to TensorRT. YOLOP is 58% of pipeline cost — this is the big win.

YOLOP has 3 outputs (detection, drivable-area seg, lane-line seg). We only use the
two segmentation heads, but we export all three since the graph shares a backbone.
"""

import sys
from pathlib import Path

import torch
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR

IMGSZ = 384          # Smaller than the old 480 — seg masks upscale fine
ONNX_PATH = MODELS_DIR / f"yolop_{IMGSZ}.onnx"
ENGINE_PATH = MODELS_DIR / f"yolop_{IMGSZ}.engine"


class SegOnly(torch.nn.Module):
    """Wrapper exposing just the two segmentation heads (skips detection NMS)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, da_seg, ll_seg = self.model(x)
        # argmax here so the GPU does it, not numpy on CPU later
        return da_seg.argmax(1).to(torch.uint8), ll_seg.argmax(1).to(torch.uint8)


def export_onnx():
    if ONNX_PATH.exists():
        print(f"  ONNX exists: {ONNX_PATH.name}")
        return
    print("  Loading YOLOP from torch.hub...")
    model = torch.hub.load('hustvl/YOLOP', 'yolop', pretrained=True, trust_repo=True)
    model.eval()
    wrapped = SegOnly(model).eval()

    dummy = torch.randn(1, 3, IMGSZ, IMGSZ)
    print(f"  Tracing to ONNX at {IMGSZ}x{IMGSZ}...")
    torch.onnx.export(
        wrapped, dummy, str(ONNX_PATH),
        input_names=["images"],
        output_names=["da_seg", "ll_seg"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  ✓ ONNX: {ONNX_PATH.name} ({ONNX_PATH.stat().st_size/1e6:.1f} MB)")


def build_engine():
    if ENGINE_PATH.exists():
        print(f"  Engine exists: {ENGINE_PATH.name}")
        return True
    print("  Building TensorRT engine (takes 1-3 min)...")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    with open(str(ONNX_PATH), 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    ERROR: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 3 << 30)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("    ERROR: build failed")
        return False
    ENGINE_PATH.write_bytes(engine_bytes)
    print(f"  ✓ Engine: {ENGINE_PATH.name} ({ENGINE_PATH.stat().st_size/1e6:.1f} MB)")
    return True


if __name__ == "__main__":
    print("=" * 55)
    print(f"YOLOP -> TensorRT  ({IMGSZ}x{IMGSZ}, seg heads only)")
    print("=" * 55)
    export_onnx()
    build_engine()
    print("\nDone. Pipeline will auto-detect the engine.")
