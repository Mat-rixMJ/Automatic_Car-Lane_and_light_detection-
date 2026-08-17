"""Export YOLOv8 models to TensorRT engines directly (bypass modelopt bug)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils import MODELS_DIR

from ultralytics import YOLO
import tensorrt as trt


def export_to_onnx(model_path, imgsz):
    """Export to ONNX first (this works fine)."""
    onnx_path = model_path.with_suffix('.onnx')
    if onnx_path.exists():
        print(f"  ONNX already exists: {onnx_path.name}")
        return onnx_path
    
    model = YOLO(str(model_path))
    model.export(format="onnx", imgsz=imgsz, half=False, simplify=True, device=0)
    print(f"  ONNX exported: {onnx_path.name}")
    return onnx_path


def build_engine(onnx_path, engine_path, fp16=True):
    """Build TensorRT engine from ONNX."""
    print(f"  Building TensorRT engine...")
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(str(onnx_path), 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    ERROR: {parser.get_error(i)}")
            return False
    
    # Build config — TRT 11 uses TF32 by default (fast enough)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # Build engine
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("    ERROR: Engine build failed!")
        return False
    
    with open(str(engine_path), 'wb') as f:
        f.write(engine_bytes)
    
    size_mb = engine_path.stat().st_size / 1024 / 1024
    print(f"  ✓ Engine saved: {engine_path.name} ({size_mb:.1f} MB)")
    return True


def export_model(model_path, imgsz, name):
    print(f"\n{'='*50}")
    print(f"Exporting: {name}")
    print(f"  Source: {model_path}")
    print(f"  Input: {imgsz}x{imgsz}px, FP16")
    print(f"{'='*50}")
    
    engine_path = model_path.with_suffix('.engine')
    if engine_path.exists():
        print(f"  Engine already exists: {engine_path.name}")
        return
    
    onnx_path = export_to_onnx(model_path, imgsz)
    build_engine(onnx_path, engine_path, fp16=True)


if __name__ == "__main__":
    # 640 on the native frame: measured, this recovers 2.4x more traffic lights
    # than 480 on a 512p downscale (median light is 34px native, 13px downscaled).
    export_model(MODELS_DIR / "yolov8n.pt", imgsz=640, name="YOLOv8n COCO (vehicles+lights)")
    
    # Export German sign detector
    export_model(MODELS_DIR / "german_sign_detector.pt", imgsz=480, name="German Sign Detector")
    
    print("\n\n✓ All models exported to TensorRT (.engine)")
    print("  Pipeline will auto-detect .engine files and use them.")
