"""Minimal TensorRT inference wrapper for the YOLOP segmentation engine."""

import numpy as np
import tensorrt as trt
import torch


class TRTSeg:
    """Runs the YOLOP seg-only engine. Keeps all buffers on GPU via torch tensors."""

    def __init__(self, engine_path, imgsz=384, device="cuda"):
        self.imgsz = imgsz
        self.device = torch.device(device)

        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        with open(str(engine_path), "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        # Map tensor names to (is_input, shape, dtype)
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names
                       if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        # Preallocate output tensors
        self.out_bufs = {}
        for n in self.outputs:
            shape = tuple(self.ctx.get_tensor_shape(n))
            dtype = self._torch_dtype(self.engine.get_tensor_dtype(n))
            self.out_bufs[n] = torch.empty(shape, dtype=dtype, device=self.device)

        self.stream = torch.cuda.Stream(device=self.device)

    @staticmethod
    def _torch_dtype(trt_dtype):
        return {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT64: torch.int64,
            trt.DataType.BOOL: torch.bool,
            trt.DataType.UINT8: torch.uint8,
        }[trt_dtype]

    def infer(self, chw_tensor):
        """chw_tensor: float32 CUDA tensor shaped (1,3,imgsz,imgsz), values 0-1.
        Returns (da_mask, ll_mask) as uint8 CUDA tensors."""
        self.ctx.set_tensor_address(self.inputs[0], chw_tensor.data_ptr())
        for n, buf in self.out_bufs.items():
            self.ctx.set_tensor_address(n, buf.data_ptr())

        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        da = self.out_bufs["da_seg"]
        ll = self.out_bufs["ll_seg"]
        return da, ll
