"""
Inference backend factory.
"""
from core.backends.base import InferenceBackend
from core.backends.mock_backend import MockPoseBackend


def create_backend(backend_type: str = "mock", model_path: str = "models/yolo_pose.engine", **kwargs):
    backend_type = backend_type.lower()
    if backend_type == "mock":
        return MockPoseBackend(model_path=model_path, **kwargs)
    elif backend_type == "onnx":
        try:
            from core.backends.onnx_backend import ONNXRuntimeBackend
            return ONNXRuntimeBackend(model_path=model_path, **kwargs)
        except ImportError:
            raise ImportError("ONNX Runtime not installed. pip install onnxruntime-gpu")
    elif backend_type == "torch":
        try:
            from core.backends.torch_backend import PyTorchBackend
            return PyTorchBackend(model_path=model_path, **kwargs)
        except ImportError:
            raise ImportError("PyTorch not installed. pip install torch torchvision")
    elif backend_type == "trt":
        try:
            from core.backends.tensorrt_backend import TensorRTBackend
            return TensorRTBackend(model_path=model_path, **kwargs)
        except ImportError:
            raise ImportError("TensorRT not installed. Requires NVIDIA GPU + pycuda")
    else:
        raise ValueError(f"Unknown backend: {backend_type}. Choose: mock, onnx, torch, trt")