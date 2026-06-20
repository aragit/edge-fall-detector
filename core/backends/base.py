"""
Abstract base class for inference backends.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import time

from core.schemas import PoseFrame


class InferenceBackend(ABC):
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        self.model_path = model_path
        self.input_size = input_size
        self._warmup_done = False
        self._inference_count = 0
        self._total_latency_ms = 0.0

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[PoseFrame]:
        pass

    @abstractmethod
    def warmup(self, num_iterations: int = 3) -> None:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    def benchmark(self, num_iterations: int = 100) -> dict:
        dummy = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        self.warmup(5)
        latencies = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = self.infer(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)
        return {
            "backend": self.backend_name,
            "device": self.device,
            "mean_ms": round(sum(latencies) / len(latencies), 2),
            "p50_ms": round(sorted(latencies)[len(latencies)//2], 2),
            "p99_ms": round(sorted(latencies)[int(len(latencies)*0.99)], 2),
            "std_ms": round(np.std(latencies), 2),
            "throughput_fps": round(1000 / (sum(latencies) / len(latencies)), 1),
        }

    def _track_latency(self, latency_ms: float) -> None:
        self._inference_count += 1
        self._total_latency_ms += latency_ms

    @property
    def avg_latency_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count