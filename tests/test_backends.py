"""
Tests for inference backends.
"""
import pytest
import numpy as np

from core.backends.mock_backend import MockPoseBackend
from core.backends import create_backend


def test_mock_backend_load():
    backend = MockPoseBackend()
    backend.load()
    assert backend.backend_name == "MockPoseBackend"
    assert backend.device == "cpu (mock)"


def test_mock_backend_infer():
    backend = MockPoseBackend(seed=42)
    backend.load()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    poses = backend.infer(frame)
    assert len(poses) == 1
    pose = poses[0]
    assert 0 <= pose.head.x <= 1
    assert 0 <= pose.head.y <= 1
    assert 0 <= pose.head.confidence <= 1
    assert pose.head.confidence >= 0.5


def test_mock_backend_fall_simulation():
    backend = MockPoseBackend(seed=42, fall_start_frame=5, fall_duration_frames=10)
    backend.load()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    head_positions = []
    for _ in range(20):
        poses = backend.infer(frame)
        head_positions.append(poses[0].head.y)
    pre_fall = head_positions[:4]
    post_fall = head_positions[10:]
    assert max(pre_fall) < min(post_fall), "Head should descend during fall"


def test_mock_backend_benchmark():
    backend = MockPoseBackend()
    backend.load()
    metrics = backend.benchmark(num_iterations=10)
    assert "mean_ms" in metrics
    assert "p50_ms" in metrics
    assert "p99_ms" in metrics
    assert "throughput_fps" in metrics
    assert metrics["throughput_fps"] > 0


def test_backend_factory():
    mock = create_backend("mock")
    assert isinstance(mock, MockPoseBackend)
    with pytest.raises(ValueError):
        create_backend("invalid_backend")