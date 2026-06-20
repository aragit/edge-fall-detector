"""
Integration tests for the full pipeline.
"""
import pytest

from core.schemas import PipelineConfig
from core.pipeline import FallDetectionPipeline


def test_mock_pipeline_no_fall():
    config = PipelineConfig(
        device="mock",
        temporal_window_size=5,
        velocity_threshold=2.0,
        torso_angle_threshold=80.0,
        min_confidence=0.5,
    )
    pipeline = FallDetectionPipeline(config)
    stats = pipeline.run(num_frames=60)
    assert stats.frames_processed == 60
    assert stats.falls_detected == 0
    assert stats.alerts_dispatched == 0
    assert stats.effective_fps > 20


def test_mock_pipeline_with_fall():
    import core.backends.mock_backend as mock_mod
    original_init = mock_mod.MockPoseBackend.__init__

    def fall_init(self, *args, **kwargs):
        kwargs["fall_start_frame"] = 10  # Earlier fall start
        kwargs["fall_duration_frames"] = 8  # Faster fall
        original_init(self, *args, **kwargs)

    mock_mod.MockPoseBackend.__init__ = fall_init

    try:
        config = PipelineConfig(
            device="mock",
            temporal_window_size=5,
            velocity_threshold=0.3,  # Lower threshold to catch fall
            torso_angle_threshold=20.0,  # Lower angle threshold
            min_confidence=0.5,
            min_detection_frames=2,  # Fewer consecutive frames needed
        )
        pipeline = FallDetectionPipeline(config)
        stats = pipeline.run(num_frames=60)
        assert stats.falls_detected >= 1, f"Expected at least 1 fall, got {stats.falls_detected}"
        assert stats.alerts_dispatched >= 1
    finally:
        mock_mod.MockPoseBackend.__init__ = original_init


def test_pipeline_stats_format():
    config = PipelineConfig(device="mock")
    pipeline = FallDetectionPipeline(config)
    stats = pipeline.run(num_frames=30)
    d = stats.to_dict()
    expected_keys = [
        "frames_processed", "frames_dropped", "inferences_run",
        "falls_detected", "alerts_dispatched", "alerts_failed",
        "uptime_seconds", "effective_fps"
    ]
    for key in expected_keys:
        assert key in d, f"Missing key: {key}"