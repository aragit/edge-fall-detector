"""
Tests for the temporal kinematic state machine.
"""
import pytest
import numpy as np

from core.schemas import PoseFrame, Joint, JointName
from core.state_machine import KinematicStateMachine


def make_pose(frame_id: int, head_y: float, head_conf: float = 0.95,
              torso_angle: float = 0.0, timestamp_ms: float = None) -> PoseFrame:
    ts = timestamp_ms or frame_id * 33.33
    # Clamp head_y to valid normalized range
    head_y = max(0.0, min(1.0, head_y))
    return PoseFrame(
        frame_id=frame_id,
        timestamp_ms=ts,
        source_fps=30.0,
        head=Joint(x=0.5, y=head_y, confidence=head_conf, name=JointName.NOSE),
        center_hip=Joint(x=0.5, y=0.50, confidence=0.90),
        left_ankle=Joint(x=0.40, y=0.90, confidence=0.88),
        right_ankle=Joint(x=0.60, y=0.90, confidence=0.88),
        torso_angle_deg=torso_angle,
        torso_vector=(0.0, -0.35),
    )


def test_window_accumulation():
    sm = KinematicStateMachine(window_size=5)
    for i in range(3):
        detected, _ = sm.update_and_check(make_pose(i, 0.2))
        assert not detected
    assert len(sm.window) == 3
    for i in range(3, 10):
        detected, _ = sm.update_and_check(make_pose(i, 0.2))
    assert len(sm.window) == 5


def test_no_fall_upright():
    sm = KinematicStateMachine(window_size=5, velocity_threshold=1.5)
    for i in range(10):
        detected, kinematics = sm.update_and_check(make_pose(i, 0.2, torso_angle=5.0))
        assert not detected
        if kinematics:
            assert abs(kinematics.torso_angle_mean) < 10


def test_fall_detection():
    sm = KinematicStateMachine(
        window_size=5,
        velocity_threshold=0.5,
        torso_angle_threshold=30.0,
        min_consecutive_frames=3,
        acceleration_threshold=None,
    )
    for i in range(5):
        detected, _ = sm.update_and_check(make_pose(i, 0.2, torso_angle=5.0))
        assert not detected

    # Rapid fall: head drops from 0.2 to 0.85 over 10 frames, angle goes to 85 degrees
    fall_found = False
    for i in range(5, 15):
        head_y = 0.2 + (i - 5) * 0.065  # Slower descent, stays within 0-1
        angle = 5 + (i - 5) * 8         # Slower angle change
        detected, kinematics = sm.update_and_check(
            make_pose(i, head_y, torso_angle=angle)
        )
        if detected:
            fall_found = True
            break

    assert fall_found, "Fall should be detected with rapid descent and angle change"


def test_confidence_gate():
    sm = KinematicStateMachine(window_size=5, min_confidence=0.6)
    for i in range(5):
        detected, _ = sm.update_and_check(make_pose(i, 0.2, head_conf=0.3))
        assert not detected
    diagnostics = sm.get_diagnostics()
    assert diagnostics["frames_below_confidence"] == 5
    assert diagnostics["confidence_rejection_rate"] == 1.0


def test_cooldown():
    sm = KinematicStateMachine(
        window_size=5,
        velocity_threshold=0.1,
        torso_angle_threshold=10.0,
        min_consecutive_frames=1,
        cooldown_ms=1000,
        acceleration_threshold=None,
    )
    for i in range(5):
        detected, _ = sm.update_and_check(
            make_pose(i, 0.2 + i * 0.15, torso_angle=80.0, timestamp_ms=i * 33)
        )
    assert detected
    diagnostics = sm.get_diagnostics()
    assert diagnostics["last_alert_ms"] is not None


def test_diagnostics():
    sm = KinematicStateMachine(window_size=5)
    for i in range(10):
        sm.update_and_check(make_pose(i, 0.2))
    diag = sm.get_diagnostics()
    assert diag["total_frames_processed"] == 10
    assert diag["window_fill_ratio"] == 1.0
    assert "confidence_rejection_rate" in diag