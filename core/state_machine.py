"""
Temporal Kinematic State Machine for Fall Detection.
"""
from collections import deque
from typing import List, Optional, Tuple
import numpy as np
import time

from core.schemas import PoseFrame, KinematicState


class KinematicStateMachine:
    def __init__(
        self,
        window_size: int = 10,
        velocity_threshold: float = 1.5,
        acceleration_threshold: Optional[float] = 3.0,
        torso_angle_threshold: float = 75.0,
        min_confidence: float = 0.5,
        min_consecutive_frames: int = 3,
        cooldown_ms: int = 30000,
    ):
        self.window = deque(maxlen=window_size)
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.torso_angle_threshold = torso_angle_threshold
        self.min_confidence = min_confidence
        self.min_consecutive_frames = min_consecutive_frames
        self.cooldown_ms = cooldown_ms

        self._consecutive_triggers = 0
        self._last_alert_ms: Optional[float] = None
        self._total_frames = 0
        self._frames_below_confidence = 0

    def update_and_check(self, frame: PoseFrame) -> Tuple[bool, Optional[KinematicState]]:
        self._total_frames += 1

        if frame.head.confidence < self.min_confidence:
            self._frames_below_confidence += 1
            return False, None

        self.window.append(frame)

        if len(self.window) < self.window.maxlen:
            return False, None

        state = self._compute_kinematics()
        is_trigger = self._evaluate_trigger(state)

        if is_trigger:
            self._consecutive_triggers += 1
        else:
            self._consecutive_triggers = 0

        if self._last_alert_ms is not None:
            elapsed_ms = frame.timestamp_ms - self._last_alert_ms
            if elapsed_ms < self.cooldown_ms:
                return False, state

        fall_detected = self._consecutive_triggers >= self.min_consecutive_frames

        if fall_detected:
            self._last_alert_ms = frame.timestamp_ms
            self._consecutive_triggers = 0

        return fall_detected, state

    def _compute_kinematics(self) -> KinematicState:
        frames = list(self.window)
        n = len(frames)

        timestamps = np.array([f.timestamp_ms for f in frames])
        dt = np.diff(timestamps) / 1000.0
        dt = np.maximum(dt, 0.001)

        head_x = np.array([f.head.x for f in frames])
        head_y = np.array([f.head.y for f in frames])
        hip_x = np.array([f.center_hip.x for f in frames])
        hip_y = np.array([f.center_hip.y for f in frames])

        head_vx = self._compute_velocity(head_x, dt)
        head_vy = self._compute_velocity(head_y, dt)
        hip_vx = self._compute_velocity(hip_x, dt)
        hip_vy = self._compute_velocity(hip_y, dt)

        head_velocity = (head_vx[-1], head_vy[-1])
        hip_velocity = (hip_vx[-1], hip_vy[-1])
        head_speed = np.sqrt(head_vx[-1]**2 + head_vy[-1]**2)
        vertical_velocity = head_vy[-1]

        head_ay = self._compute_velocity(head_vy, dt)
        vertical_acceleration = head_ay[-1] if len(head_ay) > 0 else None

        torso_angles = np.array([
            f.torso_angle_deg if f.torso_angle_deg is not None else 0.0
            for f in frames
        ])
        confidences = np.array([f.head.confidence for f in frames])
        mean_confidence = np.mean(confidences)

        window_duration_ms = timestamps[-1] - timestamps[0]
        effective_fps = (n - 1) / (window_duration_ms / 1000.0) if window_duration_ms > 0 else 0.0

        return KinematicState(
            frame_window=frames,
            head_velocity=head_velocity,
            hip_velocity=hip_velocity,
            head_speed=round(float(head_speed), 4),
            vertical_velocity=round(float(vertical_velocity), 4),
            vertical_acceleration=round(float(vertical_acceleration), 4) if vertical_acceleration is not None else None,
            torso_angle_mean=round(float(np.mean(torso_angles)), 2),
            torso_angle_variance=round(float(np.var(torso_angles)), 4),
            mean_confidence=round(float(mean_confidence), 3),
            window_duration_ms=round(float(window_duration_ms), 2),
            effective_fps=round(float(effective_fps), 2),
        )

    def _compute_velocity(self, positions: np.ndarray, dt: np.ndarray) -> np.ndarray:
        v = np.zeros(len(positions))
        v[0] = (positions[1] - positions[0]) / dt[0] if len(dt) > 0 else 0
        v[-1] = (positions[-1] - positions[-2]) / dt[-1] if len(dt) > 0 else 0
        for i in range(1, len(positions) - 1):
            v[i] = (positions[i+1] - positions[i-1]) / (dt[i-1] + dt[i])
        return v

    def _evaluate_trigger(self, state: KinematicState) -> bool:
        velocity_trigger = state.vertical_velocity > self.velocity_threshold
        angle_trigger = abs(state.torso_angle_mean) > self.torso_angle_threshold
        accel_trigger = True
        if self.acceleration_threshold is not None and state.vertical_acceleration is not None:
            accel_trigger = state.vertical_acceleration > self.acceleration_threshold
        confidence_trigger = state.mean_confidence >= self.min_confidence
        return velocity_trigger and angle_trigger and accel_trigger and confidence_trigger

    def get_diagnostics(self) -> dict:
        return {
            "total_frames_processed": self._total_frames,
            "frames_below_confidence": self._frames_below_confidence,
            "confidence_rejection_rate": (
                self._frames_below_confidence / max(self._total_frames, 1)
            ),
            "window_fill_ratio": len(self.window) / self.window.maxlen,
            "current_consecutive_triggers": self._consecutive_triggers,
            "last_alert_ms": self._last_alert_ms,
        }

    def reset(self) -> None:
        self.window.clear()
        self._consecutive_triggers = 0
        self._last_alert_ms = None
        self._total_frames = 0
        self._frames_below_confidence = 0