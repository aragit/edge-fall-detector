"""
Mock inference backend for Active Blueprint demonstration.
"""
import numpy as np
import time
import random
from typing import List, Tuple, Optional

from core.backends.base import InferenceBackend
from core.schemas import PoseFrame, Joint, BoundingBox, JointName


class MockPoseBackend(InferenceBackend):
    def __init__(
        self,
        model_path: str = "models/yolo_pose.engine",
        input_size: Tuple[int, int] = (640, 640),
        seed: int = 42,
        fall_start_frame: Optional[int] = None,
        fall_duration_frames: int = 10,
    ):
        super().__init__(model_path, input_size)
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.fall_start_frame = fall_start_frame
        self.fall_duration_frames = fall_duration_frames
        self._frame_counter = 0

        self._base_pose = {
            "head": (0.50, 0.15),
            "neck": (0.50, 0.22),
            "left_shoulder": (0.40, 0.25),
            "right_shoulder": (0.60, 0.25),
            "center_shoulder": (0.50, 0.25),
            "center_hip": (0.50, 0.50),
            "left_hip": (0.42, 0.50),
            "right_hip": (0.58, 0.50),
            "left_knee": (0.40, 0.70),
            "right_knee": (0.60, 0.70),
            "left_ankle": (0.38, 0.90),
            "right_ankle": (0.62, 0.90),
        }
        self._position_noise = 0.02
        self._confidence_noise = 0.05
        self._confidence_base = 0.92

    @property
    def backend_name(self) -> str:
        return "MockPoseBackend"

    @property
    def device(self) -> str:
        return "cpu (mock)"

    def load(self) -> None:
        print(f"[MockBackend] Allocating virtual memory for {self.model_path}...")
        print(f"[MockBackend] Mock CUDA Context Initialized (device: {self.device})")
        print(f"[MockBackend] Ready. Inference will be deterministic.")

    def infer(self, frame: np.ndarray) -> List[PoseFrame]:
        t0 = time.perf_counter()
        time.sleep(0.015)
        self._frame_counter += 1
        fall_offset_y = self._compute_fall_offset()
        fall_progress = self._compute_fall_progress()
        joints = self._generate_joints(fall_offset_y, fall_progress)

        torso_dx = joints["center_shoulder"][0] - joints["center_hip"][0]
        torso_dy = joints["center_shoulder"][1] - joints["center_hip"][1]
        torso_angle = np.degrees(np.arctan2(torso_dx, -torso_dy))

        pose = PoseFrame(
            frame_id=self._frame_counter,
            timestamp_ms=time.time() * 1000,
            source_fps=30.0,
            head=Joint(x=joints["head"][0], y=joints["head"][1],
                      confidence=self._noise_confidence(), name=JointName.NOSE),
            neck=Joint(x=joints["neck"][0], y=joints["neck"][1],
                      confidence=self._noise_confidence()),
            left_shoulder=Joint(x=joints["left_shoulder"][0], y=joints["left_shoulder"][1],
                               confidence=self._noise_confidence()),
            right_shoulder=Joint(x=joints["right_shoulder"][0], y=joints["right_shoulder"][1],
                                confidence=self._noise_confidence()),
            center_shoulder=Joint(x=joints["center_shoulder"][0], y=joints["center_shoulder"][1],
                                 confidence=self._noise_confidence()),
            center_hip=Joint(x=joints["center_hip"][0], y=joints["center_hip"][1],
                            confidence=self._noise_confidence()),
            left_hip=Joint(x=joints["left_hip"][0], y=joints["left_hip"][1],
                          confidence=self._noise_confidence()),
            right_hip=Joint(x=joints["right_hip"][0], y=joints["right_hip"][1],
                           confidence=self._noise_confidence()),
            left_knee=Joint(x=joints["left_knee"][0], y=joints["left_knee"][1],
                           confidence=self._noise_confidence()),
            right_knee=Joint(x=joints["right_knee"][0], y=joints["right_knee"][1],
                            confidence=self._noise_confidence()),
            left_ankle=Joint(x=joints["left_ankle"][0], y=joints["left_ankle"][1],
                            confidence=self._noise_confidence()),
            right_ankle=Joint(x=joints["right_ankle"][0], y=joints["right_ankle"][1],
                             confidence=self._noise_confidence()),
            bounding_box=BoundingBox(
                x1=max(0.0, joints["head"][0] - 0.15),
                y1=max(0.0, joints["head"][1] - 0.05),
                x2=min(1.0, joints["right_ankle"][0] + 0.15),
                y2=min(1.0, joints["right_ankle"][1] + 0.05),
                confidence=0.95,
            ),
            torso_vector=(torso_dx, torso_dy),
            torso_angle_deg=round(float(torso_angle), 2),
        )

        latency_ms = (time.perf_counter() - t0) * 1000
        self._track_latency(latency_ms)
        return [pose]

    def _compute_fall_offset(self) -> float:
        if self.fall_start_frame is None:
            return 0.0
        if self._frame_counter < self.fall_start_frame:
            return 0.0
        progress = (self._frame_counter - self.fall_start_frame) / self.fall_duration_frames
        progress = min(progress, 1.0)
        return progress * progress * progress * 0.65

    def _compute_fall_progress(self) -> float:
        if self.fall_start_frame is None:
            return 0.0
        if self._frame_counter < self.fall_start_frame:
            return 0.0
        progress = (self._frame_counter - self.fall_start_frame) / self.fall_duration_frames
        return min(progress, 1.0)

    def _generate_joints(self, fall_offset_y: float, fall_progress: float = 0.0) -> dict:
        joints = {}
        for name, (bx, by) in self._base_pose.items():
            noise_x = self.np_rng.normal(0, self._position_noise)
            noise_y = self.np_rng.normal(0, self._position_noise)

            extra_x, extra_y = 0.0, 0.0
            if fall_progress > 0 and name in (
                "head", "neck", "left_shoulder", "right_shoulder", "center_shoulder"
            ):
                extra_x = fall_progress * 0.25
                extra_y = fall_progress * 0.12

            x = np.clip(bx + noise_x + extra_x, 0.0, 1.0)
            y = np.clip(by + noise_y + extra_y + fall_offset_y, 0.0, 1.0)
            joints[name] = (round(x, 4), round(y, 4))
        return joints

    def _noise_confidence(self) -> float:
        noise = self.np_rng.normal(0, self._confidence_noise)
        return round(np.clip(self._confidence_base + noise, 0.0, 1.0), 3)

    def warmup(self, num_iterations: int = 3) -> None:
        time.sleep(0.005 * num_iterations)
        self._warmup_done = True

    def release(self) -> None:
        print("[MockBackend] Releasing virtual resources...")
        self._warmup_done = False