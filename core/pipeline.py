"""
Edge Fall Detection Pipeline — Main Orchestrator.
"""
import time
import uuid
import signal
import sys
from typing import Optional, Callable
from dataclasses import dataclass, field

import numpy as np

from core.schemas import (
    PipelineConfig, PoseFrame, FallEvent, KinematicState,
    DeviceTelemetry, SeverityLevel
)
from core.backends.base import InferenceBackend
from core.backends import create_backend
from core.state_machine import KinematicStateMachine
from core.alert_dispatcher import AlertDispatcher, create_dispatcher


@dataclass
class PipelineStats:
    frames_processed: int = 0
    frames_dropped: int = 0
    inferences_run: int = 0
    falls_detected: int = 0
    alerts_dispatched: int = 0
    alerts_failed: int = 0
    start_time_ms: float = field(default_factory=lambda: time.time() * 1000)

    @property
    def uptime_seconds(self) -> float:
        return (time.time() * 1000 - self.start_time_ms) / 1000.0

    @property
    def effective_fps(self) -> float:
        return self.frames_processed / max(self.uptime_seconds, 0.001)

    def to_dict(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "frames_dropped": self.frames_dropped,
            "inferences_run": self.inferences_run,
            "falls_detected": self.falls_detected,
            "alerts_dispatched": self.alerts_dispatched,
            "alerts_failed": self.alerts_failed,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "effective_fps": round(self.effective_fps, 2),
        }


class FallDetectionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device_id = str(uuid.uuid4())
        self.room_id = config.input_source

        self._backend: Optional[InferenceBackend] = None
        self._state_machine: Optional[KinematicStateMachine] = None
        self._dispatcher: Optional[AlertDispatcher] = None

        self._running = False
        self._stats = PipelineStats()
        self._frame_callback: Optional[Callable] = None
        self._fall_callback: Optional[Callable] = None

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        def handler(signum, frame):
            print(f"\n[Pipeline] Received signal {signum}, shutting down gracefully...")
            self._running = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def initialize(self) -> None:
        print("=" * 60)
        print("🏥 Edge Fall Detection Pipeline — Initializing")
        print("=" * 60)

        print(f"\n[1/4] Loading inference backend: {self.config.device}")
        self._backend = create_backend(
            backend_type=self.config.device,
            model_path=self.config.tensorrt_engine_path or "models/yolo_pose.engine",
        )
        self._backend.load()
        self._backend.warmup()

        print(f"\n[2/4] Initializing temporal state machine")
        print(f"      Window size: {self.config.temporal_window_size} frames")
        print(f"      Velocity threshold: {self.config.velocity_threshold} norm-units/sec")
        print(f"      Torso angle threshold: {self.config.torso_angle_threshold}°")
        self._state_machine = KinematicStateMachine(
            window_size=self.config.temporal_window_size,
            velocity_threshold=self.config.velocity_threshold,
            acceleration_threshold=self.config.acceleration_threshold,
            torso_angle_threshold=self.config.torso_angle_threshold,
            min_confidence=self.config.min_confidence,
            min_consecutive_frames=self.config.min_detection_frames,
            cooldown_ms=self.config.cooldown_ms,
        )

        print(f"\n[3/4] Configuring alert dispatcher")
        dispatcher_backend = "mqtt" if self.config.mqtt_broker else "console"
        self._dispatcher = create_dispatcher(
            backend=dispatcher_backend,
            broker=self.config.mqtt_broker or "localhost",
            port=self.config.mqtt_port,
            topic=self.config.mqtt_topic,
        )

        print(f"\n[4/4] Health check")
        health = self._dispatcher.health_check()
        print(f"      Dispatcher: {health['status']}")
        print(f"      Device ID: {self.device_id}")

        print("\n" + "=" * 60)
        print("✅ Pipeline initialized and ready")
        print("=" * 60 + "\n")

    def run(
        self,
        num_frames: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        source: Optional[str] = None,
    ) -> PipelineStats:
        if self._backend is None:
            self.initialize()

        self._running = True
        frame_idx = 0
        start_time = time.time()

        print(f"[*] Starting stream processing...")
        print(f"    Source: {source or self.config.input_source}")
        print(f"    Target FPS: {self.config.target_fps}")
        print(f"    Resolution: {self.config.frame_width}x{self.config.frame_height}")
        print()

        try:
            while self._running:
                if num_frames is not None and frame_idx >= num_frames:
                    break
                if duration_seconds is not None and (time.time() - start_time) >= duration_seconds:
                    break

                frame = self._acquire_frame(frame_idx)
                if frame is None:
                    self._stats.frames_dropped += 1
                    continue

                poses = self._backend.infer(frame)
                self._stats.inferences_run += 1

                for pose in poses:
                    fall_detected, kinematics = self._state_machine.update_and_check(pose)

                    if fall_detected and kinematics:
                        self._handle_fall(pose, kinematics, frame_idx)

                self._stats.frames_processed += 1
                frame_idx += 1

                if frame_idx % 30 == 0:
                    self._print_status(frame_idx)

                if self.config.target_fps > 0:
                    target_interval = 1.0 / self.config.target_fps
                    elapsed = time.time() - start_time
                    expected_time = frame_idx * target_interval
                    sleep_time = expected_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except Exception as e:
            print(f"\n[Pipeline] ERROR: {e}")
            raise
        finally:
            self._shutdown()

        return self._stats

    def _acquire_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        return np.zeros(
            (self.config.frame_height, self.config.frame_width, 3),
            dtype=np.uint8
        )

    def _handle_fall(self, pose: PoseFrame, kinematics: KinematicState, frame_idx: int) -> None:
        self._stats.falls_detected += 1

        event = FallEvent(
            event_id=str(uuid.uuid4()),
            timestamp_ms=pose.timestamp_ms,
            severity=SeverityLevel.CRITICAL,
            trigger_velocity=kinematics.vertical_velocity,
            trigger_acceleration=kinematics.vertical_acceleration,
            trigger_torso_angle=kinematics.torso_angle_mean,
            pre_fall_frames=list(self._state_machine.window)[:-1] if self._state_machine else [],
            detector_version="0.2.0-blueprint",
            confidence_score=kinematics.mean_confidence,
            room_id=self.room_id,
            device_id=self.device_id,
        )

        success = self._dispatcher.dispatch(event)
        if success:
            self._stats.alerts_dispatched += 1
        else:
            self._stats.alerts_failed += 1

        if self._fall_callback:
            self._fall_callback(event, kinematics)

    def _print_status(self, frame_idx: int) -> None:
        fps = self._stats.effective_fps
        latency = self._backend.avg_latency_ms if self._backend else 0
        print(
            f"[Stream] Frame {frame_idx:>5} | "
            f"FPS: {fps:>5.1f} | "
            f"Inference: {latency:>5.1f}ms | "
            f"Status: NOMINAL"
        )

    def _shutdown(self) -> None:
        print(f"\n[Pipeline] Shutting down...")
        print(f"         Processed: {self._stats.frames_processed} frames")
        print(f"         Falls detected: {self._stats.falls_detected}")
        print(f"         Alerts dispatched: {self._stats.alerts_dispatched}")
        print(f"         Uptime: {self._stats.uptime_seconds:.1f}s")

        if self._backend:
            self._backend.release()
        if self._dispatcher:
            self._dispatcher.close()

        print("[Pipeline] Shutdown complete.\n")

    def get_telemetry(self) -> DeviceTelemetry:
        import psutil

        return DeviceTelemetry(
            device_id=self.device_id,
            timestamp_ms=time.time() * 1000,
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            gpu_utilization=None,
            gpu_memory_mb=None,
            inference_latency_ms=self._backend.avg_latency_ms if self._backend else 0,
            inference_fps=self._stats.effective_fps,
            model_version="0.2.0-blueprint",
            frames_processed=self._stats.frames_processed,
            falls_detected=self._stats.falls_detected,
            false_positives_estimated=0,
            mqtt_connected=self._dispatcher.health_check().get("status") == "connected" if self._dispatcher else False,
            mqtt_messages_sent=self._dispatcher.health_check().get("alerts_sent", 0) if self._dispatcher else 0,
            mqtt_messages_dropped=self._dispatcher.health_check().get("alerts_dropped", 0) if self._dispatcher else 0,
        )