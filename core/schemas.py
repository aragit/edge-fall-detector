"""
Pydantic data schemas for the edge fall detection pipeline.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Optional, Literal
from enum import Enum


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class JointName(str, Enum):
    NOSE = "nose"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


class Joint(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    name: Optional[JointName] = Field(default=None)

    @field_validator('x', 'y')
    @classmethod
    def clamp_normalized(cls, v):
        return max(0.0, min(1.0, v))


class BoundingBox(BaseModel):
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class PoseFrame(BaseModel):
    frame_id: int = Field(..., ge=0)
    timestamp_ms: float = Field(...)
    source_fps: float = Field(default=30.0, gt=0)
    head: Joint = Field(...)
    neck: Optional[Joint] = Field(default=None)
    left_shoulder: Optional[Joint] = Field(default=None)
    right_shoulder: Optional[Joint] = Field(default=None)
    center_shoulder: Optional[Joint] = Field(default=None)
    center_hip: Joint = Field(...)
    left_hip: Optional[Joint] = Field(default=None)
    right_hip: Optional[Joint] = Field(default=None)
    left_knee: Optional[Joint] = Field(default=None)
    right_knee: Optional[Joint] = Field(default=None)
    left_ankle: Joint = Field(...)
    right_ankle: Joint = Field(...)
    bounding_box: Optional[BoundingBox] = Field(default=None)
    torso_vector: Optional[Tuple[float, float]] = Field(default=None)
    torso_angle_deg: Optional[float] = Field(default=None, ge=-180, le=180)


class KinematicState(BaseModel):
    frame_window: List[PoseFrame] = Field(...)
    head_velocity: Tuple[float, float] = Field(...)
    hip_velocity: Tuple[float, float] = Field(...)
    head_speed: float = Field(..., ge=0.0)
    vertical_velocity: float = Field(...)
    vertical_acceleration: Optional[float] = Field(default=None)
    torso_angle_mean: float = Field(..., ge=-180, le=180)
    torso_angle_variance: float = Field(..., ge=0.0)
    mean_confidence: float = Field(..., ge=0.0, le=1.0)
    window_duration_ms: float = Field(..., gt=0)
    effective_fps: float = Field(..., gt=0)


class FallEvent(BaseModel):
    event_id: str = Field(...)
    timestamp_ms: float = Field(...)
    severity: SeverityLevel = Field(default=SeverityLevel.CRITICAL)
    trigger_velocity: float = Field(...)
    trigger_acceleration: Optional[float] = Field(default=None)
    trigger_torso_angle: float = Field(..., ge=-180, le=180)
    pre_fall_frames: List[PoseFrame] = Field(default_factory=list)
    detector_version: str = Field(default="0.2.0-blueprint")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    room_id: Optional[str] = Field(default=None)
    device_id: Optional[str] = Field(default=None)


class FallAlert(BaseModel):
    alert_id: str = Field(...)
    event_id: str = Field(...)
    timestamp_ms: float = Field(...)
    severity: SeverityLevel = Field(default=SeverityLevel.CRITICAL)
    kinematic_velocity: float = Field(...)
    kinematic_acceleration: Optional[float] = Field(default=None)
    torso_angle_at_fall: float = Field(..., ge=-180, le=180)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    room_id: Optional[str] = Field(default=None)
    device_id: Optional[str] = Field(default=None)
    suggested_response_time_sec: int = Field(default=30)
    escalation_path: Literal["nursing_station", "charge_nurse", "code_team"] = Field(
        default="nursing_station"
    )


class DeviceTelemetry(BaseModel):
    device_id: str
    timestamp_ms: float
    cpu_percent: float = Field(..., ge=0.0, le=100.0)
    memory_percent: float = Field(..., ge=0.0, le=100.0)
    gpu_utilization: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    gpu_memory_mb: Optional[float] = Field(default=None, ge=0)
    inference_latency_ms: float = Field(..., ge=0.0)
    inference_fps: float = Field(..., ge=0.0)
    model_version: str
    frames_processed: int = Field(..., ge=0)
    falls_detected: int = Field(..., ge=0)
    false_positives_estimated: int = Field(default=0, ge=0)
    mqtt_connected: bool
    mqtt_messages_sent: int = Field(..., ge=0)
    mqtt_messages_dropped: int = Field(default=0, ge=0)


class PipelineConfig(BaseModel):
    device: Literal["cpu", "cuda", "tensorrt", "mock"] = Field(default="mock")
    tensorrt_engine_path: Optional[str] = Field(default="models/yolo_pose.engine")
    input_source: str = Field(default="0")
    target_fps: float = Field(default=30.0, gt=0)
    frame_width: int = Field(default=640, gt=0)
    frame_height: int = Field(default=480, gt=0)
    temporal_window_size: int = Field(default=10, ge=3, le=60)
    velocity_threshold: float = Field(default=1.5, gt=0)
    acceleration_threshold: Optional[float] = Field(default=3.0)
    torso_angle_threshold: float = Field(default=75.0, ge=0, le=90)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_detection_frames: int = Field(default=3, ge=1)
    cooldown_ms: int = Field(default=30000, ge=0)
    mqtt_broker: Optional[str] = Field(default=None)
    mqtt_port: int = Field(default=1883, ge=1, le=65535)
    mqtt_topic: str = Field(default="hospital/fall_alerts")
    save_frames_on_alert: bool = Field(default=False)
    frame_retention_hours: int = Field(default=24, ge=0)