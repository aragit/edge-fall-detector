
# 🏥 Edge Fall Detector — Active Blueprint
<p align="center"><b>Edge ML: Real-Time Patient Fall Detection</b></p>

<p align="center"><sub>NVIDIA Jetson · YOLO-Pose · Temporal Kinematics · MQTT Alerts</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-📐%20Active%20Blueprint-blue" alt="Active Blueprint">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.12+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Tests-14%20passing-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
</p>

---
Real-time patient fall detection on NVIDIA Jetson using YOLOv11-Pose and temporal kinematics. All inference stays on-device — no video leaves the edge.
---

## Architecture

```
┌─────────────┐   ┌────────────────┐   ┌───────────────── ─┐    ┌───────────── ─┐
│ Video Frame │──▶│ Inference      │──▶│ Kinematic State   │ ──▶│ Alert         │
│ (OpenCV)    │   │ (Mock/ONNX/TRT)│   │ Machine (window   │    │ (MQTT/Console)│
└─────────────┘   └────────────────┘   │ vel, accel, angle)│    └──────────── ──┘
                                       └───────────────── ─┘
```

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Inference** | Mock / ONNX / PyTorch / TensorRT | Swappable backends (mock ships; others need install) |
| **Temporal** | Sliding-window kinematics | Velocity, acceleration, torso angle over N frames |
| **Dispatch** | MQTT / Console | Alerts contain kinematics only — no pixels, no PII |
| **Telemetry** | psutil | Device health, latency, throughput |

## Quick Start

```bash
git clone https://github.com/aragit/edge-fall-detector.git
cd edge-fall-detector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run mock demo (no GPU needed)

```bash
# Stable scene — no fall
python scripts/run_pipeline.py --mode mock --frames 90

# Simulated fall at frame 60
python scripts/run_pipeline.py --mode mock --frames 120 --fall-at 60
```

On fall detection the console dispatcher prints:

```
============================================================
🚨 FALL DETECTION ALERT
============================================================
   Alert ID:     550e8400-e29b-41d4-a716-446655440000
   Event ID:     6ba7b810-9dad-11d1-80b4-00c04fd430c8
   Severity:     CRITICAL
   Timestamp:    1781985791798 ms
   Down Velocity: 4.30 norm-units/sec
   Acceleration:  12.84 norm-units/sec²
   Torso Angle:  82.5°
   Confidence:   91.2%
   Room ID:      0
   Device:       3f2e3066-65f2-415b-99a9-4f88983c58d8
   Response SLA: 30s
============================================================
```

### Run tests

```bash
pytest tests/ -v
```

14 tests covering window accumulation, fall detection, confidence gating, cooldown debouncing, backend benchmarks, and pipeline integration.

## Key Differentiators

| Feature | Typical Approach | This Implementation |
|---------|----------------|-------------------|
| Detection logic | Single-frame threshold ("head below knee") | Temporal window: velocity + acceleration + torso angle |
| False-positive handling | Manual tuning | Confidence gating + consecutive frame requirement + cooldown |
| Hardware portability | CUDA-only | 4 backend stubs: Mock → ONNX → PyTorch → TensorRT |
| Privacy | Cloud video streaming | On-device inference; MQTT transmits kinematics only |
| Testing | Manual validation | 14 automated tests, deterministic mock, CI-ready |

## Roadmap

**Phase 1: Active Blueprint (current)**
- [x] Inference backend abstraction with factory
- [x] Temporal kinematic state machine (velocity, acceleration, angle)
- [x] Deterministic mock backend with configurable fall simulation
- [x] Alert dispatch (console + MQTT)
- [x] 14 automated tests

**Phase 2: Hardware integration**
- [ ] ONNX Runtime backend
- [ ] PyTorch backend
- [ ] TensorRT engine build from YOLO11-Pose
- [ ] Jetson Orin deployment (MAXN power, jetson_clocks)
- [ ] Real camera capture (RTSP / V4L2 / GStreamer)

**Phase 3: Production hardening**
- [ ] Multi-person tracking
- [ ] Fall vs. intentional lie-down classification
- [ ] Config hot-reload
- [ ] Prometheus metrics endpoint

## Repository Structure

```
edge-fall-detector/
├── README.md
├── requirements.txt
├── configs/
│   └── base.yaml                  # Pipeline configuration
├── core/
│   ├── schemas.py                 # Pydantic data contracts
│   ├── backends/
│   │   ├── __init__.py            # Backend factory
│   │   ├── base.py                # Abstract inference backend
│   │   └── mock_backend.py        # Deterministic fall simulation
│   ├── state_machine.py           # Temporal kinematics engine
│   ├── alert_dispatcher.py        # Console + MQTT dispatch
│   └── pipeline.py                # Main orchestrator
├── scripts/
│   └── run_pipeline.py            # CLI entry point
├── tests/
│   ├── test_backends.py           # 5 tests
│   ├── test_state_machine.py      # 6 tests
│   └── test_pipeline.py           # 3 tests
└── models/                        # .engine / .onnx / .pt (gitignored)
```

## Technology Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Pose model | YOLO11-Pose | 1.72ms @ TensorRT on Jetson Orin Nano |
| Temporal logic | Custom state machine | Domain-specific, interpretable |
| Config | Pydantic + YAML | Type-safe, IDE-friendly |
| Testing | pytest | Standard, CI-ready |
| Target deployment | TensorRT + Jetson Orin | INT8: ~168 FPS reported |

## Privacy

- **No raw pixels** transmitted — only kinematic summaries leave the device
- **No patient identifiers** — rooms mapped to anonymized UUIDs
- **Deterministic mock** — reproducible for validation
- **Human-in-the-loop** — alerts notify staff; no autonomous actions

## References

- [YOLO11 Pose — Ultralytics Docs](https://docs.ultralytics.com/models/yolo11/#key-features)
- [TensorRT on Jetson — NVIDIA Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html)
- [Jetson Orin Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks)

---

<p align="center"><sub>Built for patient safety. Designed for edge deployment.</sub></p>
