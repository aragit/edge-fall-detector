#!/usr/bin/env python3
"""
Edge Fall Detection Pipeline — Entry Point
"""
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import PipelineConfig
from core.pipeline import FallDetectionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Edge Fall Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode mock --frames 300 --fall-at 200
  %(prog)s --mode mock --frames 300
  %(prog)s --mode trt --source 0
        """
    )

    parser.add_argument("--mode", type=str, default="mock",
                       choices=["mock", "onnx", "torch", "trt"],
                       help="Inference backend (default: mock)")
    parser.add_argument("--frames", type=int, default=300,
                       help="Number of frames to process")
    parser.add_argument("--fall-at", type=int, default=None,
                       help="Simulate fall starting at this frame (mock mode only)")
    parser.add_argument("--duration", type=float, default=None,
                       help="Run for N seconds instead of frame count")
    parser.add_argument("--source", type=str, default="0",
                       help="Video source: camera index, file path, or RTSP URL")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Target processing FPS")
    parser.add_argument("--width", type=int, default=640,
                       help="Frame width")
    parser.add_argument("--height", type=int, default=480,
                       help="Frame height")
    parser.add_argument("--window-size", type=int, default=10,
                       help="Temporal window size (frames)")
    parser.add_argument("--velocity-threshold", type=float, default=1.5,
                       help="Vertical velocity threshold (norm-units/sec)")
    parser.add_argument("--angle-threshold", type=float, default=75.0,
                       help="Torso angle threshold (degrees from vertical)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                       help="Minimum detection confidence")
    parser.add_argument("--mqtt-broker", type=str, default=None,
                       help="MQTT broker address (omit for console-only)")
    parser.add_argument("--mqtt-topic", type=str, default="hospital/fall_alerts",
                       help="MQTT topic for alerts")
    parser.add_argument("--save-stats", type=str, default=None,
                       help="Save pipeline stats to JSON file")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    config = PipelineConfig(
        device=args.mode,
        input_source=args.source,
        target_fps=args.fps,
        frame_width=args.width,
        frame_height=args.height,
        temporal_window_size=args.window_size,
        velocity_threshold=args.velocity_threshold,
        torso_angle_threshold=args.angle_threshold,
        min_confidence=args.min_confidence,
        mqtt_broker=args.mqtt_broker,
        mqtt_topic=args.mqtt_topic,
    )

    if args.mode == "mock" and args.fall_at is not None:
        import core.backends.mock_backend as mock_mod
        original_init = mock_mod.MockPoseBackend.__init__

        def patched_init(self, *init_args, **init_kwargs):
            init_kwargs["fall_start_frame"] = args.fall_at
            init_kwargs["fall_duration_frames"] = 15
            original_init(self, *init_args, **init_kwargs)

        mock_mod.MockPoseBackend.__init__ = patched_init

    pipeline = FallDetectionPipeline(config)
    stats = pipeline.run(
        num_frames=args.frames if args.duration is None else None,
        duration_seconds=args.duration,
    )

    print("\n" + "=" * 60)
    print("📊 PIPELINE SUMMARY")
    print("=" * 60)
    for key, value in stats.to_dict().items():
        print(f"   {key}: {value}")

    if args.save_stats:
        with open(args.save_stats, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        print(f"\n   Stats saved: {args.save_stats}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()