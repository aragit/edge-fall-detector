"""
Alert dispatch layer for nursing station integration.
"""
import json
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from core.schemas import FallEvent, FallAlert, SeverityLevel


class AlertDispatcher(ABC):
    @abstractmethod
    def dispatch(self, event: FallEvent) -> bool:
        pass

    @abstractmethod
    def health_check(self) -> dict:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class ConsoleDispatcher(AlertDispatcher):
    def __init__(self, verbose: bool = True, **kwargs):
        self.verbose = verbose
        self._alerts_sent = 0

    def dispatch(self, event: FallEvent) -> bool:
        alert = self._event_to_alert(event)

        print("\n" + "=" * 60)
        print("🚨 FALL DETECTION ALERT")
        print("=" * 60)
        print(f"   Alert ID:     {alert.alert_id}")
        print(f"   Event ID:     {alert.event_id}")
        print(f"   Severity:     {alert.severity.value}")
        print(f"   Timestamp:    {alert.timestamp_ms:.0f} ms")
        print(f"   Down Velocity: {alert.kinematic_velocity:.2f} norm-units/sec")
        if alert.kinematic_acceleration:
            print(f"   Acceleration: {alert.kinematic_acceleration:.2f} norm-units/sec²")
        print(f"   Torso Angle:  {alert.torso_angle_at_fall:.1f}°")
        print(f"   Confidence:   {alert.confidence_score:.2%}")
        print(f"   Room ID:      {alert.room_id or 'N/A'}")
        print(f"   Device:       {alert.device_id or 'N/A'}")
        print(f"   Response SLA: {alert.suggested_response_time_sec}s")
        print("=" * 60 + "\n")

        self._alerts_sent += 1
        return True

    def health_check(self) -> dict:
        return {"status": "healthy", "alerts_sent": self._alerts_sent}

    def close(self) -> None:
        pass

    def _event_to_alert(self, event: FallEvent) -> FallAlert:
        return FallAlert(
            alert_id=str(uuid.uuid4()),
            event_id=event.event_id,
            timestamp_ms=event.timestamp_ms,
            severity=event.severity,
            kinematic_velocity=event.trigger_velocity,
            kinematic_acceleration=event.trigger_acceleration,
            torso_angle_at_fall=event.trigger_torso_angle,
            confidence_score=event.confidence_score,
            room_id=event.room_id,
            device_id=event.device_id,
        )


class MQTTDispatcher(AlertDispatcher):
    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "hospital/fall_alerts",
        client_id: Optional[str] = None,
        qos: int = 1,
        retain: bool = False,
        **kwargs
    ):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id or f"fall-detector-{uuid.uuid4().hex[:8]}"
        self.qos = qos
        self.retain = retain

        self._client = None
        self._connected = False
        self._alerts_sent = 0
        self._alerts_dropped = 0

        self._init_mqtt()

    def _init_mqtt(self):
        try:
            import paho.mqtt.client as mqtt

            self._client = mqtt.Client(client_id=self.client_id)
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_publish = self._on_publish

            self._client.connect(self.broker, self.port, keepalive=60)
            self._client.loop_start()
            time.sleep(0.5)

        except ImportError:
            print("[MQTT] paho-mqtt not installed. Alerts will be logged locally.")
            self._client = None
        except Exception as e:
            print(f"[MQTT] Connection failed: {e}. Alerts will be logged locally.")
            self._client = None

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[MQTT] Connected to {self.broker}:{self.port}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        print(f"[MQTT] Disconnected (code {rc})")

    def _on_publish(self, client, userdata, mid):
        pass

    def dispatch(self, event: FallEvent) -> bool:
        alert = FallAlert(
            alert_id=str(uuid.uuid4()),
            event_id=event.event_id,
            timestamp_ms=event.timestamp_ms,
            severity=event.severity,
            kinematic_velocity=event.trigger_velocity,
            kinematic_acceleration=event.trigger_acceleration,
            torso_angle_at_fall=event.trigger_torso_angle,
            confidence_score=event.confidence_score,
            room_id=event.room_id,
            device_id=event.device_id,
        )

        payload = alert.json()

        if self._client and self._connected:
            try:
                result = self._client.publish(
                    self.topic, payload, qos=self.qos, retain=self.retain
                )
                if result.rc == 0:
                    self._alerts_sent += 1
                    print(f"[MQTT] Alert dispatched to {self.topic}")
                    return True
                else:
                    self._alerts_dropped += 1
                    print(f"[MQTT] Publish failed (rc={result.rc})")
                    return False
            except Exception as e:
                self._alerts_dropped += 1
                print(f"[MQTT] Publish error: {e}")
                return False
        else:
            print(f"[MQTT] Not connected. Logging locally:")
            print(payload)
            return False

    def health_check(self) -> dict:
        return {
            "status": "connected" if self._connected else "disconnected",
            "broker": self.broker,
            "port": self.port,
            "alerts_sent": self._alerts_sent,
            "alerts_dropped": self._alerts_dropped,
        }

    def close(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            print("[MQTT] Client disconnected")


def create_dispatcher(backend: str = "console", **kwargs) -> AlertDispatcher:
    backend = backend.lower()
    if backend == "console":
        return ConsoleDispatcher(**kwargs)
    elif backend == "mqtt":
        return MQTTDispatcher(**kwargs)
    elif backend == "mock":
        return ConsoleDispatcher(verbose=False)
    else:
        raise ValueError(f"Unknown dispatcher: {backend}")