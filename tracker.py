"""
Auto-Tracker: YOLO detection + PTZ tracking
===========================================
On motion: detect person/dog → track with C210 PTZ.

Plug into pipeline:
  from tracker import AutoTracker
  tracker = AutoTracker("YOUR_CAMERA_IP")
  pipeline.on_motion(lambda e: tracker.track(e))
"""
import time
import threading
from pathlib import Path
from io import BytesIO


class AutoTracker:
    """Detects and tracks targets using C210 PTZ."""

    def __init__(self, camera_ip="YOUR_CAMERA_IP", model="yolov8n.pt"):
        self.camera_ip = camera_ip
        self.model_name = model
        self._model = None
        self._tapo = None
        self._last_move = 0
        self._cooldown = 1.5  # seconds between moves
        self._tracking = False
        self._lock = threading.Lock()

        # Target classes we care about
        self.target_classes = {
            0: "person",
            16: "dog",
            2: "car",
        }

        # Detection state
        self.total_detections = 0
        self.last_detection = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_name)
        return self._model

    def _load_tapo(self):
        if self._tapo is None:
            from pytapo import Tapo
            self._tapo = Tapo(
                host=self.camera_ip,
                user="YOUR_USER",
                password="YOUR_PASSWORD",
                cloudPassword="YOUR_CLOUD_PASS",
                isKLAP=False,
            )
        return self._tapo

    def track(self, event):
        """
        Called by pipeline on motion event.
        event has: camera_id, frame_jpeg (bytes), timestamp
        """
        if event.camera_id != "C210":
            return  # Only C210 has PTZ

        with self._lock:
            now = time.time()
            if now - self._last_move < self._cooldown:
                return  # Cooldown

            try:
                result = self._detect(event.frame_jpeg)
                if result:
                    x, y, label, confidence = result
                    self._move_to(x, y, label, confidence)
                    self._last_move = now
            except Exception as e:
                pass  # Don't break the pipeline on tracker errors

    def _detect(self, jpeg_bytes):
        """Run YOLO on a JPEG, return (x, y, label, confidence) for best target."""
        model = self._load_model()
        results = model(jpeg_bytes, verbose=False)

        best = None
        best_conf = 0

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])

                if cls_id not in self.target_classes or conf < 0.3:
                    continue

                if conf > best_conf:
                    # Get center of bounding box
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    best = (cx, cy, self.target_classes[cls_id], conf)
                    best_conf = conf

        if best:
            self.total_detections += 1
            self.last_detection = {
                "label": best[2],
                "confidence": best[3],
                "x": best[0],
                "y": best[1],
                "time": time.time(),
            }

        return best

    def _move_to(self, x_norm, y_norm, label, confidence):
        """
        Map normalized image coordinates (0-1) to motor angles.
        Image center = camera center, edges = max pan/tilt.
        """
        tapo = self._load_tapo()

        # Image dimensions (from pipeline): 1920x1080
        # But YOLO coordinates are absolute pixels, not normalized.
        # Let's convert to motor angles.
        # C210 pan range: ~360 degrees, tilt: ~113 degrees

        # Map x (0-1920) to pan angle (-180 to +180)
        pan_angle = ((x_norm / 1920) - 0.5) * 360  # degrees from center

        # Map y (0-1080) to tilt angle (-56 to +56)
        tilt_angle = ((y_norm / 1080) - 0.5) * 113

        # Move camera in small steps based on which side target is on.
        # Avoid over-correction; pytapo step commands are reliable.
        if pan_angle > 25:
            tapo.moveMotorClockWise()
        elif pan_angle < -25:
            tapo.moveMotorCounterClockWise()

        # Tilt support is coarse; only adjust if target is very high/low.
        if tilt_angle < -20:
            tapo.moveMotorVertical()

        print(f"  [TRACK] {label} ({confidence:.2f}) → x={x_norm:.0f} y={y_norm:.0f} pan={pan_angle:.0f}° tilt={tilt_angle:.0f}°")

    def manual_move(self, direction, steps=1):
        """Manual PTZ: 'left', 'right', 'up', 'down'. steps = number of increments."""
        tapo = self._load_tapo()
        for _ in range(steps):
            if direction == "left":
                tapo.moveMotorCounterClockWise()
            elif direction == "right":
                tapo.moveMotorClockWise()
            elif direction == "up":
                tapo.moveMotorVertical()
            elif direction == "down":
                # moveMotorVertical goes up, so down isn't directly available
                pass

    def go_to_preset(self, preset_id):
        tapo = self._load_tapo()
        tapo.setPreset(int(preset_id))

    def calibrate(self):
        tapo = self._load_tapo()
        tapo.calibrateMotor()

    def stats(self):
        return {
            "detections": self.total_detections,
            "last": self.last_detection,
            "tracking": self._tracking,
        }


# ─── Standalone test ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing PTZ...")
        tracker = AutoTracker()
        tracker.manual_move("right", 30)
        time.sleep(2)
        tracker.manual_move("left", 30)
        print("Done - camera should have moved right then left")

    elif len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        print("Calibrating...")
        tracker = AutoTracker()
        tracker.calibrate()
        print("Done")

    else:
        print("AutoTracker ready.")
        print("Usage: python3 tracker.py test|calibrate")
        print("Or import: from tracker import AutoTracker")
