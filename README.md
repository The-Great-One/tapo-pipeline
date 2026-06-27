# Tapo Cam Pipeline

**Real-time Tapo camera streaming + motion detection + AI auto-tracking.**

Turns cheap TP-Link Tapo C210/C200 cameras into a full surveillance & tracking system.  
Runs on any Linux server. 1080p 23fps, WebSocket live view, YOLO object detection,  
PTZ auto-tracking, motion event logging, global access via WireGuard.

```
http://your-server:8090          ← Live view (WebSocket real-time)
      /api/stats                 ← JSON: fps, detections, tracker state
      /api/events                ← JSON: motion events with 1080p snapshots
      /api/frame/C210            ← Latest JPEG
      /api/stream                ← MJPEG live stream
      /api/snapshot/C210         ← Force snapshot
```

---

## Hardware

- **TP-Link Tapo C210** (PTZ — pan/tilt/zoom)
- **TP-Link Tapo C200** (fixed — gate/entry duty)
- Any Linux server (Oracle ARM, Raspberry Pi 4, x86 box)

Cameras connect via RTSP on your LAN.  
For global access, run the server on a cloud VM and tunnel through WireGuard.

---

## Quick Start

### 1. Enable RTSP on your cameras

Tapo app → Camera → ⚙️ → Advanced → **Camera Account**

```
Username: tapocam      (or any 6-char name)
Password: yourpass
```

### 2. Clone and install

```bash
git clone https://github.com/The-Great-One/tapo-pipeline.git
cd tapo-pipeline

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure cameras

Edit `config.json`:

```json
{
  "cameras": {
    "C210": {
      "name": "Lift Cam",
      "rtsp": "rtsp://youruser:yourpass@YOUR_C210_IP:554/stream1",
      "ptz": true
    },
    "C200": {
      "name": "Gate Cam",
      "rtsp": "rtsp://youruser:yourpass@YOUR_C200_IP:554/stream1",
      "ptz": false
    }
  }
}
```

### 4. Run

```bash
python pipeline.py --port 8090
```

Open `http://localhost:8090`.

---

## Auto-Tracking (C210 PTZ)

The pipeline automatically loads the tracker. When motion is detected on the C210:

1. YOLOv8 detects **person**, **dog**, or **car**
2. Tracker maps the detection position to motor angles
3. Camera pans/tilts to follow the target

To enable on an existing install:

```bash
source venv/bin/activate
pip install ultralytics pytapo
python pipeline.py --port 8090
# → "Auto-tracker active — YOLO + PTZ tracking"
```

Tracker state is in `/api/stats`:

```json
"tracker": {
  "detections": 42,
  "last": {"label": "person", "confidence": 0.87, "x": 723, "y": 345},
  "tracking": true
}
```

---

## Systemd (production)

```ini
[Unit]
Description=Tapo Camera Pipeline
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tapo-pipeline
ExecStart=/home/ubuntu/tapo-pipeline/venv/bin/python pipeline.py --port 8090
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now tapo-pipeline
```

---

## WireGuard Global Access

Run the pipeline on a cloud server. Point cameras at it through WireGuard:

```
Cameras (LAN) → WireGuard → Oracle Cloud → pipeline → nginx → tapo.yourdomain.com
```

The pipeline only needs RTSP access to the cameras.  
Cameras don't need public IPs or port forwarding.

Example nginx proxy:

```nginx
server {
    listen 80;
    server_name tapo.yourdomain.com;

    location / {
        proxy_pass http://localhost:8090;
    }
    location /ws {
        proxy_pass http://localhost:8091;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Python API

```python
from pipeline import CameraPipeline

p = CameraPipeline()
p.start()

# Get frame as numpy array
frame = p.get_frame_numpy("C210")  # shape: (360, 640, 3)

# Capture full 1080p snapshot
jpeg = p.get_full_res("C210")

# Hook into motion events
p.on_motion(lambda event: print(
    f"Motion on {event.camera_id} at {event.timestamp}"
))

# Access the tracker
p.tracker.manual_move("right", steps=3)
p.tracker.calibrate()
print(p.tracker.stats())
```

---

## Requirements

- Python 3.10+
- ffmpeg (for RTSP decode)
- Pillow (fast JPEG encoding)
- pytapo (PTZ control)
- ultralytics (YOLO tracking)

```bash
pip install -r requirements.txt
```

---

## Architecture

```
┌──────────┐     RTSP      ┌──────────┐    WebSocket    ┌──────────┐
│  C210 PTZ│──────────────→│          │←───────────────→│ Browser  │
│          │               │ Pipeline │                 │          │
│  C200    │               │  ffmpeg  │    HTTP API     │ curl/API │
└──────────┘               │  YOLO    │←────────────────│          │
                           │  pytapo  │                 └──────────┘
                           │  Motion  │
                           │  Events  │
                           └──────────┘
```

- **ffmpeg** decodes RTSP → raw RGB frames (640×360 @ 25fps)
- **Pillow** encodes JPEG in-memory (no subprocess overhead)
- **YOLOv8 nano** runs inference on motion events (not every frame)
- **pytapo** sends PTZ commands via HTTPS to the C210
- **WebSocket** pushes frames to browsers at 30fps (no polling)

---

## License

MIT — see [LICENSE](LICENSE).
