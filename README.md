<div align="center">

<img src="icon.png" alt="tapo-pipeline logo" width="120" height="120" />

# tapo-pipeline

**Real-time Tapo camera streaming + motion detection + AI auto-tracking.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/The-Great-One/tapo-pipeline)](https://github.com/The-Great-One/tapo-pipeline/issues)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/The-Great-One/tapo-pipeline)](https://github.com/The-Great-One/tapo-pipeline/pulls)
[![GitHub stars](https://img.shields.io/github/stars/The-Great-One/tapo-pipeline?style=social)](https://github.com/The-Great-One/tapo-pipeline/stargazers)

[Report Bug](https://github.com/The-Great-One/tapo-pipeline/issues/new?template=bug_report.md) · [Request Feature](https://github.com/The-Great-One/tapo-pipeline/issues/new?template=feature_request.md)

</div>

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## About

`tapo-pipeline` turns cheap TP-Link Tapo C210/C200 cameras into a full
surveillance and tracking system. Runs on any Linux server: 1080p 23fps,
WebSocket live view, YOLO object detection, PTZ auto-tracking, motion event
logging, global access via WireGuard.

## Features

- ✅ Real-time 1080p 23fps streaming (WebSocket)
- ✅ YOLO object detection (person, dog, car)
- ✅ PTZ auto-tracking on C210
- ✅ Motion event logging with 1080p snapshots
- ✅ HTTP API (stats, events, frames, snapshots)
- ✅ systemd production deployment
- ✅ WireGuard global access (no port forwarding)
- ✅ Python API for frame access and motion hooks
- 🚧 Additional camera models

## Getting Started

### Prerequisites

- Python 3.10+
- ffmpeg
- TP-Link Tapo C210/C200 cameras with RTSP enabled

### Installation

```bash
git clone https://github.com/The-Great-One/tapo-pipeline.git
cd tapo-pipeline

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Enable RTSP on your cameras: Tapo app → Camera → ⚙️ → Advanced →
**Camera Account** (username + password).

## Usage

### Configure cameras

Edit `config.json`:

```json
{
  "cameras": {
    "C210": {"name": "Lift Cam", "rtsp": "rtsp://user:pass@C210_IP:554/stream1", "ptz": true},
    "C200": {"name": "Gate Cam", "rtsp": "rtsp://user:pass@C200_IP:554/stream1", "ptz": false}
  }
}
```

### Run

```bash
python pipeline.py --port 8090
```

Open `http://localhost:8090`. API endpoints: `/api/stats`, `/api/events`,
`/api/frame/C210`, `/api/stream`, `/api/snapshot/C210`.

### Auto-tracking (C210 PTZ)

YOLOv8 detects person/dog/car, tracker maps detection to motor angles, camera
pans/tilts to follow. Enable:

```bash
pip install ultralytics pytapo
python pipeline.py --port 8090
# → "Auto-tracker active — YOLO + PTZ tracking"
```

### Systemd (production)

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

### WireGuard global access

Run on a cloud server; cameras tunnel via WireGuard (no public IPs needed).

```
Cameras (LAN) → WireGuard → Oracle Cloud → pipeline → nginx → tapo.yourdomain.com
```

### Python API

```python
from pipeline import CameraPipeline

p = CameraPipeline()
p.start()
frame = p.get_frame_numpy("C210")   # shape: (360, 640, 3)
jpeg = p.get_full_res("C210")
p.on_motion(lambda event: print(f"Motion on {event.camera_id} at {event.timestamp}"))
p.tracker.manual_move("right", steps=3)
p.tracker.calibrate()
```

### Architecture

- **ffmpeg** decodes RTSP → raw RGB frames (640×360 @ 25fps)
- **Pillow** encodes JPEG in-memory
- **YOLOv8 nano** runs inference on motion events only
- **pytapo** sends PTZ commands via HTTPS to the C210
- **WebSocket** pushes frames to browsers at 30fps

## Roadmap

- [x] v0.1 – Streaming, motion detection, PTZ tracking
- [ ] v0.2 – Additional camera models
- [ ] v1.0 – Stable release

See the open issues for the full list of proposed features and known bugs.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) and
follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contact

Sahil Goel – [@The-Great-One](https://github.com/The-Great-One) –
sahilgoel187@gmail.com

Project link: [https://github.com/The-Great-One/tapo-pipeline](https://github.com/The-Great-One/tapo-pipeline)

## Acknowledgments

- [Img Shields](https://shields.io)
- [Contributor Covenant](https://www.contributor-covenant.org)
