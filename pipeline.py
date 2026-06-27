"""
Tapo Camera Pipeline — ML-Ready Frame Capture
=============================================
Full-res frame capture with motion detection, timestamped snapshots,
and a clean Python API for face recognition / alerting on top.

Architecture:
  Camera RTSP → ffmpeg (1080p raw) → MotionDetector → FrameStore
                                                      ↓
                                              Python API / WebSocket

Usage (import into your ML project):
  from pipeline import CameraPipeline, start_pipeline

  pipeline = start_pipeline()
  frame = pipeline.get_latest("C210")     # numpy array (1080p RGB)
  events = pipeline.get_recent_events()    # motion events with frames
  pipeline.on_motion = my_callback         # real-time callbacks

Or run standalone:
  python3 pipeline.py --serve             # HTTP + WebSocket server
"""

import subprocess
import threading
import time
import json
import os
import io
import hashlib
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

# Fast in-memory JPEG encoding (no subprocess overhead)
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ─── Configuration ────────────────────────────────────────────────

CAMERAS = {
    "C210": {
        "ip": "YOUR_C210_IP",
        "rtsp": "rtsp://youruser:yourpass@YOUR_C210_IP:554/stream1",
        "name": "C210 (PTZ)",
    },
    "C200": {
        "ip": "YOUR_C200_IP",
        "rtsp": "rtsp://youruser:yourpass@YOUR_C200_IP:554/stream1",
        "name": "C200",
    },
}

DATA_DIR = Path(os.environ.get("TAPO_DATA_DIR", Path.home() / "tapo-data"))
FRAME_DIR = DATA_DIR / "frames"        # timestamped frame snapshots
EVENT_DIR = DATA_DIR / "events"        # motion-triggered event frames
LOG_FILE = DATA_DIR / "events.jsonl"   # append-only event log

HTTP_PORT = int(os.environ.get("TAPO_PORT", "8090"))
PREVIEW_RES = (1920, 1080)            # full res — Oracle has 23GB RAM
FULL_RES = (1920, 1080)                # native camera resolution (on-demand)
JPEG_QUALITY = 90


# ─── Frame Grabber (Low-Res Continuous + 1080p On-Demand) ─────────

class FrameGrabber:
    """
    Continuous stream at 640x360 for motion detection + live view.
    Captures full 1080p snapshots only on motion triggers.
    Keeps the Pi stable with ~1GB RAM.
    """

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.latest_frame: Optional[bytes] = None  # 640x360 RGB24
        self.latest_jpeg: Optional[bytes] = None
        self.frame_count = 0
        self.fps = 0.0
        self.errors: deque = deque(maxlen=10)
        self._running = False
        self._process: Optional[subprocess.Popen] = None
        self._start_time = 0.0
        self._frame_size = PREVIEW_RES[0] * PREVIEW_RES[1] * 3  # 640*360*3 = 691,200

    def start(self):
        self._running = True
        self._start_time = time.time()
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-rtsp_transport", "tcp",
                "-i", self.rtsp_url,
                "-an",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-vf", f"fps=25,scale={PREVIEW_RES[0]}:{PREVIEW_RES[1]}",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while self._running and self._process:
            try:
                raw = self._process.stdout.read(self._frame_size)
                if len(raw) < self._frame_size:
                    break
                self.latest_frame = raw
                self.frame_count += 1
            except Exception as e:
                self.errors.append(str(e))
                break

        if self._running:
            self.errors.append("ffmpeg exited, restarting in 3s...")
            time.sleep(3)
            self.start()

    def capture_full_res(self) -> Optional[bytes]:
        """Pull a single 1080p frame on demand (for motion events). Returns JPEG bytes."""
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-rtsp_transport", "tcp",
                    "-i", self.rtsp_url,
                    "-an",
                    "-vframes", "1",               # single frame
                    "-vf", f"scale={FULL_RES[0]}:{FULL_RES[1]}",
                    "-f", "mjpeg",
                    "-q:v", "3",
                    "-",
                ],
                capture_output=True,
                timeout=8,
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
        except Exception as e:
            self.errors.append(f"full-res capture: {e}")
        return None

    def get_jpeg(self):
        if self.latest_frame is None:
            return None
        try:
            if HAS_PIL:
                img = Image.frombuffer("RGB", PREVIEW_RES, self.latest_frame, "raw", "RGB", 0, 1)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=85)
                self.latest_jpeg = buf.getvalue()
                return self.latest_jpeg
            else:
                proc = subprocess.run(
                    ["ffmpeg", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{PREVIEW_RES[0]}x{PREVIEW_RES[1]}",
                     "-i", "-", "-f", "mjpeg", "-q:v", "3", "-"],
                    input=self.latest_frame, capture_output=True, timeout=3,
                )
                if proc.returncode == 0:
                    self.latest_jpeg = proc.stdout
                    return proc.stdout
        except Exception:
            pass
        return None

    def stats(self) -> dict:
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "camera": self.camera_id,
            "frames": self.frame_count,
            "fps": round(self.frame_count / uptime, 1) if uptime > 0 else 0,
            "uptime": round(uptime, 0),
            "errors": list(self.errors)[-3:],
            "resolution": f"{PREVIEW_RES[0]}x{PREVIEW_RES[1]}",
        }

    def stop(self):
        self._running = False
        if self._process:
            self._process.terminate()


# ─── Motion Detector ──────────────────────────────────────────────

@dataclass
class MotionEvent:
    """A detected motion event with the triggering frame."""
    camera_id: str
    timestamp: float
    frame_rgb: bytes          # raw RGB24 at full res
    frame_jpeg: bytes         # JPEG for storage
    motion_level: float       # 0.0 - 1.0
    frame_path: str = ""      # path to saved JPEG
    event_id: str = ""

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "camera": self.camera_id,
            "timestamp": self.timestamp,
            "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
            "motion_level": round(self.motion_level, 3),
            "frame_path": self.frame_path,
        }


class MotionDetector:
    """
    Simple frame-differencing motion detector.
    Compares consecutive frames, triggers when change exceeds threshold.
    Replace this with OpenCV MOG2 or your own detector later.
    """

    def __init__(self, threshold: float = 0.03, cooldown: float = 1.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self._prev_frame: dict[str, bytes] = {}
        self._last_trigger: dict[str, float] = {}
        self._callbacks: list[Callable] = []

    def on_motion(self, callback: Callable):
        """Register a callback: callback(MotionEvent)"""
        self._callbacks.append(callback)

    def check(self, camera_id: str, frame: bytes) -> Optional[MotionEvent]:
        """Check if a frame shows motion. Returns MotionEvent if triggered."""
        now = time.time()

        # Cooldown
        if camera_id in self._last_trigger:
            if now - self._last_trigger[camera_id] < self.cooldown:
                return None

        if camera_id not in self._prev_frame or self._prev_frame[camera_id] is None:
            self._prev_frame[camera_id] = frame
            return None

        # Simple pixel difference (subsampled for speed)
        prev = self._prev_frame[camera_id]
        self._prev_frame[camera_id] = frame

        diff = self._frame_diff(prev, frame)
        if diff < self.threshold:
            return None

        self._last_trigger[camera_id] = now

        # Convert to JPEG for storage
        jpeg = self._rgb_to_jpeg(frame)

        event = MotionEvent(
            camera_id=camera_id,
            timestamp=now,
            frame_rgb=frame,
            frame_jpeg=jpeg,
            motion_level=diff,
        )

        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass

        return event

    def _frame_diff(self, a: bytes, b: bytes) -> float:
        """Fast subsampled frame difference (0.0 = identical, 1.0 = completely different)."""
        # Compare every 100th pixel for speed
        sample_step = 300  # every 100th RGB pixel
        diff_count = 0
        total = 0
        for i in range(0, min(len(a), len(b)), sample_step):
            total += 1
            if a[i] != b[i]:
                diff_count += 1
        return diff_count / max(total, 1)

    def _rgb_to_jpeg(self, rgb: bytes) -> bytes:
        try:
            proc = subprocess.run(
                ["ffmpeg", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{FULL_RES[0]}x{FULL_RES[1]}",
                 "-i", "-", "-f", "mjpeg", "-q:v", "3", "-"],
                input=rgb, capture_output=True, timeout=3,
            )
            return proc.stdout if proc.returncode == 0 else b""
        except Exception:
            return b""


# ─── Camera Pipeline (Main API) ───────────────────────────────────

class CameraPipeline:
    """
    Main entry point for ML projects.

    Usage:
        pipeline = CameraPipeline()
        pipeline.start()

        # Get latest frame from a camera
        frame = pipeline.get_latest("C210")  # bytes (1920*1080*3 RGB24)

        # Get as numpy array (if opencv/numpy available)
        # import numpy as np
        # arr = pipeline.get_frame_numpy("C210")

        # React to motion
        def on_person_detected(event):
            # Your face recognition logic here
            # event.frame_jpeg is a JPEG you can feed to your model
            pass

        pipeline.on_motion(on_person_detected)

        # Get recent events
        events = pipeline.get_recent_events(limit=50)
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "frames").mkdir(exist_ok=True)
        (self.data_dir / "events").mkdir(exist_ok=True)

        self.grabbers: dict[str, FrameGrabber] = {}
        self.motion = MotionDetector(threshold=0.03, cooldown=1.0)
        self.events: deque[MotionEvent] = deque(maxlen=1000)
        self._running = False
        self._motion_callbacks: list[Callable] = []
        self._log_file = self.data_dir / "events.jsonl"

    def on_motion(self, callback: Callable):
        """Register a callback for motion events. callback(MotionEvent)."""
        self._motion_callbacks.append(callback)

    def start(self):
        """Start all camera streams."""
        self._running = True
        for cam_id, cfg in CAMERAS.items():
            g = FrameGrabber(cam_id, cfg["rtsp"])
            g.start()
            self.grabbers[cam_id] = g
            print(f"  [{cam_id}] Started — {cfg['name']}")

        # Start motion detection loop
        threading.Thread(target=self._motion_loop, daemon=True).start()
        print(f"  Motion detection active (threshold={self.motion.threshold})")

        # Start auto-tracker if requested
        self.tracker = None
        try:
            from tracker import AutoTracker
            self.tracker = AutoTracker("YOUR_C210_IP")
            self.on_motion(lambda e: self.tracker.track(e))
            print("  Auto-tracker active — YOLO + PTZ tracking")
        except ImportError:
            pass

    def _motion_loop(self):
        """Continuously check frames for motion. Captures 1080p on trigger."""
        while self._running:
            for cam_id, g in self.grabbers.items():
                frame = g.latest_frame
                if frame is None:
                    continue

                event = self.motion.check(cam_id, frame)
                if event is None:
                    continue

                # Capture full 1080p frame for the event
                full_jpeg = g.capture_full_res()
                if full_jpeg:
                    event.frame_jpeg = full_jpeg
                # else: use the low-res jpeg as fallback

                # Save event frame to disk
                ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.timestamp))
                event.event_id = f"{cam_id}_{ts}_{hashlib.md5(event.frame_jpeg[:100]).hexdigest()[:6]}"
                event.frame_path = str(
                    self.data_dir / "events" / f"{event.event_id}.jpg"
                )

                with open(event.frame_path, "wb") as f:
                    f.write(event.frame_jpeg)

                self.events.append(event)

                # Write to event log (JSONL — easy to ingest)
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(event.to_dict()) + "\n")

                # Fire callbacks
                for cb in self._motion_callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        print(f"  Motion callback error: {e}")

            time.sleep(0.1)  # ~10 checks/sec

    def get_latest(self, camera_id: str) -> Optional[bytes]:
        """Get latest raw RGB24 frame (640x360)."""
        g = self.grabbers.get(camera_id)
        return g.latest_frame if g else None

    def get_frame_numpy(self, camera_id: str):
        """Get latest frame as numpy array (requires numpy). 640x360 RGB."""
        frame = self.get_latest(camera_id)
        if frame is None:
            return None
        import numpy as np
        return np.frombuffer(frame, dtype=np.uint8).reshape(PREVIEW_RES[1], PREVIEW_RES[0], 3)

    def get_full_res(self, camera_id: str) -> Optional[bytes]:
        """Capture a full 1080p JPEG frame on demand."""
        g = self.grabbers.get(camera_id)
        return g.capture_full_res() if g else None

    def get_jpeg(self, camera_id: str) -> Optional[bytes]:
        """Get latest frame as JPEG."""
        g = self.grabbers.get(camera_id)
        return g.get_jpeg() if g else None

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Get recent motion events with frame paths."""
        return [e.to_dict() for e in list(self.events)[-limit:]]

    def get_stats(self) -> dict:
        """Get stats for all cameras."""
        stats = {cid: g.stats() for cid, g in self.grabbers.items()}
        if getattr(self, "tracker", None):
            stats["tracker"] = self.tracker.stats()
        return stats

    def snapshot(self, camera_id: str) -> Optional[str]:
        """Take an immediate snapshot and return the file path."""
        jpeg = self.get_jpeg(camera_id)
        if not jpeg:
            return None
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = str(self.data_dir / "frames" / f"{camera_id}_{ts}.jpg")
        with open(path, "wb") as f:
            f.write(jpeg)
        return path

    def stop(self):
        self._running = False
        for g in self.grabbers.values():
            g.stop()


# ─── HTTP + WebSocket Server ──────────────────────────────────────

class PipelineHTTPHandler(BaseHTTPRequestHandler):
    pipeline: Optional[CameraPipeline] = None

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/":
            self._serve_html()
        elif path == "/api/stats":
            self._json(self.pipeline.get_stats())
        elif path == "/api/events":
            limit = 50
            if "?limit=" in self.path:
                try:
                    limit = int(self.path.split("limit=")[1].split("&")[0])
                except:
                    pass
            self._json(self.pipeline.get_recent_events(limit))
        elif path.startswith("/api/snapshot/"):
            cam = path.split("/api/snapshot/")[1]
            path_saved = self.pipeline.snapshot(cam)
            self._json({"snapshot": path_saved})
        elif path.startswith("/api/frame/"):
            cam = path.split("/api/frame/")[1]
            jpeg = self.pipeline.get_jpeg(cam)
            if jpeg:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            self.send_error(404)
        elif path == "/api/stream":
            self._mjpeg_stream()
        else:
            self.send_error(404)

    def _json(self, data):
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        while self.pipeline._running:
            for cam_id in self.pipeline.grabbers:
                jpeg = self.pipeline.get_jpeg(cam_id)
                if jpeg:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode() +
                        b"\r\n\r\n" + jpeg + b"\r\n"
                    )
            time.sleep(0.05)

    def _serve_html(self):
        cams_html = "".join(
            '<div class="cam"><h2>' + cid + '</h2>'
            '<img src="/api/frame/' + cid + '" id="img_' + cid + '">'
            '<div class="stat" id="st_' + cid + '">loading...</div></div>'
            for cid in self.pipeline.grabbers
        )
        html = '''<!DOCTYPE html>
<html><head><title>Tapo Pipeline</title>
<style>
body{background:#0a0a0a;color:#e0e0e0;font-family:monospace;margin:20px}
h1{color:#00ff88} .cam{border:1px solid #333;padding:15px;margin:10px 0;border-radius:8px}
img{max-width:100%;border-radius:4px} .stat{color:#888;font-size:12px;margin-top:5px}
.endpoints{background:#111;padding:10px;margin-top:20px;border-radius:4px;font-size:13px}
a{color:#00ff88}
</style></head><body><h1>Tapo Camera Pipeline</h1>
<div id="events" class="stat">Events: 0</div>
''' + cams_html + '''
<div class="endpoints">
<h3>API</h3><pre>
GET /api/stats           JSON: fps, frames, uptime
GET /api/events          JSON: recent motion events
GET /api/frame/C210      Latest JPEG frame
GET /api/stream          MJPEG live stream
</pre>
<h3>Python API</h3><pre>
from pipeline import CameraPipeline
p = CameraPipeline()
p.start()
frame = p.get_frame_numpy("C210")  # numpy array
p.on_motion(lambda e: print("Motion on", e.camera_id))
</pre></div>
<script>
var ws=new WebSocket("ws://"+location.host+"/ws");
ws.binaryType="blob";
var camIds=[],idx=0;
fetch("/api/stats").then(function(r){return r.json()}).then(function(d){camIds=Object.keys(d)});
ws.onmessage=function(e){
 if(camIds.length===0)return;
 var cid=camIds[idx%camIds.length];idx++;
 var u=URL.createObjectURL(e.data);
 var img=document.getElementById("img_"+cid);
 if(img){var o=img.src;img.src=u;if(o&&o.startsWith("blob:"))URL.revokeObjectURL(o)}
};
setInterval(async function(){
 var r=await fetch("/api/stats");var d=await r.json();
 for(var c in d){
  var el=document.getElementById("st_"+c);
  if(el)el.textContent=d[c].frames+" fps | "+d[c].fps+" fps | "+d[c].resolution
 }
 var ev=await fetch("/api/events?limit=1");
 if(ev.ok){var events=await ev.json();document.getElementById("events").textContent="Events: "+events.length}
},1000);
</script></body></html>'''
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_http_server(pipeline: CameraPipeline, port: int = HTTP_PORT):
    PipelineHTTPHandler.pipeline = pipeline
    server = HTTPServer(("0.0.0.0", port), PipelineHTTPHandler)
    print(f"\n  HTTP server: http://0.0.0.0:{port}")
    print(f"  Data dir:    {pipeline.data_dir}")
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # Start WebSocket server for real-time frame push
    threading.Thread(target=_run_ws_server, args=(pipeline, port + 1), daemon=True).start()
    print(f"  WebSocket:   ws://0.0.0.0:{port + 1}")
    return server


# ─── Minimal WebSocket Server (stdlib asyncio) ────────────────────

def _run_ws_server(pipeline: CameraPipeline, port: int):
    import asyncio, hashlib, base64, struct

    WS_MAGIC = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
    clients: set[asyncio.StreamWriter] = set()
    lock = asyncio.Lock()

    async def ws_handler(reader, writer):
        # Read HTTP upgrade request
        data = await reader.readuntil(b"\r\n\r\n")
        headers = data.decode()
        key = ""
        for line in headers.split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                key = line.split(":", 1)[1].strip()

        if not key:
            writer.close()
            return

        # Send upgrade response
        accept = base64.b64encode(hashlib.sha1(key.encode() + WS_MAGIC).digest()).decode()
        resp = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        writer.write(resp.encode())
        await writer.drain()

        async with lock:
            clients.add(writer)

        try:
            while pipeline._running:
                try:
                    msg = await asyncio.wait_for(reader.read(1024), timeout=1.0)
                    if not msg:
                        break
                except asyncio.TimeoutError:
                    pass
        finally:
            async with lock:
                clients.discard(writer)
            writer.close()

    async def broadcast_loop():
        """Push frames to all WebSocket clients at native speed."""
        nonlocal clients
        print("[WS] Broadcast loop started", flush=True)
        sent = 0
        while pipeline._running:
            try:
                for cam_id in list(pipeline.grabbers.keys()):
                    jpeg = pipeline.get_jpeg(cam_id)
                    if jpeg is None:
                        continue

                    # WebSocket binary frame: fin=1, opcode=2 (binary), masked=0
                    frame = b"\x82" + _ws_frame_header(len(jpeg)) + jpeg

                    async with lock:
                        dead = set()
                        for w in clients:
                            try:
                                w.write(frame)
                            except Exception:
                                dead.add(w)
                        clients -= dead

                    if clients:
                        for w in clients:
                            try:
                                await w.drain()
                            except Exception:
                                pass

                sent += 1
                if sent % 30 == 0:
                    print(f"[WS] Sent {sent} batches, {len(clients)} clients", flush=True)
            except Exception as ex:
                print(f"[WS] Error in broadcast: {ex}", flush=True)
            await asyncio.sleep(1 / 30)  # ~30fps push rate

    def _ws_frame_header(length: int) -> bytes:
        if length < 126:
            return bytes([length])
        elif length < 65536:
            return struct.pack("!BH", 126, length)
        else:
            return struct.pack("!BQ", 127, length)

    async def main():
        server = await asyncio.start_server(ws_handler, "0.0.0.0", port)
        asyncio.create_task(broadcast_loop())
        await server.serve_forever()

    asyncio.run(main())


# ─── Quick start ──────────────────────────────────────────────────

def start_pipeline(serve: bool = True) -> CameraPipeline:
    """One-call setup. Returns the pipeline for use in your ML code."""
    pipeline = CameraPipeline()
    pipeline.start()
    if serve:
        start_http_server(pipeline)
    return pipeline


# ─── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-serve", action="store_true", help="Don't start HTTP server")
    ap.add_argument("--port", type=int, default=HTTP_PORT)
    ap.add_argument("--motion-threshold", type=float, default=0.03)
    ap.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    args = ap.parse_args()

    pipeline = CameraPipeline(Path(args.data_dir))
    pipeline.motion.threshold = args.motion_threshold
    pipeline.start()

    print(f"\n  Motion events logged to: {pipeline._log_file}")
    print(f"  Event frames saved to:   {pipeline.data_dir / 'events'}")

    if args.no_serve:
        print("\n  Pipeline running. Press Ctrl+C to stop.")
        print("  Use in Python:")
        print("    from pipeline import CameraPipeline")
        print("    p = CameraPipeline()")
        print("    p.start()")
        print("    frame = p.get_frame_numpy('C210')")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        start_http_server(pipeline, args.port)
        print("\n  Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    pipeline.stop()
    print("Stopped.")
