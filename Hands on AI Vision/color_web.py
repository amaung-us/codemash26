#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time
import threading
from flask import Flask, jsonify, Response

app = Flask(__name__)

# Shared state (updated by camera thread, read by web thread)
STATE = {
    "label": "STARTING",
    "green_ratio": 0.0,
    "white_ratio": 0.0,
    "timestamp": time.time(),
}
STATE_LOCK = threading.Lock()


def classify_green_or_white(bgr_roi: np.ndarray,
                            green_min_ratio: float = 0.20,
                            white_min_ratio: float = 0.60):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    # GREEN
    green_lower = np.array([35, 50, 50], dtype=np.uint8)
    green_upper = np.array([85, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # WHITE = low sat + high value
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([179, 35, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)

    total = bgr_roi.shape[0] * bgr_roi.shape[1]
    green_ratio = float(np.count_nonzero(green_mask)) / total
    white_ratio = float(np.count_nonzero(white_mask)) / total

    if white_ratio >= white_min_ratio and green_ratio < green_min_ratio:
        return "WHITE", green_ratio, white_ratio
    if green_ratio >= green_min_ratio and white_ratio < white_min_ratio:
        return "GREEN", green_ratio, white_ratio

    if green_ratio > white_ratio * 1.2 and green_ratio >= 0.10:
        return "GREEN", green_ratio, white_ratio
    if white_ratio > green_ratio * 1.2 and white_ratio >= 0.10:
        return "WHITE", green_ratio, white_ratio

    return "UNKNOWN", green_ratio, white_ratio


def camera_loop(args):
    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        with STATE_LOCK:
            STATE["label"] = f"ERROR: cannot open camera device {args.device}"
            STATE["timestamp"] = time.time()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    interval = 1.0 / max(args.fps, 0.1)
    next_time = time.time()

    while True:
        now = time.time()
        if now < next_time:
            time.sleep(min(0.02, next_time - now))
            continue
        next_time = now + interval

        ok, frame = cap.read()
        if not ok or frame is None:
            with STATE_LOCK:
                STATE["label"] = "ERROR: frame read failed"
                STATE["timestamp"] = time.time()
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        roi_frac = float(np.clip(args.roi, 0.1, 1.0))
        rw, rh = int(w * roi_frac), int(h * roi_frac)
        x0 = (w - rw) // 2
        y0 = (h - rh) // 2
        roi = frame[y0:y0 + rh, x0:x0 + rw]

        ds = max(1, int(args.downsample))
        if ds > 1:
            roi = cv2.resize(
                roi,
                (max(1, roi.shape[1] // ds), max(1, roi.shape[0] // ds)),
                interpolation=cv2.INTER_NEAREST
            )

        label, green_ratio, white_ratio = classify_green_or_white(roi)

        with STATE_LOCK:
            STATE["label"] = label
            STATE["green_ratio"] = green_ratio
            STATE["white_ratio"] = white_ratio
            STATE["timestamp"] = time.time()


@app.get("/status")
def status():
    with STATE_LOCK:
        payload = dict(STATE)
    payload["age_sec"] = round(time.time() - payload["timestamp"], 3)
    return jsonify(payload)


@app.get("/")
def index():
    # Single lightweight HTML page that polls /status
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Pi Color Status</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;margin:24px}
    .card{max-width:520px;padding:18px;border:1px solid #ddd;border-radius:14px}
    .label{font-size:44px;font-weight:800;letter-spacing:1px}
    .meta{margin-top:12px;color:#444;font-size:15px;line-height:1.6}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#f2f2f2;margin-top:10px}
    .stale{background:#ffe6e6}
  </style>
</head>
<body>
  <div class="card">
    <div id="label" class="label">STARTING</div>
    <div id="pill" class="pill">Loading…</div>
    <div class="meta">
      green_ratio: <span id="g">0.000</span><br/>
      white_ratio: <span id="w">0.000</span><br/>
      age_sec: <span id="age">0.0</span>
    </div>
  </div>

<script>
async function tick(){
  try{
    const r = await fetch('/status', {cache:'no-store'});
    const j = await r.json();

    document.getElementById('label').textContent = j.label;
    document.getElementById('g').textContent = Number(j.green_ratio).toFixed(3);
    document.getElementById('w').textContent = Number(j.white_ratio).toFixed(3);
    document.getElementById('age').textContent = Number(j.age_sec).toFixed(2);

    const pill = document.getElementById('pill');
    pill.textContent = (j.age_sec > 3.0) ? "STALE (camera not updating?)" : "LIVE";
    pill.className = (j.age_sec > 3.0) ? "pill stale" : "pill";
  }catch(e){
    document.getElementById('pill').textContent = "ERROR fetching /status";
    document.getElementById('pill').className = "pill stale";
  }
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--roi", type=float, default=0.50)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = accessible on LAN")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    t = threading.Thread(target=camera_loop, args=(args,), daemon=True)
    t.start()

    # threaded=True lets Flask serve / and /status while camera thread runs
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
