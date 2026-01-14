import time
import threading
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, Response

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    # Optional fallback if you have full tensorflow installed (rare on Pi 2B)
    try:
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    except ImportError:
        raise SystemExit(
            "No TFLite Interpreter found.\n"
            "Install with:\n"
            "  python3 -m pip install tflite-runtime\n"
        )

# ----------------------------
# Config (tune for Pi 2B)
# ----------------------------
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

CAMERA_INDEX = 0

# Prefer forcing V4L2 to avoid some GStreamer weirdness/warnings
FORCE_V4L2 = True

CAP_WIDTH = 320
CAP_HEIGHT = 240

# Run inference only every N frames to reduce CPU
INFER_EVERY_N_FRAMES = 3

# Below this confidence -> "none"
CONF_THRESHOLD = 0.60

# Terminal printing behavior
PRINT_ON_CHANGE_ONLY = False      # set True if you only want prints when label changes
PRINT_INTERVAL_SEC = 1.0          # if not change-only, print at most once per this many seconds

# For FLOAT models only: try "0_1" first; if results seem stuck, try "-1_1"
# Options: "0_1", "-1_1", "none"
FLOAT_PREPROCESS = "none"

# Debug output
DEBUG_PRINT_TENSORS_ON_START = True
DEBUG_TOPK_EVERY = 10            # print top-3 every N inferences (0 disables)
DEBUG_FRAME_MEAN_EVERY = 20      # print frame mean every N inferences (0 disables)

# Web server
HOST = "0.0.0.0"
PORT = 5000


def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def topk(probs: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
    idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in idx]


def get_quant_params(details0) -> Tuple[float, int]:
    """
    Returns (scale, zero_point) for quantized tensors.
    Works with both older and newer TFLite metadata.
    """
    qp = details0.get("quantization_parameters", {})
    scales = qp.get("scales", None)
    zero_points = qp.get("zero_points", None)
    if scales is not None and len(scales) > 0:
        return float(scales[0]), int(zero_points[0])
    # fallback older field
    scale, zp = details0.get("quantization", (0.0, 0))
    return float(scale), int(zp)


@dataclass
class LatestResult:
    label: str = "none"
    conf: float = 0.0
    infer_fps: float = 0.0
    last_update_ts: float = 0.0
    top3: str = ""


latest = LatestResult()
latest_lock = threading.Lock()
stop_event = threading.Event()

app = Flask(__name__)


@app.get("/")
def index():
    with latest_lock:
        age = time.time() - latest.last_update_ts if latest.last_update_ts else 0.0
        html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta http-equiv="refresh" content="1">
<title>Pi Detector</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 18px; max-width: 620px; }}
    .label {{ font-size: 48px; font-weight: 700; }}
    .meta {{ margin-top: 10px; color: #444; font-size: 18px; }}
    .small {{ color: #666; font-size: 14px; margin-top: 10px; }}
    code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 6px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
</style>
</head>
<body>
<div class="card">
    <div class="label">{latest.label}</div>
    <div class="meta">Confidence: <b>{latest.conf:.2f}</b></div>
    <div class="meta">Infer FPS: <b>{latest.infer_fps:.2f}</b></div>
    <div class="small mono">Top3: {latest.top3}</div>
    <div class="small">Last update: {age:.1f}s ago</div>
    <div class="small">JSON: <code>/json</code></div>
</div>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


@app.get("/json")
def json_status():
    with latest_lock:
        payload = {
            "label": latest.label,
            "confidence": round(latest.conf, 4),
            "infer_fps": round(latest.infer_fps, 3),
            "last_update_ts": latest.last_update_ts,
            "top3": latest.top3,
        }
    import json
    return Response(json.dumps(payload), mimetype="application/json")


def inference_loop():
    labels = load_labels(LABELS_PATH)
    print(f"Loaded labels ({len(labels)}): {labels}")

    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if DEBUG_PRINT_TENSORS_ON_START:
        print("INPUT DETAILS:")
        for d in input_details:
            print(" ", {k: d.get(k) for k in ["name", "shape", "dtype", "quantization", "quantization_parameters", "index"]})
        print("OUTPUT DETAILS:")
        for d in output_details:
            print(" ", {k: d.get(k) for k in ["name", "shape", "dtype", "quantization", "quantization_parameters", "index"]})

    # Single-input expected
    in0 = input_details[0]
    in_idx = in0["index"]
    in_shape = in0["shape"]  # e.g. [1, H, W, 3]
    in_dtype = in0["dtype"]

    if len(in_shape) != 4:
        raise SystemExit(f"Unexpected input shape: {in_shape} (expected [1,H,W,C])")

    _, in_h, in_w, in_c = in_shape
    if in_c != 3:
        raise SystemExit(f"Expected 3 channels input, got {in_c}")

    print(f"Model input: {in_w}x{in_h} dtype={in_dtype}")
    if in_dtype in (np.uint8, np.int8):
        scale, zp = get_quant_params(in0)
        print(f"Input quantization: scale={scale} zero_point={zp}")
    else:
        print(f"Float preprocess mode: {FLOAT_PREPROCESS}")

    # Camera open
    if FORCE_V4L2:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise SystemExit("Could not open camera. Try CAMERA_INDEX=1 or check /dev/video*")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    frame_count = 0
    infer_count = 0

    last_print_label = None
    last_print_time = 0.0
    last_infer_time = time.time()

    print(f"Inference started. Open: http://{HOST}:{PORT}/  (Ctrl+C to stop)")

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            frame_count += 1
            if frame_count % INFER_EVERY_N_FRAMES != 0:
                continue

            infer_count += 1

            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to model input
            resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)

            # Debug: ensure frames change
            if DEBUG_FRAME_MEAN_EVERY and infer_count % DEBUG_FRAME_MEAN_EVERY == 0:
                print(f"DEBUG frame mean: {float(np.mean(resized)):.2f}")

            # Shape to [1,H,W,3]
            input_tensor = np.expand_dims(resized, axis=0)

            # Feed tensor according to dtype
            if in_dtype in (np.uint8, np.int8):
                scale, zp = get_quant_params(in0)
                if in_dtype == np.uint8:
                    # Most uint8 image models accept raw 0..255 uint8
                    q = input_tensor.astype(np.uint8)
                else:
                    # int8: map 0..255 into int8 using scale/zp
                    # Assume "real" domain was 0..255 (common for CV image models)
                    if scale > 0:
                        q = (input_tensor.astype(np.float32) / scale + zp)
                        q = np.clip(np.round(q), -128, 127).astype(np.int8)
                    else:
                        q = input_tensor.astype(np.int8)

                interpreter.set_tensor(in_idx, q)

            else:
                x = input_tensor.astype(np.float32)
                if FLOAT_PREPROCESS == "0_1":
                    x = x / 255.0
                elif FLOAT_PREPROCESS == "-1_1":
                    x = (x / 127.5) - 1.0
                elif FLOAT_PREPROCESS == "none":
                    pass
                else:
                    raise SystemExit(f"Unknown FLOAT_PREPROCESS={FLOAT_PREPROCESS}")

                interpreter.set_tensor(in_idx, x)

            interpreter.invoke()

            # IMPORTANT: classification models usually have 1 output.
            # If you have more, you're probably using an object detection model export.
            out0 = output_details[0]
            out_idx = out0["index"]
            output = interpreter.get_tensor(out_idx)[0].astype(np.float32)

            # Determine if output is probs or logits
            if not (0.95 <= float(np.sum(output)) <= 1.05) and np.max(output) > 1.0:
                probs = softmax(output)
            else:
                probs = output

            best_i = int(np.argmax(probs))
            conf = float(probs[best_i])
            best_label = labels[best_i] if best_i < len(labels) else str(best_i)

            # Top-3 string for web/debug
            t3 = topk(probs, k=min(3, len(probs)))
            top3_str = ", ".join([f"{labels[i] if i < len(labels) else i}:{p:.2f}" for i, p in t3])

            label = best_label
            if conf < CONF_THRESHOLD:
                label = "none"

            now = time.time()
            infer_fps = 1.0 / max(now - last_infer_time, 1e-6)
            last_infer_time = now

            with latest_lock:
                latest.label = label
                latest.conf = conf
                latest.infer_fps = infer_fps
                latest.last_update_ts = now
                latest.top3 = top3_str

            # Debug top-3
            if DEBUG_TOPK_EVERY and infer_count % DEBUG_TOPK_EVERY == 0:
                print(f"DEBUG top3: {top3_str}")

            # Terminal output
            should_print = True
            if PRINT_ON_CHANGE_ONLY:
                should_print = (label != last_print_label)
            else:
                should_print = (now - last_print_time) >= PRINT_INTERVAL_SEC

            if should_print:
                print(
                    f"{time.strftime('%H:%M:%S')}  Detected: {label:>5}  "
                    f"conf={conf:.2f}  fps≈{infer_fps:.2f}  top3=[{top3_str}]"
                )
                last_print_label = label
                last_print_time = now

    finally:
        cap.release()


def main():
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nStopping...")