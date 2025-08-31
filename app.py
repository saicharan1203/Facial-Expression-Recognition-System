import os
import io
import time
import base64
import logging
import traceback
from collections import deque, defaultdict
from typing import Dict, Any, Deque, Tuple, List

import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
try:
    from deepface import DeepFace
except Exception:
    DeepFace = None  # type: ignore
try:
    import mediapipe as mp
except Exception:
    mp = None  # type: ignore


ROLLING_WINDOW_SIZE: int = 30
_rolling_emotions: Deque[Dict[str, float]] = deque(maxlen=ROLLING_WINDOW_SIZE)
# Per-session storage
_session_rolling: Dict[str, Deque[Dict[str, float]]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW_SIZE))
_session_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
_req_times: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=100))
RATE_WINDOW_SEC = 3.0
RATE_MAX_REQUESTS = 15  # per-IP per window
MODEL_READY = False


def _compute_rolling_average() -> Dict[str, float]:
    if not _rolling_emotions:
        return {}
    keys = set().union(*_rolling_emotions)
    totals: Dict[str, float] = {k: 0.0 for k in keys}
    for item in _rolling_emotions:
        for k in keys:
            totals[k] += float(item.get(k, 0.0))
    n = float(len(_rolling_emotions))
    return {k: (totals[k] / n) for k in keys}


def _compute_rolling_average_session(session_id: str) -> Dict[str, float]:
    dq = _session_rolling.get(session_id)
    if not dq:
        return {}
    keys = set().union(*dq)
    totals: Dict[str, float] = {k: 0.0 for k in keys}
    for item in dq:
        for k in keys:
            totals[k] += float(item.get(k, 0.0))
    n = float(len(dq))
    return {k: (totals[k] / n) for k in keys}


def _annotate_frame(bgr_frame: np.ndarray, dominant: str, emotions: Dict[str, float], faces: List[Dict[str, Any]], blur_faces: bool = False) -> np.ndarray:
    h, w = bgr_frame.shape[:2]
    overlay = bgr_frame.copy()
    # Header bar
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), thickness=-1)
    text = f"{dominant.upper() if dominant else 'UNKNOWN'}"
    cv2.putText(overlay, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA)
    # List top-3 emotions
    sorted_items = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:3]
    y = 90
    for k, v in sorted_items:
        cv2.putText(overlay, f"{k}: {v:.1f}%", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
    # Draw face boxes
    try:
        for f in faces or []:
            fa = f.get('facial_area', {})
            x, y1, w1, h1 = int(fa.get('x', 0)), int(fa.get('y', 0)), int(fa.get('w', 0)), int(fa.get('h', 0))
            if w1 > 0 and h1 > 0:
                if blur_faces:
                    roi = overlay[y1:y1+h1, x:x+w1]
                    if roi.size:
                        k = max(5, (w1 // 15) | 1)
                        roi = cv2.GaussianBlur(roi, (k, k), 0)
                        overlay[y1:y1+h1, x:x+w1] = roi
                else:
                    cv2.rectangle(overlay, (x, y1), (x + w1, y1 + h1), (0, 255, 0), 2)
    except Exception:
        pass
    return overlay


def _to_plain_floats(d: Dict[str, Any]) -> Dict[str, float]:
    plain: Dict[str, float] = {}
    for k, v in d.items():
        try:
            plain[k] = float(v)
        except Exception:
            # skip non-numerics
            pass
    return plain


def _mp_detect_faces_bboxes(bgr_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Use MediaPipe FaceDetection for robust presence check
    if mp is None:
        return []
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    bboxes: List[Tuple[int, int, int, int]] = []
    try:
        with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as fd:
            res = fd.process(rgb)
            if res.detections:
                h, w = bgr_frame.shape[:2]
                for det in res.detections:
                    box = det.location_data.relative_bounding_box
                    x = max(0, int(box.xmin * w))
                    y = max(0, int(box.ymin * h))
                    ww = max(0, int(box.width * w))
                    hh = max(0, int(box.height * h))
                    if ww >= 20 and hh >= 20:
                        bboxes.append((x, y, ww, hh))
    except Exception:
        pass
    return bboxes


def _cv2_haar_bboxes(bgr_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    # Lightweight OpenCV Haar cascade fallback when MediaPipe / DeepFace are unavailable
    try:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        clf = cv2.CascadeClassifier(cascade_path)
        faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
    except Exception:
        return []


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)
    app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger("vision-sense")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health() -> Any:
        return jsonify({"ok": True, "model_ready": MODEL_READY})

    def allow_request(ip: str) -> bool:
        now = time.time()
        dq = _req_times[ip]
        while dq and (now - dq[0] > RATE_WINDOW_SEC):
            dq.popleft()
        if len(dq) >= RATE_MAX_REQUESTS:
            return False
        dq.append(now)
        return True

    @app.route("/api/analyze", methods=["POST"])
    def analyze() -> Any:
        try:
            ip = request.remote_addr or "unknown"
            if not allow_request(ip):
                return jsonify({"error": "rate_limited"}), 429

            data: Dict[str, Any] = request.get_json(force=True)
            image_b64: str = data.get("image", "")
            session_id: str = str(data.get("session_id") or "default")
            mode: str = str(data.get("mode") or "default")
            max_side_cfg = int(data.get("max_side", 640))
            jpeg_quality = int(data.get("jpeg_quality", 80))
            jpeg_quality = 1 if jpeg_quality < 1 else (95 if jpeg_quality > 95 else jpeg_quality)
            blur_faces = bool(data.get("blur_faces", False))
            if not image_b64:
                return jsonify({"error": "missing image"}), 400

            # image is expected as dataURL (e.g., data:image/jpeg;base64,XXXX)
            if "," in image_b64:
                image_b64 = image_b64.split(",", 1)[1]

            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Validate dimensions
            h, w = frame.shape[:2]
            if h < 10 or w < 10:
                return jsonify({"error": "image_too_small"}), 400

            # Downscale on server to stabilize runtime
            max_side = max_side_cfg
            scale = min(1.0, float(max_side) / float(max(h, w)))
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            # Face detection for presence: try MediaPipe, DeepFace, then OpenCV Haar
            mp_boxes = _mp_detect_faces_bboxes(frame)
            faces = []
            if DeepFace is not None:
                try:
                    faces = DeepFace.extract_faces(frame, enforce_detection=False)
                except Exception:
                    faces = []
            haar_boxes = [] if (mp is not None and (mp_boxes or faces)) else _cv2_haar_bboxes(frame)

            # Determine presence using any detector
            person_present = bool(mp_boxes) or bool(faces) or bool(haar_boxes)

            if not person_present:
                emotions: Dict[str, float] = {}
                dominant_emotion: str = "no person detected"
                # present empty rolling stats to avoid implying a detection
                rolling_avg: Dict[str, float] = {}
                rolling_avg_session: Dict[str, float] = {}
                annotated = _annotate_frame(frame, dominant_emotion, emotions, faces, blur_faces=blur_faces)
            else:
                # Run emotion analysis only if DeepFace is available; otherwise provide unknown
                emotions: Dict[str, float] = {}
                dominant_emotion: str = "unknown"
                if DeepFace is not None:
                    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                    payload = result[0] if isinstance(result, list) else result
                    emotions_raw: Dict[str, Any] = payload.get("emotion", {})
                    emotions = _to_plain_floats(emotions_raw)
                    dominant_emotion = payload.get("dominant_emotion") or (
                        max(emotions, key=emotions.get) if emotions else "unknown"
                    )

                # Track rolling window only when faces detected
                _rolling_emotions.append(emotions)
                _session_rolling[session_id].append(emotions)
                rolling_avg = _compute_rolling_average()
                rolling_avg_session = _compute_rolling_average_session(session_id)

                annotated = _annotate_frame(frame, dominant_emotion, emotions, faces, blur_faces=blur_faces)
            _, buf = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            annotated_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

            # Record for session reports
            _session_records[session_id].append({
                "ts": time.time(),
                "mode": mode,
                "dominant": dominant_emotion,
                "emotions": emotions,
                "face_count": len(faces),
            })

            resp = {
                "dominant_emotion": dominant_emotion,
                "emotions": _to_plain_floats(emotions),
                "rolling_average": _to_plain_floats(rolling_avg),
                "rolling_average_session": _to_plain_floats(rolling_avg_session),
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                "face_count": len(faces),
                "mode": mode,
                "person_detected": bool(len(faces) > 0),
            }
            return jsonify(resp)
        except Exception as exc:
            logger.exception("analyze_failed")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/reset", methods=["POST"]) 
    def reset() -> Any:
        try:
            data: Dict[str, Any] = request.get_json(silent=True) or {}
            session_id: str = str(data.get("session_id") or "default")
            _rolling_emotions.clear()
            _session_rolling[session_id].clear()
            _session_records[session_id].clear()
            return jsonify({"ok": True})
        except Exception as exc:
            logger.exception("reset_failed")
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/report/json", methods=["GET"]) 
    def report_json() -> Any:
        session_id: str = str(request.args.get("session_id") or "default")
        records = _session_records.get(session_id, [])
        return jsonify({
            "session_id": session_id,
            "count": len(records),
            "records": records,
            "rolling_average": _to_plain_floats(_compute_rolling_average_session(session_id)),
        })

    @app.route("/api/report/csv", methods=["GET"]) 
    def report_csv() -> Any:
        import csv
        from flask import Response
        session_id: str = str(request.args.get("session_id") or "default")
        records = _session_records.get(session_id, [])
        fieldnames = ["ts", "mode", "dominant", "face_count"] + sorted({k for r in records for k in r.get("emotions", {}).keys()})
        def generate():
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            yield output.getvalue()
            output.seek(0); output.truncate(0)
            for r in records:
                row = {k: r.get(k) for k in ["ts", "mode", "dominant", "face_count"]}
                for k in fieldnames[4:]:
                    row[k] = float(r.get("emotions", {}).get(k, 0.0))
                writer.writerow(row)
                yield output.getvalue()
                output.seek(0); output.truncate(0)
        return Response(generate(), mimetype='text/csv', headers={"Content-Disposition": f"attachment; filename={session_id}.csv"})

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    # Warm up DeepFace in a lightweight way
    try:
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        DeepFace.analyze(dummy, actions=["emotion"], enforce_detection=False)
        MODEL_READY = True
    except Exception:
        MODEL_READY = False
    app.run(host="0.0.0.0", port=port, debug=True)


