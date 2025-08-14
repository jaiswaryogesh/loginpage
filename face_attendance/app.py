import base64
import csv
import os
from datetime import datetime

from flask import Flask, jsonify, render_template, request
import numpy as np
import cv2

try:
    import face_recognition  # type: ignore
    FACE_LIB_AVAILABLE = True
except Exception:
    FACE_LIB_AVAILABLE = False


app = Flask(__name__)

ATTENDANCE_CSV = os.path.join(app.root_path, "attendance.csv")
KNOWN_DIR = os.path.join(app.root_path, "static", "known")


def ensure_attendance_file() -> None:
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as file_handle:
            csv_writer = csv.writer(file_handle)
            csv_writer.writerow(["name", "timestamp"])


ensure_attendance_file()


known_face_encodings: list[np.ndarray] = []
known_face_names: list[str] = []


def load_known_faces() -> None:
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []

    if not FACE_LIB_AVAILABLE:
        return

    os.makedirs(KNOWN_DIR, exist_ok=True)

    for filename in os.listdir(KNOWN_DIR):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        file_path = os.path.join(KNOWN_DIR, filename)
        image = face_recognition.load_image_file(file_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])


load_known_faces()


@app.route("/")
def index():  # type: ignore[override]
    return render_template(
        "index.html",
        face_ready=FACE_LIB_AVAILABLE,
        num_known=len(known_face_names),
    )


def write_attendance(name: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(ATTENDANCE_CSV, "a", newline="") as file_handle:
        csv.writer(file_handle).writerow([name, timestamp])


@app.post("/recognize")
def recognize():  # type: ignore[override]
    if not FACE_LIB_AVAILABLE:
        return jsonify({"ok": False, "error": "face_recognition not installed"}), 500

    payload = request.get_json(silent=True) or {}
    data_url: str | None = payload.get("image")
    if not data_url or "," not in data_url:
        return jsonify({"ok": False, "error": "Invalid image payload"}), 400

    try:
        encoded_part = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded_part)
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if bgr_image is None:
            return jsonify({"ok": False, "error": "Failed to decode image"}), 400

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        recognized_names: list[str] = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            if True in matches:
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_index = int(np.argmin(distances))
                if matches[best_index]:
                    name = known_face_names[best_index]
            recognized_names.append(name)
            if name != "Unknown":
                write_attendance(name)

        return jsonify({
            "ok": True,
            "faces_detected": len(face_encodings),
            "recognized": recognized_names,
        })
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500


@app.post("/reload-known")
def reload_known():  # type: ignore[override]
    if not FACE_LIB_AVAILABLE:
        return jsonify({"ok": False, "error": "face_recognition not installed"}), 500
    load_known_faces()
    return jsonify({"ok": True, "num_known": len(known_face_names)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)