from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import os
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

def log(msg):
    print(f"[DEBUG] {msg}")

def get_dominant_color(image, k=3):
    log("Running KMeans to get dominant color...")
    image = image.reshape((-1, 3))
    clt = KMeans(n_clusters=k, n_init=10)
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    log(f"Dominant RGB color: {dominant_color}")
    return dominant_color.astype(int)

def classify_skin_tone(rgb):
    log(f"Classifying skin tone for RGB: {rgb}")
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    log(f"Converted to HSV: H={h}, S={s}, V={v}")
    if v > 210 and s < 60:
        return "Fair"
    elif 180 < v <= 210:
        return "Medium"
    elif 140 < v <= 180:
        return "Olive"
    elif 90 < v <= 140:
        return "Brown"
    else:
        return "Dark"

def extract_face_region(image):
    log("Extracting face region...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    log(f"Using Haar cascade from: {face_cascade_path}")
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    log(f"Detected {len(faces)} faces")
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    roi = image[y + int(h*0.4): y + int(h*0.65), x + int(w*0.3): x + int(w*0.7)]
    return roi

def detect_skin_tone(image_path):
    try:
        log(f"Reading image from: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            log("Failed to read image.")
            return "Invalid image path"
        face_roi = extract_face_region(image)
        if face_roi is None:
            log("No face detected in image.")
            return "No face detected"
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        dominant_rgb = get_dominant_color(face_roi, k=3)
        tone = classify_skin_tone(dominant_rgb)
        log(f"Final classified tone: {tone}")
        return tone
    except Exception as e:
        log(f"Error during detection: {e}")
        return str(e)

@app.route("/", methods=["GET"])
def home():
    log("Home endpoint hit.")
    return "Skin Tone Detector API"

@app.route("/detect", methods=["POST"])
def detect():
    log("Detect endpoint hit.")
    if "image" not in request.files:
        log("No image file in request.")
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    image_path = "temp_img.png"
    file.save(image_path)
    log(f"Saved image to {image_path}")
    result = detect_skin_tone(image_path)
    os.remove(image_path)
    log("Temp image deleted after processing.")
    return jsonify({"skin_tone": result})

@app.route("/detect_base64", methods=["POST"])
def detect_skin_tone_base64():
    try:
        data = request.get_json()
        if not data or "image_base64" not in data:
            return jsonify({"error": "Missing image_base64"}), 400

        image_data = base64.b64decode(data["image_base64"])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = np.array(image)

        face_roi = extract_face_region(image)
        if face_roi is None:
            return jsonify({"error": "No face detected"}), 400

        dominant_rgb = get_dominant_color(face_roi)
        tone = classify_skin_tone(dominant_rgb)

        return jsonify({"skin_tone": tone})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    log("Starting Flask server...")
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
