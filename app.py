from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import os

app = Flask(__name__)

def get_dominant_color(image, k=3):
    image = image.reshape((-1, 3))
    clt = KMeans(n_clusters=k, n_init=10)
    labels = clt.fit_predict(image)
    label_counts = Counter(labels)
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return dominant_color.astype(int)

def classify_skin_tone(rgb):
    r, g, b = rgb
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    roi = image[y + int(h*0.4): y + int(h*0.65), x + int(w*0.3): x + int(w*0.7)]
    return roi

def detect_skin_tone(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Invalid image path"
        face_roi = extract_face_region(image)
        if face_roi is None:
            return "No face detected"
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        dominant_rgb = get_dominant_color(face_roi, k=3)
        tone = classify_skin_tone(dominant_rgb)
        return tone
    except Exception as e:
        return str(e)

@app.route("/", methods=["GET"])
def home():
    return "Skin Tone Detector API"

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    image_path = "temp_img.png"
    file.save(image_path)
    result = detect_skin_tone(image_path)
    os.remove(image_path)
    return jsonify({"skin_tone": result})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
