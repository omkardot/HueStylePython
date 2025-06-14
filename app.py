import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from flask import Flask, request, jsonify
import logging
import base64
from io import BytesIO
from PIL import Image
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

def perceived_brightness(rgb):
    return np.sqrt(0.299 * (rgb[0] ** 2) + 0.587 * (rgb[1] ** 2) + 0.114 * (rgb[2] ** 2))

def hex_to_color_name(hex_value):
    try:
        named_colors = mcolors.CSS4_COLORS
        min_distance = float('inf')
        closest_color = None
        target_rgb = mcolors.hex2color(hex_value)
        for color_name, color_hex in named_colors.items():
            color_rgb = mcolors.hex2color(color_hex)
            distance = sum((target_rgb[i] - color_rgb[i]) ** 2 for i in range(3)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_color = color_name
        return closest_color
    except Exception as e:
        print(f"Error finding closest color name: {e}")
        return None

def extract_face_region(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        return image[y + int(h * 0.4): y + int(h * 0.65), x + int(w * 0.3): x + int(w * 0.7)]
    except Exception as e:
        logging.error(f"Face extraction error: {str(e)}")
        return None

def get_dominant_color(image, k=3):
    try:
        image = image.reshape((-1, 3))
        clt = KMeans(n_clusters=k, n_init=10)
        labels = clt.fit_predict(image)
        label_counts = Counter(labels)
        dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        return dominant_color.astype(int)
    except Exception as e:
        logging.error(f"Error in KMeans: {str(e)}")
        raise

def classify_skin_tone(rgb):
    try:
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
    except Exception as e:
        logging.error(f"Error in classify_skin_tone: {str(e)}")
        return "Unknown"


@app.route("/recommend_colors", methods=["POST"])
def recommend_colors():
    try:
        data = request.get_json()
        skin_hex = data.get("skin_hex")
        saved_colors = data.get("saved_colors")

        if not skin_hex or not saved_colors:
            return jsonify({"error": "Missing skin_hex or saved_colors"}), 400

        skin_rgb = hex_to_rgb(skin_hex)
        skin_brightness = perceived_brightness(skin_rgb)
        color_contrasts = []

        for hex_code in saved_colors:
            color_rgb = hex_to_rgb(hex_code)
            brightness = perceived_brightness(color_rgb)
            contrast = abs(skin_brightness - brightness)
            color_contrasts.append({
                "name": hex_to_color_name(hex_code),
                "hexCode": hex_code,
                "contrast": contrast
            })

        color_contrasts.sort(key=lambda x: x['contrast'], reverse=True)
        recommended = color_contrasts[:5]
        if len(recommended) < 5:
            recommended += color_contrasts[5:5 + (5 - len(recommended))]

        return jsonify({"recommended": [{"name": c["name"], "hexCode": c["hexCode"]} for c in recommended]})

    except Exception as e:
        logging.error(f"Error in recommend_colors: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/detect", methods=["POST"])
def detect():
    logging.info("Detect endpoint hit.")

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_path = "temp_img.png"
    file.save(image_path)

    try:
        image_bgr = cv2.imread(image_path)
        face_roi = extract_face_region(image_bgr)

        if face_roi is None:
            os.remove(image_path)
            return jsonify({"error": "No face detected"}), 400

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        dominant_rgb = get_dominant_color(face_rgb)
        tone = classify_skin_tone(dominant_rgb)
        hex_color = '#{:02x}{:02x}{:02x}'.format(*dominant_rgb)

        os.remove(image_path)

        return jsonify({
            "skin_tone": tone,
            "dominant_hex": hex_color
        }), 200

    except Exception as e:
        os.remove(image_path)
        logging.error(f"Error in detect: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
