from flask import Flask, request, jsonify
import cv2
import numpy as np
import time
import os
import mediapipe as mp
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from dotenv import load_dotenv
import datetime

load_dotenv()

# Flask app setup
app = Flask(__name__)

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# MongoDB config
client = MongoClient(os.getenv("MONGO_URI"))
db = client["camera_db"]
collection = db["detected_faces"]

# MediaPipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

@app.route('/upload', methods=['POST'])
def upload():
    timestamp = int(time.time())

    # Decode image from request
    image = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return "Invalid image", 400

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)

    if results.detections:
        print(f"[DETECTED] Found {len(results.detections)} face(s)")

        for detection in results.detections:
            mp_drawing.draw_detection(image, detection)

        # Save to temp file
        filename = f"detected_{timestamp}.jpg"
        cv2.imwrite(filename, image)

        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(filename, public_id=f"faces/face_{timestamp}")
        image_url = upload_result["secure_url"]

        # Save to MongoDB
        collection.insert_one({
            "url": image_url,
            "timestamp": datetime.datetime.utcnow()
        })

        os.remove(filename)
        return jsonify({"status": "face_detected", "url": image_url}), 200

    else:
        return "No face detected", 200

@app.route('/gallery', methods=['GET'])
def gallery():
    results = collection.find().sort("timestamp", -1)
    image_urls = [doc["url"] for doc in results]
    return jsonify(image_urls)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)