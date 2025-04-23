from flask import Flask, request, send_file
import cv2
import numpy as np
import time
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/upload', methods=['POST'])
def upload():
    # Save the incoming image with timestamp
    timestamp = int(time.time())
    filename = f"{UPLOAD_FOLDER}/image_{timestamp}.jpg"
    
    with open(filename, 'wb') as f:
        f.write(request.data)
    print(f"[INFO] Image received and saved as {filename}")

    # Load image for face detection
    image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        print("[ERROR] Failed to decode image!")
        return "Invalid image", 400

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        print(f"[DETECTED] Found {len(faces)} face(s)!")
        return "Face detected!", 200
    else:
        print("[INFO] No face detected.")
        return "No face detected", 200

@app.route('/view-latest')
def view_latest():
    images = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)
    if not images:
        return "No images uploaded yet", 404
    latest_image_path = os.path.join(UPLOAD_FOLDER, images[0])
    return send_file(latest_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)