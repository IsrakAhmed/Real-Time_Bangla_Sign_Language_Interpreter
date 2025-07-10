import cv2
import numpy as np
import threading
import time
import os
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image, ImageDraw, ImageFont

# Load models
detection_model = YOLO('runs/detect/train7/weights/best.pt')

# Load TensorFlow Lite model
tflite_model_path = 'convNextmodel.tflite'
if not os.path.exists(tflite_model_path):
    raise FileNotFoundError(f"TFLite model file {tflite_model_path} not found.")
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img_size = 224
class_names = [str(i) for i in range(49)]

# Class mapping dictionary
class_mapping = {
    0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17,
    10: 18, 11: 19, 12: 2, 13: 20, 14: 21, 15: 22, 16: 23, 17: 24, 18: 25, 19: 26,
    20: 27, 21: 28, 22: 29, 23: 3, 24: 30, 25: 31, 26: 32, 27: 33, 28: 34, 29: 35,
    30: 36, 31: 37, 32: 38, 33: 39, 34: 4, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
    40: 45, 41: 46, 42: 47, 43: 48, 44: 5, 45: 6, 46: 7, 47: 8, 48: 9
}

bangla_mapping = {
    0: "অ", 1: "আ", 2: "ই", 3: "উ", 4: "এ", 5: "ও", 6: "ক", 7: "খ", 8: "গ", 9: "ঘ",
    10: "চ", 11: "ছ", 12: "জ", 13: "ঝ", 14: "ট", 15: "ঠ", 16: "ড", 17: "ঢ", 18: "ত",
    19: "থ", 20: "দ", 21: "ধ", 22: "প", 23: "ফ", 24: "ব", 25: "ভ", 26: "ম", 27: "য়",
    28: "র", 29: "ল", 30: "ন", 31: "স", 32: "হ", 33: "ড়", 34: "ং", 35: "ঃ", 36: "০",
    37: "১", 38: "২", 39: "৩", 40: "৪", 41: "৫", 42: "৬", 43: "৭", 44: "৮", 45: "৯",
    46: "্‌", 47: " ", 48: "ঞ"
}

# Load Bangla font for Pillow
font_path = 'C:/Windows/Fonts/NotoSansBengali-Regular.ttf'  # Try system fonts first
if not os.path.exists(font_path):
    font_path = 'NotoSansBengali-Regular.ttf'  # Fallback to current directory
    if not os.path.exists(font_path):
        print(f"Font file not found in system or current directory. Please install Noto Sans Bengali or place NotoSansBengali-Regular.ttf in the script directory.")
        font = ImageFont.load_default()  # Fallback (won't support Bangla)
    else:
        try:
            font = ImageFont.truetype(font_path, size=48)  # Larger font size
            print(f"Successfully loaded font: {font_path}")
        except Exception as e:
            print(f"Error loading font: {e}. Falling back to default font (won't support Bangla).")
            font = ImageFont.load_default()
else:
    try:
        font = ImageFont.truetype(font_path, size=48)  # Larger font size
        print(f"Successfully loaded font: {font_path}")
    except Exception as e:
        print(f"Error loading font: {e}. Falling back to default font (won't support Bangla).")
        font = ImageFont.load_default()

# Shared data between threads
frame = None
result_frame = None
lock = threading.Lock()
running = True

# Function to overlay Bangla text on OpenCV frame using Pillow
def put_bangla_text(frame, text, position, font, color=(0, 255, 0)):
    # Ensure position is within frame bounds
    x, y = position
    if y < 0:
        y = 0  # Prevent text from being drawn outside the frame
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw text
    draw.text((x, y), text, font=font, fill=color)
    
    # Convert back to OpenCV format (BGR)
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame_bgr

# Detection + recognition thread
def detect_and_recognize():
    global frame, result_frame, running
    while running:
        if frame is not None:
            with lock:
                input_frame = frame.copy()

            # Detection
            results = detection_model(input_frame)[0]
            boxes = results.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped = input_frame[y1:y2, x1:x2]

                if cropped.size == 0:  # Skip if box is invalid
                    continue

                try:
                    resized = cv2.resize(cropped, (img_size, img_size))
                    normed = preprocess_input(resized.astype(np.float32))
                    # Prepare input for TFLite model
                    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(normed, axis=0))
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])[0]
                    predicted_label = int(class_names[np.argmax(pred)])  # Get predicted class index
                    mapped_label = class_mapping[predicted_label]  # Map to new label
                    bangla_label = bangla_mapping[mapped_label]  # Get Bangla character

                    # Debug: Print the labels to verify
                    print(f"Predicted: {predicted_label}, Mapped: {mapped_label}, Bangla: {bangla_label}")

                    # Draw rectangle
                    cv2.rectangle(input_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Overlay Bangla text using Pillow
                    input_frame = put_bangla_text(input_frame, f'Sign: {bangla_label}', (x1, y1 - 50), font, color=(0, 255, 0))

                except Exception as e:
                    print(f"Recognition error: {e}")

            with lock:
                result_frame = input_frame

        time.sleep(0.01)  # Prevent 100% CPU usage

# Start worker thread
worker_thread = threading.Thread(target=detect_and_recognize)
worker_thread.start()

# Main loop to capture and display frames
cap = cv2.VideoCapture(0)
while True:
    ret, frame_raw = cap.read()
    if not ret:
        break

    with lock:
        frame = frame_raw.copy()

    with lock:
        display = result_frame.copy() if result_frame is not None else frame_raw.copy()

    cv2.imshow("Bangla Sign Language Detection & Recognition", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cap.release()
cv2.destroyAllWindows()
worker_thread.join()