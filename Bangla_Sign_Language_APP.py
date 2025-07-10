import cv2
import numpy as np
import threading
import time
import os
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import ImageTk
import platform
from gtts import gTTS
from io import BytesIO
import pygame

# Check for GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
use_gpu = len(physical_devices) > 0
if use_gpu:
    print("GPU detected. Using GPU for computations.")
else:
    print("No GPU detected. Falling back to CPU.")

# Model configuration
MODEL_CONFIG = {
    'type': 'tflite',  # Options: 'tflite', 'keras', 'h5'
    'path': 'convNextmodel.tflite'  # Update path to your model file
}

# Load YOLO detection model
detection_model = YOLO('runs/detect/train7/weights/best.pt')

# Function to load model based on type
def load_model(config):
    model_type = config['type'].lower()
    model_path = config['path']
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    if model_type == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return {
            'model': interpreter,
            'input_details': input_details,
            'output_details': output_details,
            'type': 'tflite'
        }
    elif model_type in ['keras', 'h5']:
        model = tf.keras.models.load_model(model_path)
        return {
            'model': model,
            'type': model_type
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'tflite', 'keras', or 'h5'.")

# Load recognition model
recognition_model = load_model(MODEL_CONFIG)

img_size = 224
class_names = [str(i) for i in range(49)]
confidence_threshold = 0.7  # Minimum confidence for accepting a prediction

# Class mappings
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

# Vowel to কার mapping
vowel_to_kar = {
    "আ": "া",  # আ-কার
    "ই": "ি",  # ই-কার
    "উ": "ু",  # উ-কার
    "এ": "ে",  # এ-কার
    "ও": "ো"   # ও-কার
}

# Define consonants (excluding vowels, digits, special chars, and space)
consonants = set([
    "ক", "খ", "গ", "ঘ", "চ", "ছ", "জ", "ঝ", "ট", "ঠ", "ড", "ঢ", "ত", "থ", "দ", "ধ",
    "প", "ফ", "ব", "ভ", "ম", "য়", "র", "ল", "ন", "স", "হ", "ড়", "ঞ"
])

# Define vowels
vowels = set(["অ", "আ", "ই", "উ", "এ", "ও"])

# Define kar forms
kar_forms = set(["া", "ি", "ু", "ে", "ো"])

# Define special characters
special_chars = set(["ং", "ঃ", "্‌"])

# Define digits
digits = set(["০", "১", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯"])

# Load Bangla font
font_path = 'C:/Windows/Fonts/NotoSansBengali-Regular.ttf'
if not os.path.exists(font_path):
    font_path = 'NotoSansBengali-Regular.ttf'
    if not os.path.exists(font_path):
        print("Font file not found. Please install Noto Sans Bengali or place NotoSansBengali-Regular.ttf in the script directory.")
        font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.truetype(font_path, size=48)
            print(f"Successfully loaded font: {font_path}")
        except Exception as e:
            print(f"Error loading font: {e}. Using default font.")
            font = ImageFont.load_default()
else:
    try:
        font = ImageFont.truetype(font_path, size=48)
        print(f"Successfully loaded font: {font_path}")
    except Exception as e:
        print(f"Error loading font: {e}. Using default font.")
        font = ImageFont.load_default()

# Shared data
frame = None
result_frame = None
lock = threading.Lock()
running = True
sentence = []
last_detected_time = 0
last_detected_label = None
detection_interval = 1.0  # Time (seconds) to wait before adding a new character
last_was_kar = False  # Track if the last addition was a কার form
last_was_consonant_then_a = False  # Track if the last sequence is consonant + অ

# Function to overlay Bangla text on OpenCV frame using Pillow
def put_bangla_text(frame, text, position, font, color=(0, 255, 0)):
    x, y = position
    if y < 0:
        y = 0
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)
    draw.text((x, y), text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Function to perform inference based on model type
def run_inference(model_info, input_data):
    if model_info['type'] == 'tflite':
        interpreter = model_info['model']
        input_details = model_info['input_details']
        output_details = model_info['output_details']
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])[0]
    else:  # keras or h5
        model = model_info['model']
        return model.predict(input_data, verbose=0)[0]

# Function to convert sentence to speech
def text_to_speech(sentence_text):
    if not sentence_text.strip():
        return False, "No sentence to speak."
    
    try:
        # Create in-memory MP3
        tts = gTTS(text=sentence_text, lang='bn')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)

        # Initialize pygame mixer and play audio
        pygame.mixer.init()
        pygame.mixer.music.load(mp3_fp)
        pygame.mixer.music.play()

        # Wait until audio finishes
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        pygame.mixer.quit()
        return True, "Speech played successfully."
    except Exception as e:
        return False, f"TTS error: {str(e)}"

# Function to build logical sentence with advanced Bangla rules
def build_sentence(new_label):
    global sentence, last_was_kar, last_was_consonant_then_a
    # Handle space (end of word)
    if new_label == " ":
        if sentence and sentence[-1] != " ":
            sentence.append(" ")
            last_was_kar = False
            last_was_consonant_then_a = False
        return
    
    # Handle special characters (ং, ঃ, ্‌)
    if new_label in special_chars:
        if sentence and sentence[-1] not in [" ", *special_chars]:
            sentence.append(new_label)
            last_was_kar = False
            last_was_consonant_then_a = False
        return
    
    # Handle digits (০-৯)
    if new_label in digits:
        sentence.append(new_label)
        last_was_kar = False
        last_was_consonant_then_a = False
        return
    
    # Handle 'অ'
    if new_label == "অ":
        if not sentence or sentence[-1] == " " or sentence[-1] in consonants:
            sentence.append(new_label)
            last_was_kar = False
            last_was_consonant_then_a = (sentence and sentence[-1] == "অ" and len(sentence) > 1 and sentence[-2] in consonants)
        return
    
    # Handle consonants
    if new_label in consonants:
        sentence.append(new_label)
        last_was_kar = False
        last_was_consonant_then_a = False
        return
    
    # Handle vowels (except অ, which is handled above)
    if new_label in vowels:
        if sentence and sentence[-1] in consonants and not last_was_kar and not last_was_consonant_then_a:
            # Consonant + vowel → কার form (unless preceded by consonant + অ)
            sentence.append(vowel_to_kar[new_label])
            last_was_kar = True
            last_was_consonant_then_a = False
        else:
            # Vowel after non-consonant, after a কার form, or after consonant + অ → standalone vowel
            sentence.append(new_label)
            last_was_kar = False
            last_was_consonant_then_a = False

# Detection and recognition thread
def detect_and_recognize():
    global frame, result_frame, running, sentence, last_detected_time, last_detected_label
    while running:
        if frame is not None:
            with lock:
                input_frame = frame.copy()

            results = detection_model(input_frame)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            current_time = time.time()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cropped = input_frame[y1:y2, x1:x2]

                if cropped.size == 0:
                    continue

                try:
                    resized = cv2.resize(cropped, (img_size, img_size))
                    normed = preprocess_input(resized.astype(np.float32))
                    input_data = np.expand_dims(normed, axis=0)
                    pred = run_inference(recognition_model, input_data)
                    confidence = np.max(pred)
                    if confidence < confidence_threshold:
                        continue
                    predicted_label = int(class_names[np.argmax(pred)])
                    mapped_label = class_mapping[predicted_label]
                    bangla_label = bangla_mapping[mapped_label]

                    # Debounce: Only add if different from last label or enough time has passed
                    if (current_time - last_detected_time >= detection_interval) and (bangla_label != last_detected_label):
                        build_sentence(bangla_label)
                        last_detected_time = current_time
                        last_detected_label = bangla_label

                    cv2.rectangle(input_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    input_frame = put_bangla_text(input_frame, f'Sign: {bangla_label}', (x1, y1 - 50), font, color=(0, 255, 0))

                except Exception as e:
                    print(f"Recognition error: {e}")

            with lock:
                result_frame = input_frame

        time.sleep(0.01)

# Tkinter UI
class SignLanguageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bangla Sign Language Recognition")
        self.geometry("1000x820")  # Fixed larger window size
        self.configure(bg="#e8ecef")  # Light gray background
        self.resizable(False, False)  # Disable resizing and maximize button

        # Styling
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Modern theme
        self.style.configure("TButton", font=("Helvetica", 12, "bold"), padding=10, background="#4a90e2", foreground="white")
        self.style.map("TButton", background=[('active', '#357abd')])  # Hover effect
        self.style.configure("TLabel", font=("Helvetica", 14), background="#e8ecef")
        self.style.configure("Status.TLabel", font=("Helvetica", 10), foreground="#555555")
        self.style.configure("Video.TFrame", background="#ffffff")
        self.style.configure("Sentence.TFrame", background="#f5f5f5")

        # Main frame for centering content
        self.main_frame = ttk.Frame(self, padding=20)
        self.main_frame.pack(expand=True, fill='both')

        # Title label
        self.title_label = ttk.Label(self.main_frame, text="Bangla Sign Language Interpreter", font=("Helvetica", 18, "bold"))
        self.title_label.pack(pady=(0, 20))

        # Video feed frame with border
        self.video_frame = ttk.Frame(self.main_frame, relief="solid", borderwidth=1, style="Video.TFrame")
        self.video_frame.pack(pady=10)
        self.video_label = tk.Label(self.video_frame, bg="#ffffff", bd=2, relief="flat")
        self.video_label.pack(padx=5, pady=5)

        # Sentence display frame
        self.sentence_frame = ttk.Frame(self.main_frame, relief="sunken", borderwidth=1, style="Sentence.TFrame")
        self.sentence_frame.pack(pady=20, padx=20, fill='x')
        self.sentence_var = tk.StringVar()
        self.sentence_label = ttk.Label(
            self.sentence_frame,
            textvariable=self.sentence_var,
            font=("Noto Sans Bengali", 18, "bold"),
            wraplength=900,  # Adjusted for larger window
            background="#f5f5f5",
            anchor="center"
        )
        self.sentence_label.pack(padx=10, pady=10)

        # Button frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        # Clear sentence button
        self.clear_button = ttk.Button(self.button_frame, text="Clear Sentence", command=self.clear_sentence)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        # Text-to-speech button
        self.speak_button = ttk.Button(self.button_frame, text="Speak Sentence", command=self.speak_sentence)
        self.speak_button.pack(side=tk.LEFT, padx=10)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(pady=10)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open webcam.")
            self.destroy()
            return
        self.update_video()

    def clear_sentence(self):
        global sentence, last_detected_label, last_was_kar, last_was_consonant_then_a
        sentence = []
        last_detected_label = None
        last_was_kar = False
        last_was_consonant_then_a = False
        self.sentence_var.set("")
        self.status_var.set("Sentence cleared")

    def speak_sentence(self):
        def run_tts():
            sentence_text = "".join(sentence)
            success, message = text_to_speech(sentence_text)
            self.after(0, lambda: self.status_var.set(message))
            if not success:
                self.after(0, lambda: messagebox.showerror("TTS Error", message))
        
        # Run TTS in a separate thread to avoid blocking UI
        self.status_var.set("Playing speech...")
        tts_thread = threading.Thread(target=run_tts)
        tts_thread.daemon = True
        tts_thread.start()

    def update_video(self):
        global frame, result_frame
        ret, frame_raw = self.cap.read()
        if ret:
            with lock:
                frame = frame_raw.copy()
                display = result_frame.copy() if result_frame is not None else frame_raw.copy()

            # Convert to PhotoImage
            display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Update sentence
            self.sentence_var.set("".join(sentence))

        self.after(10, self.update_video)

    def on_closing(self):
        global running
        running = False
        self.cap.release()
        self.destroy()

# Main execution
if __name__ == "__main__":
    if platform.system() != "Emscripten":
        worker_thread = threading.Thread(target=detect_and_recognize)
        worker_thread.start()
        app = SignLanguageApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
        worker_thread.join()