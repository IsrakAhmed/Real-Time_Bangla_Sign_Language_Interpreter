# Real-Time Bangla Sign Language Interpreter

This is a real-time Bangla Sign Language (BdSL) interpreter that detects and recognizes hand signs from video input, translates them into Bangla characters, and forms meaningful sentences. It leverages deep learning models for detection (YOLOv8) and recognition (ConvNeXt), with a user-friendly Tkinter GUI for real-time visualization and text-to-speech output in Bangla. The application supports advanced Bangla sentence formation rules.

## Features
- **Real-Time Detection**: Uses YOLOv8 to detect hand signs in webcam video feed.
- **Sign Recognition**: Employs a ConvNeXt-based model for classifying 49 Bangla characters, including vowels, consonants, digits, and special characters.
- **Sentence Formation**: Implements advanced Bangla grammar rules:
  - Converts vowels after consonants to কার forms (e.g., দ + ই → দি).
  - Handles special cases like consonant + অ + vowel (e.g., দ + অ + ই → দই).
  - Supports word boundaries with spaces and special characters (ং, ঃ, ্‌).
  - Example: র + ব + ই + উ + ল → রবিউল.
- **User Interface**: Tkinter-based GUI with a fixed 1000x820 window, displaying:
  - Live video feed with detected signs and bounding boxes.
  - Real-time sentence display in Bangla (Noto Sans Bengali font).
  - Buttons for clearing sentences and triggering text-to-speech.
- **Text-to-Speech**: Converts formed sentences to spoken Bangla using gTTS.
- **GPU Support**: Optimizes computations with TensorFlow GPU support if available.

## Project Structure
```
Real-Time_Bangla_Sign_Language_Interpreter/
├── dataset/
│   ├── Detection/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   ├── test/
│   │   │   ├── images/
│   │   │   ├── labels/
│   │   └── data.yaml
│   ├── Recognition/
│   │   ├── train/
│   │   │   ├── <class_folders>/
│   │   └── test/
│   │       ├── <class_folders>/
├── runs/
│   ├── detect/
│   │   └── train7/
│   │       └── weights/
│   │           ├── best.pt
│   │           ├── last.pt
├── bestModels/
│   ├── best_model.keras
├── models/
│   ├── recognition_model_convnext - Test Loss 0.4509 - Test Accuracy 0.9803.keras
├── training_plots.png
├── convNextmodel.tflite
├── Bangla_Sign_Language_APP.py
├── Train_Test.ipynb
└── README.md
```

## Prerequisites
- **Python**: 3.10.12 or compatible version.
- **Hardware**:
  - GPU recommended for training and inference.
  - Webcam for real-time video input.
- **Dependencies**:
  ```bash
  pip install -U ultralytics==8.3.152 tensorflow==2.16.1 opencv-python pillow gTTS pygame numpy matplotlib scikit-learn torch==2.7.0
  ```
- **Font**: Install `NotoSansBengali-Regular.ttf`.
- **Model Files**:
  - YOLOv8 detection model: `runs/detect/train7/weights/best.pt`
  - Recognition model: `convNextmodel.tflite` (or `recognition_model_convnext - Test Loss 0.4509 - Test Accuracy 0.9803.keras` if not using TFLite)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/IsrakAhmed/Real-Time_Bangla_Sign_Language_Interpreter.git
   cd Real-Time_Bangla_Sign_Language_Interpreter
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Create a `requirements.txt` with:
   ```
   ultralytics==8.3.152
   tensorflow==2.16.1
   opencv-python
   pillow
   gTTS
   pygame
   numpy
   matplotlib
   scikit-learn
   torch==2.7.0
   ```

3. **Set Up Models**:
   - Place `best.pt` in `runs/detect/train7/weights/`.
   - Place `convNextmodel.tflite` in the project root or update `MODEL_CONFIG['path']` in `Bangla_Sign_Language_APP.py` to use `models/recognition_model_convnext - Test Loss 0.4509 - Test Accuracy 0.9803.keras`.
   - Ensure `NotoSansBengali-Regular.ttf` is available.

4. **Prepare Dataset** (if training):
   - Organize detection dataset in `dataset/Detection/` with `train/` (11,774 images) and `test/` (2,940 images) folders containing images and YOLO-compatible label files.
   - Organize recognition dataset in `dataset/Recognition/` with `train/` (8,247 images) and `test/` (2,940 images) folders containing class subfolders (0-48).
   - Update `data.yaml` with correct paths and class count (`nc: 49`).

## Usage
1. **Run the Application**:
   ```bash
   python Bangla_Sign_Language_APP.py
   ```
   - The Tkinter GUI opens with a fixed 1000x820 window.
   - The webcam feed displays detected signs with green bounding boxes and labels (e.g., "Sign: ই").
   - Formed sentences appear in the sentence frame (e.g., দই, রবিউল).
   - Use the **Clear Sentence** button to reset the sentence.
   - Use the **Speak Sentence** button to hear the sentence in Bangla.

2. **Training Models** (optional):
   - **Detection Model** (`train_detection.py`):
     ```bash
     python train_detection.py
     ```
     - Trains YOLOv8s on `dataset/Detection/` with `imgsz=128`, `batch=4`, `epochs=20`.
     - Outputs saved to `runs/detect/train7/weights/best.pt`.
     - Achieved mAP50: 0.942, mAP50-95: 0.762 on test set.
   - **Recognition Model** (`train_recognition.py`):
     ```bash
     python train_recognition.py
     ```
     - Trains ConvNeXtBase with a custom head on `dataset/Recognition/` with `img_size=224`, `batch_size=16`, `epochs=25` + 10 fine-tuning.
     - Uses data augmentation, class weights, and cosine decay learning rate.
     - Saves best model to `bestModels/best_model.keras` and final model to `models/recognition_model_convnext - Test Loss 0.4509 - Test Accuracy 0.9803.keras`.
     - Generates `training_plots.png` for loss and accuracy curves.

3. **Converting to TFLite** (if needed):
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('bestModels/best_model.keras')
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   with open('convNextmodel.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

## Dataset
- **Detection** (`dataset/Detection/`):
  - Train: 11,774 images, Test: 2,940 images.
  - YOLO format: `images/` (JPEG/PNG), `labels/` (.txt files with class ID, bounding box coordinates).
  - 49 classes corresponding to Bangla characters.
  - `data.yaml`:
    ```yaml
    train: dataset/Detection/train
    val: dataset/Detection/test
    nc: 49
    names: ['0', '1', ..., '48']
    ```
- **Recognition** (`dataset/Recognition/`):
  - Train: 8,247 images, Validation: 3,527 images, Test: 2,940 images (49 classes).
  - Class folders (0-48) with images in JPEG/PNG format.
  - Classes map to Bangla characters via `class_mapping` and `bangla_mapping`.

## Sentence Formation Rules
- **Space**: Ends a word, added only if the last character isn’t a space.
- **Special Characters** (ং, ঃ, ্‌): Added after valid characters (not space or another special character).
- **Digits** (০-৯): Added as is.
- **Vowel "অ"**: Allowed at word start or after consonants.
- **Consonants**: Added as is.
- **Vowels** (except অ):
  - After a consonant (without a prior কার or consonant + অ): Converts to কার form (e.g., দ + ই → দি).
  - After a non-consonant, কার form, or consonant + অ: Added as a standalone vowel (e.g., দ + অ + ই → দই).
- **Examples**:
  - দ, অ, ই → দই
  - র, ব, ই, উ, ল → রবিউল
  - ক, ই, ই → কিই
  - দ, অ, আ → দআ

## Performance
- **Detection Model** (YOLOv8s):
  - mAP50: 0.942, mAP50-95: 0.762 (test set, 2,940 images).
  - Trained on NVIDIA GTX 1660 Ti (6GB VRAM).
  - Image size: 128x128, batch size: 4, epochs: 20.
- **Recognition Model** (ConvNeXtBase):
  - **Test Accuracy**: 0.9803 (98.03%)
  - **Test Loss**: 0.4509
  - Image size: 224x224, batch size: 16, epochs: 25 + 10 fine-tuning.
  - Validation accuracy (best): 0.9685 (epoch 10 of fine-tuning).
  - Uses L2 regularization, dropout (0.5, 0.3), and cosine decay learning rate.

## Limitations
- Requires a webcam and proper lighting for accurate detection.
- YOLO model may miss detections in complex backgrounds; consider increasing `imgsz` to 416 or 640 if VRAM allows.
- Sentence formation supports basic Bangla grammar but not complex conjuncts (e.g., ক্ষ).
- Tkinter UI is functional but less modern; consider PyQt for future enhancements.
- TFLite model assumes conversion from `.keras`; ensure compatibility.

## Acknowledgments
- Built by **Israk Ahmed** and **Esrat Jahan Riya** as part of an academic project.
- Special thanks to the open-source community for tools and libraries.