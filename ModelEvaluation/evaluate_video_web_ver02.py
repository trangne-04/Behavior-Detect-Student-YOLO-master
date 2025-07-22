import cv2
import torch
from ultralytics import YOLO
from flask import Flask, Response, render_template, send_file
import numpy as np
import os
from datetime import datetime

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, 
           template_folder=TEMPLATE_DIR)

# Constants
MODEL_PATH = r"/hdd2/minhnv/CodingYOLOv12/Behavior-Detect-Student-YOLO/StaticModels/yolov12s_29032025_013131_exp1_baseline/runs/detect/train/weights/best.pt"
VIDEO_PATH = r"/hdd2/minhnv/CodingYOLOv12/TestingVideo/13.00.40-13.05.00[M][0@0][0] ver2.mp4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESH = 0.3
IOU_THRESH = 0.45

# Custom display settings
BOX_THICKNESS = 2
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
COLOR = (0, 255, 0)

# Global variables
model = None
cap = None
is_processing = False

def initialize_model():
    try:
        global model
        model = YOLO(MODEL_PATH).to(DEVICE)
        print(f"Model loaded on {DEVICE.upper()}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def initialize_video():
    try:
        global cap
        print(f"Attempting to open video: {VIDEO_PATH}")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video at {VIDEO_PATH}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video opened successfully:")
        print(f"- Resolution: {width}x{height}")
        print(f"- FPS: {fps:.2f}")
        print(f"- Total frames: {frame_count}")
        return True
    except Exception as e:
        print(f"Error initializing video: {e}")
        return False

def process_frame(frame):
    if frame is None or model is None:
        return frame
    
    try:
        # Run inference
        results = model.predict(
            frame,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            device=DEVICE,
            verbose=False,
            stream=True
        )

        # Process results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (x1, y1), 
                            (x2, y2), 
                            COLOR, 
                            BOX_THICKNESS)
                
                # Draw label
                label = f"{model.names[cls_id]} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label,
                                             cv2.FONT_HERSHEY_SIMPLEX,
                                             TEXT_SCALE,
                                             TEXT_THICKNESS)
                
                cv2.rectangle(frame,
                            (x1, y1 - 20),
                            (x1 + tw, y1),
                            COLOR,
                            -1)
                
                cv2.putText(frame,
                           label,
                           (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           TEXT_SCALE,
                           (64, 64, 64),
                           TEXT_THICKNESS)
        
        return frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame

def generate_frames():
    global cap, is_processing
    print("Starting frame generation...")
    
    if not initialize_video():
        return None
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached, resetting to beginning...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # Process frame if model is loaded
        if is_processing:
            frame = process_frame(frame)
            
        # Resize frame for better streaming performance
        frame = cv2.resize(frame, (960, 540))
            
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
            
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global is_processing
    if not initialize_model():
        return "Error loading model"
    is_processing = True
    return "Model loaded successfully"

@app.route('/stop')
def stop():
    global is_processing
    is_processing = False
    return "Processing stopped"

# Add cleanup function
@app.teardown_appcontext
def cleanup(error):
    global cap
    if cap is not None:
        cap.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 