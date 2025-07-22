import cv2
import torch
from ultralytics import YOLO
from flask import Flask, Response, render_template
import os
import time
import threading
from queue import Queue

# Khởi tạo Flask app
app = Flask(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = r"/hdd2/minhnv/CodingYOLOv12/Behavior-Detect-Student-YOLO/StaticModels/yolov12s13032025_000207_ver-dataset6/_ver2/runs/detect/train/weights/best.pt"
VIDEO_PATH = r"/hdd2/minhnv/CodingYOLOv12/TestingVideo/13.00.40-13.05.00[M][0@0][0] ver2.mp4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONF_THRESH = 0.1
IOU_THRESH = 0.45
FRAME_SIZE = (1280, 720)  # Giảm kích thước để tăng tốc xử lý

# Cấu hình ứng dụng
app.config.update({
    'model': None,
    'video_queue': Queue(maxsize=2),
    'processing': False,
    'inference_size': 640  # Kích thước đầu vào cho model
})

def initialize_resources():
    """Khởi tạo model và video reader khi khởi động ứng dụng"""
    try:
        print("\n=== Initializing Model ===")

        # Kiểm tra CUDA availability
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Using device: {DEVICE}")

        # Load model với verbose
        print(f"Loading model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        print("Model architecture loaded")
        print("Model info:", model)
        # Chuyển model sang device và half precision (nếu cần)
        model = model.to(DEVICE)
        if DEVICE == 'cuda':
            print("Converting model to FP16")
            model.half()  # Thử bỏ dòng này nếu có lỗi về precision

        # Kiểm tra model
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Model dtype: {next(model.parameters()).dtype}")
        app.config['model'] = model

        # Khởi tạo video capture thread
        def video_reader():
            cap = cv2.VideoCapture(VIDEO_PATH)
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.resize(frame, FRAME_SIZE)
                if app.config['video_queue'].qsize() < 2:
                    app.config['video_queue'].put(frame)
                else:
                    time.sleep(0.01)

        threading.Thread(target=video_reader, daemon=True).start()
        return True

    except Exception as e:
        print(f"Initialization failed: {e}")
        return False

def process_frame(frame):
    """Xử lý frame với YOLO model"""
    model = app.config['model']
    if frame is None or model is None:
        return frame

    try:
        # Resize và chuyển đổi màu sắc
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(
            img,
            imgsz=app.config['inference_size'],
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            device=DEVICE,
            verbose=False,
            stream=True,  # Sử dụng streaming mode
            half=(DEVICE == 'cuda')
        )

        # Vẽ bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame
    except Exception as e:
        print(f"Processing error: {e}")
        return frame

def generate_frames():
    """Tạo streaming frames với tốc độ ổn định"""
    target_fps = 25
    frame_interval = 1.0 / target_fps
    last_time = time.time()

    while True:
        try:
            # Điều chỉnh tốc độ khung hình
            while time.time() - last_time < frame_interval:
                time.sleep(0.001)
            
            frame = app.config['video_queue'].get()
            if app.config['processing']:
                frame = process_frame(frame)
            
            # Mã hóa và truyền frame
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            last_time = time.time()
        except Exception as e:
            print(f"Streaming error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_processing')
def toggle_processing():
    app.config['processing'] = not app.config['processing']
    return {'status': app.config['processing']}

if __name__ == '__main__':
    success = initialize_resources()
    if success:
        print("\n=== Server Startup ===")
        print(f"Model status: {'Loaded' if app.config['model'] else 'Failed'}")
        print(f"Video queue: {'Active' if not app.config['video_queue'].empty() else 'Inactive'}")
        app.run(host='0.0.0.0', port=8080, debug=True)  # Bật debug để xem lỗi
    else:
        print("Failed to initialize resources. Server not started.")
