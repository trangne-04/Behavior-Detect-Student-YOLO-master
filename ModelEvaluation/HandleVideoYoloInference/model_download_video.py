import cv2
import os
import torch
from ultralytics import YOLO
from datetime import datetime
import time

# Cấu hình đường dẫn
MODEL_PATH = r"/hdd2/minhnv/CodingYOLOv12/Behavior-Detect-Student-YOLO/StaticModels/yolov12s13032025_000207_ver-dataset6/_ver2/runs/detect/train/weights/best.pt"
VIDEO_PATH = r"/hdd2/minhnv/CodingYOLOv12/TestingVideo/13.00.40-13.05.00[M][0@0][0] ver2.mp4"
OUTPUT_DIR = r"/hdd2/minhnv/CodingYOLOv12/TestingVideo/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

# Cấu hình xử lý
CONF_THRESH = 0.3
IOU_THRESH = 0.45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cấu hình hiển thị
SHOW_PREVIEW = False  # Đặt True nếu muốn hiển thị preview
COLORS = {
    0: (0, 227, 255),   # computer (#e300ff → (227, 0, 255) → (0, 227, 255))
    1: (255, 255, 0),   # phone (#FFFF00 → (255, 255, 0) → (255, 255, 0)) (không đổi)
    2: (255, 49, 15),   # raising_hand (#31ff0f → (49, 255, 15) → (255, 49, 15))
    3: (232, 0, 255),   # sleeping (#00e8ff → (0, 232, 255) → (232, 0, 255))
    4: (0, 254, 86),    # turning_left (#FE0056 → (254, 0, 86) → (0, 254, 86))
    5: (128, 255, 0),   # turning_right (#FF8000 → (255, 128, 0) → (128, 255, 0))
    6: (122, 14, 254),  # using_computer (#0E7AFE → (14, 122, 254) → (122, 14, 254))
    7: (34, 134, 255),  # using_phone (#8622FF → (134, 34, 255) → (34, 134, 255))
    8: (171, 255, 171), # writing (#FFABAB → (255, 171, 171) → (171, 255, 171))
}

# DEFAULT_COLOR = (0, 255, 0)

def main():
    print("Bắt đầu chương trình...")
    
    # Kiểm tra sự tồn tại của file đầu vào
    if not os.path.exists(VIDEO_PATH):
        print(f"Không tìm thấy file video: {VIDEO_PATH}")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Không tìm thấy file model: {MODEL_PATH}")
        return

    # Tải mô hình YOLO
    try:
        print(f"Đang tải mô hình từ {MODEL_PATH}...")
        model = YOLO(MODEL_PATH).to(DEVICE)
        print(f"Mô hình đã được tải lên {DEVICE}")
        class_names = model.names
        print(f"Các lớp: {class_names}")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        return

    # Mở video input
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Không thể mở video: {VIDEO_PATH}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Thông số video: {width}x{height}, FPS: {fps}, tổng frame: {total_frames}")
    except Exception as e:
        print(f"Lỗi khi mở video: {e}")
        return

    # Khởi tạo video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Không thể tạo video đầu ra: {OUTPUT_PATH}")
            cap.release()
            return
    except Exception as e:
        print(f"Lỗi khi tạo video writer: {e}")
        cap.release()
        return

    # Xử lý video
    print("Bắt đầu xử lý video...")
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # Chạy inference trên frame
            results = model.predict(
                source=frame,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                device=DEVICE,
                verbose=False,
                show=False  # Không hiển thị trực tiếp từ model
            )
            
            # Vẽ kết quả lên frame
            annotated_frame = frame.copy()
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    color = COLORS.get(cls_id, DEFAULT_COLOR)
                    label = f"{class_names.get(cls_id, 'Class ' + str(cls_id))} {conf:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Hiển thị thông tin tiến trình
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            progress_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(annotated_frame, progress_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            out.write(annotated_frame)
            
            if SHOW_PREVIEW:
                cv2.imshow("Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_count % 10 == 0 or frame_count == 1:
                percent = (frame_count / total_frames) * 100
                print(f"Tiến trình: {frame_count}/{total_frames} frames ({percent:.1f}%) - FPS: {avg_fps:.1f}")
                
    except KeyboardInterrupt:
        print("\nĐã dừng xử lý bởi người dùng.")
    except Exception as e:
        print(f"\nLỗi khi xử lý video: {e}")
    finally:
        cap.release()
        out.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print("\n" + "-"*60)
        print(f"Hoàn tất! Đã xử lý {frame_count}/{total_frames} frames.")
        print(f"Thời gian xử lý: {total_time:.2f} giây")
        print(f"FPS trung bình: {avg_fps:.2f}")
        print(f"Video đã được lưu tại: {OUTPUT_PATH}")
        print("-"*60)

if __name__ == "__main__":
    main()
