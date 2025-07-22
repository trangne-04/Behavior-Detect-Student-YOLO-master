import json
from datetime import datetime
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Thêm import pandas để đọc file CSV
from ultralytics import YOLO
import sys
import logging
from pathlib import Path

# Thiết lập seed cho tính nhất quán
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Thiết lập logging đơn giản
def setup_logging(experiment_path):
    log_file = os.path.join(experiment_path, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Hàm kiểm tra GPU và xuất thông tin
def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)
        return True, f"{gpu_name}, Total: {gpu_memory:.2f}GB, Used: {memory_allocated:.2f}GB"
    return False, "GPU không khả dụng"

# Kiểm tra cấu hình hợp lệ
def validate_config(config):
    required_fields = ["name", "model_size", "epochs", "batch", "optimizer"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Thiếu tham số bắt buộc: {field}")
        
    if config["optimizer"] == "SGD" and "momentum" not in config:
        config["momentum"] = 0.9  # Giá trị mặc định nếu thiếu

# Đường dẫn đến dataset
data_yaml_path = "/hdd2/minhnv/CodingYOLOv12/Dataset/T-Student_FIT-DNU-update-6-split/data.yaml"

# Thiết lập seed
set_seed()

# Định nghĩa cấu hình hyperparameter
hyperparameter_configs = [
    {
        "name": "baseline",
        "model_size": "yolov12s",
        "epochs": 200,
        "patience": 35,
        "imgsz": 640,
        "batch": 32,
        "optimizer": "Adam",
        "lr0": 0.01,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "default",
    },
    {
        "name": "high_lr",
        "model_size": "yolov12s",
        "epochs": 200,
        "patience": 35,
        "imgsz": 640,
        "batch": 32,
        "optimizer": "Adam",
        "lr0": 0.03,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 5.0,
        "augmentation": "default",
    },
    {
        "name": "sgd_optimizer",
        "model_size": "yolov12s",
        "epochs": 200,
        "patience": 35,
        "imgsz": 640,
        "batch": 32,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.95,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "default",
    },
    {
        "name": "heavy_aug",
        "model_size": "yolov12s",
        "epochs": 200,
        "patience": 35,
        "imgsz": 640,
        "batch": 32,
        "optimizer": "Adam",
        "lr0": 0.01,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "heavy",
    },
    {
        "name": "light_aug",
        "model_size": "yolov12s",
        "epochs": 200,
        "patience": 30,
        "imgsz": 640,
        "batch": 16,
        "optimizer": "Adam",
        "lr0": 0.01,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "augmentation": "light",
    },
]

# Cấu hình tăng cường dữ liệu
augmentation_configs = {
    "default": {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 10.0, "translate": 0.1, "scale": 0.5,
        "shear": 2.0, "perspective": 0.0001,
        "flipud": 0.5, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.1, "copy_paste": 0.1, "erasing": 0.4,
    },
    "light": {
        "hsv_h": 0.01, "hsv_s": 0.5, "hsv_v": 0.3,
        "degrees": 5.0, "translate": 0.05, "scale": 0.2,
        "shear": 1.0, "perspective": 0.0,
        "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.7, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.2,
    },
    "heavy": {
        "hsv_h": 0.02, "hsv_s": 0.9, "hsv_v": 0.6,
        "degrees": 15.0, "translate": 0.2, "scale": 0.7,
        "shear": 3.0, "perspective": 0.001,
        "flipud": 0.5, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.3, "copy_paste": 0.3, "erasing": 0.6,
    }
}

# Thư mục lưu kết quả
base_save_dir = '/hdd2/minhnv/CodingYOLOv12/Behavior-Detect-Student-YOLO/StaticModels/yolov12Training-01042025_2342'
os.makedirs(base_save_dir, exist_ok=True)

# Khởi tạo cấu trúc kết quả
experiment_results = []
model_metrics = {}

# Kiểm tra GPU
has_gpu, gpu_info = get_gpu_info()
if not has_gpu:
    print("CẢNH BÁO: Không tìm thấy GPU. Quá trình huấn luyện sẽ chậm!")
else:
    print(f"Sử dụng GPU: {gpu_info}")

# Chạy các thí nghiệm
for config_idx, config in enumerate(hyperparameter_configs):
    print(f"\n{'='*50}")
    print(f"Thực hiện thí nghiệm {config_idx+1}/{len(hyperparameter_configs)}: {config['name']}")
    print(f"{'='*50}")
    
    # Kiểm tra cấu hình
    try:
        validate_config(config)
    except ValueError as e:
        print(f"Lỗi cấu hình: {e}")
        continue
    
    # Tạo đường dẫn lưu thí nghiệm
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    model_name = config["model_size"]
    experiment_name = f"{model_name}_{timestamp}_exp{config_idx+1}_{config['name']}"
    experiment_save_path = os.path.join(base_save_dir, experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)
    
    # Thiết lập logging
    setup_logging(experiment_save_path)
    logging.info(f"Bắt đầu thí nghiệm: {config['name']}")
    
    # Lưu cấu hình
    with open(os.path.join(experiment_save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Khởi tạo mô hình
    model = YOLO(f'{model_name}.yaml')
    
    # Thay đổi thư mục làm việc
    os.chdir(experiment_save_path)
    
    # Lấy cấu hình augmentation
    aug_config = augmentation_configs[config.get("augmentation", "default")]
    
    # Chuẩn bị tham số huấn luyện
    train_args = {
        "data": data_yaml_path,
        "epochs": config["epochs"],
        "patience": config["patience"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "optimizer": config["optimizer"],
        "lr0": config["lr0"],
        "lrf": config["lrf"],
        "weight_decay": config["weight_decay"],
        "warmup_epochs": config["warmup_epochs"],
        "pretrained": True,
        "dropout": 0.0,
        "save_period": -1,  # Không lưu các epoch trung gian
        "save": True,
        "cos_lr": True,  # Cosine LR schedule
        "conf": 0.25,  # Ngưỡng confidence
        "iou": 0.7,    # Ngưỡng IoU cho NMS
        **aug_config
    }
    
    # Thêm momentum nếu sử dụng SGD
    if config["optimizer"] == "SGD" and "momentum" in config:
        train_args["momentum"] = config["momentum"]
    
    # Log tham số huấn luyện
    logging.info(f"Tham số huấn luyện:\n{json.dumps(train_args, indent=2)}")
    
    # Huấn luyện mô hình
    print(f"Bắt đầu huấn luyện với cấu hình: {config['name']}")
    start_time = datetime.now()
    
    try:
        results = model.train(**train_args)
        training_time = datetime.now() - start_time
        
        # Lưu mô hình
        model_filename = f'{model_name}_{config["name"]}.pt'
        full_model_path = os.path.join(experiment_save_path, model_filename)
        model.save(full_model_path)
        
        # Tính kích thước mô hình
        model_size_mb = os.path.getsize(full_model_path) / (1024 * 1024)

        # Đọc metrics từ file CSV (giả sử file CSV có tên bắt đầu bằng "results")
        # Xác định đường dẫn file CSV cụ thể
        csv_path = os.path.join(experiment_save_path, "runs", "detect", "train", "results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file CSV tại: {csv_path}")
        df_metrics = pd.read_csv(csv_path)
                
        # Lấy epoch tốt nhất dựa trên mAP@0.5:0.95 (key trong CSV có thể là 'metrics/mAP50-95(B)')
        best_row = df_metrics.loc[df_metrics['metrics/mAP50-95(B)'].idxmax()]
        
        # Thu thập metrics từ file CSV
        metrics = {
            "map50": float(best_row['metrics/mAP50(B)']),
            "map": float(best_row['metrics/mAP50-95(B)']),
            "precision": float(best_row['metrics/precision(B)']),
            "recall": float(best_row['metrics/recall(B)']),
            "best_epoch": int(best_row['epoch']),
            "total_epochs": int(config["epochs"]),
            "training_time_seconds": training_time.total_seconds(),
            "model_size_mb": model_size_mb
        }

        # Nếu có trainer và thuộc tính best_epoch thì cập nhật lại
        if hasattr(results, 'trainer') and hasattr(results.trainer, 'best_epoch'):
            metrics["best_epoch"] = int(results.trainer.best_epoch)
            metrics["total_epochs"] = int(results.trainer.epoch)
        
        model_metrics[config["name"]] = metrics
        
        # Lưu kết quả thí nghiệm
        experiment_result = {
            "experiment_name": config["name"],
            "model_path": full_model_path,
            "config": config,
            "metrics": metrics
        }
        experiment_results.append(experiment_result)
        
        # Log kết quả quan trọng
        logging.info(f"Kết quả thí nghiệm {config['name']}:")
        logging.info(f"  mAP@0.5: {metrics['map50']:.4f}")
        logging.info(f"  mAP@0.5:0.95: {metrics['map']:.4f}")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  Best epoch: {metrics['best_epoch']}/{metrics['total_epochs']}")
        logging.info(f"  Thời gian huấn luyện: {training_time}")
        logging.info(f"  Kích thước mô hình: {metrics['model_size_mb']:.2f} MB")

        
    except Exception as e:
        logging.error(f"Lỗi trong thí nghiệm {config['name']}: {str(e)}")
        continue

# Lưu tất cả kết quả để so sánh
results_timestamp = datetime.now().strftime("%d%m%Y_%H%M")
final_results_path = os.path.join(base_save_dir, f'all_experiments_{results_timestamp}.json')
with open(final_results_path, 'w') as f:
    json.dump(experiment_results, f, indent=4)

print(f"\nĐã hoàn tất tất cả thí nghiệm. Kết quả được lưu tại {final_results_path}")

# Tìm mô hình tốt nhất dựa trên mAP
if experiment_results:
    # Đánh giá dựa trên mAP@0.5:0.95
    best_map_experiment = max(experiment_results, key=lambda x: x['metrics']['map'])
    # Đánh giá dựa trên mAP@0.5
    best_map50_experiment = max(experiment_results, key=lambda x: x['metrics']['map50'])
    
    print("\n=== Mô hình tốt nhất ===")
    print(f"Dựa trên mAP@0.5:0.95: {best_map_experiment['experiment_name']} (mAP={best_map_experiment['metrics']['map']:.4f})")
    print(f"Dựa trên mAP@0.5: {best_map50_experiment['experiment_name']} (mAP50={best_map50_experiment['metrics']['map50']:.4f})")
    
    # Tạo bảng so sánh
    print("\n=== Bảng so sánh các mô hình ===")
    header = f"{'Tên thí nghiệm':<15} | {'mAP@0.5:0.95':<12} | {'mAP@0.5':<10} | {'Precision':<10} | {'Recall':<10} | {'Thời gian (phút)':<16} | {'Kích thước (MB)':<16}"
    print(header)
    print("-" * len(header))
    
    for exp in experiment_results:
        m = exp['metrics']
        time_mins = m['training_time_seconds'] / 60
        print(f"{exp['experiment_name']:<15} | {m['map']:<12.4f} | {m['map50']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {time_mins:<16.1f} | {m['model_size_mb']:<16.1f}")
        
    # Vẽ biểu đồ so sánh các metric
    plt.figure(figsize=(12, 8))
    
    # Chuẩn bị dữ liệu để vẽ
    names = [exp['experiment_name'] for exp in experiment_results]
    map_values = [exp['metrics']['map'] for exp in experiment_results]
    map50_values = [exp['metrics']['map50'] for exp in experiment_results]
    precision_values = [exp['metrics']['precision'] for exp in experiment_results]
    recall_values = [exp['metrics']['recall'] for exp in experiment_results]
    
    # Vẽ các thanh theo nhóm
    x = np.arange(len(names))
    width = 0.2
    
    plt.bar(x - 1.5*width, map_values, width, label='mAP@0.5:0.95')
    plt.bar(x - 0.5*width, map50_values, width, label='mAP@0.5')
    plt.bar(x + 0.5*width, precision_values, width, label='Precision')
    plt.bar(x + 1.5*width, recall_values, width, label='Recall')
    
    plt.xlabel('Thí nghiệm')
    plt.ylabel('Giá trị')
    plt.title('So sánh hiệu suất các mô hình')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Lưu biểu đồ
    chart_path = os.path.join(base_save_dir, f'model_comparison_{results_timestamp}.png')
    plt.savefig(chart_path)
    print(f"\nĐã lưu biểu đồ so sánh tại: {chart_path}")
else:
    print("\nKhông có thí nghiệm nào hoàn thành thành công!")
