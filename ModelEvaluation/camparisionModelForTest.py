import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from datetime import datetime
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import csv
from tabulate import tabulate

# Thiết lập logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'evaluation_{datetime.now().strftime("%d%m%Y_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

# Thiết lập path cho dữ liệu test và thư mục mô hình
BASE_MODELS_DIR = '/hdd2/minhnv/CodingYOLOv12/Behavior-Detect-Student-YOLO/StaticModels/yolov12Training-31032025_1833/'
TEST_DATA_PATH = '/hdd2/minhnv/CodingYOLOv12/Dataset/T-Student_FIT-DNU-update-1/test/images/'  # Đường dẫn tới tập test images
DATA_YAML = '/hdd2/minhnv/CodingYOLOv12/Dataset/T-Student_FIT-DNU-update-1/data.yaml'  # Đường dẫn tới file data.yaml

# Thư mục để lưu kết quả đánh giá
EVAL_RESULTS_DIR = os.path.join(BASE_MODELS_DIR, 'evaluation_results')
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

# Thiết lập logging
log_file = setup_logging(EVAL_RESULTS_DIR)
logging.info("Bắt đầu đánh giá mô hình YOLOv12 trên tập TEST")

# Tìm tất cả các mô hình đã lưu
def find_models():
    model_files = []
    # Tìm tất cả các file .pt trong thư mục BASE_MODELS_DIR và các thư mục con
    for root, dirs, files in os.walk(BASE_MODELS_DIR):
        for file in files:
            if file.endswith('.pt'):
                model_path = os.path.join(root, file)
                # Lấy tên thí nghiệm từ tên file
                exp_name = file.replace('.pt', '')
                model_files.append({
                    'path': model_path,
                    'name': exp_name
                })
    
    logging.info(f"Tìm thấy {len(model_files)} mô hình để đánh giá")
    return model_files

# Lấy tên tệp cấu hình đi kèm với mô hình
def get_config_for_model(model_path):
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

# Đánh giá một mô hình trên tập test
def evaluate_model(model_info):
    model_path = model_info['path']
    model_name = model_info['name']
    
    logging.info(f"Đánh giá mô hình: {model_name}")
    
    try:
        # Tải mô hình
        model = YOLO(model_path)
        
        # Lấy cấu hình
        config = get_config_for_model(model_path)
        
        # Chạy đánh giá trên tập test
        results = model.val(data=DATA_YAML, split='test')
        
        # Tạo thư mục cho kết quả của mô hình này
        model_result_dir = os.path.join(EVAL_RESULTS_DIR, model_name)
        os.makedirs(model_result_dir, exist_ok=True)
        
        # Phân tích class-wise metrics (nếu có)
        class_names = results.names
        class_metrics = {}
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            confusion_matrix = results.confusion_matrix.matrix
            for i, name in enumerate(class_names):
                if i < confusion_matrix.shape[0]:
                    tp = confusion_matrix[i, i]
                    fp_sum = confusion_matrix[:, i].sum() - tp
                    fn_sum = confusion_matrix[i, :].sum() - tp
                    precision = tp / (tp + fp_sum) if (tp + fp_sum) > 0 else 0
                    recall = tp / (tp + fn_sum) if (tp + fn_sum) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    class_metrics[name] = {
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1)
                    }
        
        # Đọc metrics từ file CSV (đường dẫn cho tập test)
        csv_path = os.path.join(model_result_dir, "runs", "detect", "test", "results.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file CSV tại: {csv_path}")
        df_metrics = pd.read_csv(csv_path)
        best_row = df_metrics.loc[df_metrics['metrics/mAP50-95(B)'].idxmax()]
        
        # Thu thập metrics từ file CSV
        metrics = {
            "map50": float(best_row['metrics/mAP50(B)']),
            "map50_95": float(best_row['metrics/mAP50-95(B)']),
            "precision": float(best_row['metrics/precision(B)']),
            "recall": float(best_row['metrics/recall(B)']),
            "best_epoch": int(best_row['epoch']),
            "total_epochs": int(config["epochs"]) if config else 0,
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Nếu có trainer và thuộc tính best_epoch thì cập nhật lại
        if hasattr(results, 'trainer') and hasattr(results.trainer, 'best_epoch'):
            metrics["best_epoch"] = int(results.trainer.best_epoch)
            metrics["total_epochs"] = int(results.trainer.epoch)
        
        # Thêm các thông số khác từ kết quả đánh giá (nếu có)
        metrics["speed_ms"] = float(results.speed['inference'])
        metrics["speed_fps"] = 1000 / float(results.speed['inference'])
        
        # Lưu kết quả metrics vào file JSON
        with open(os.path.join(model_result_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Thực hiện inference để visualize kết quả trên một số ảnh test
        vis_results = model.predict(
            source=TEST_DATA_PATH,
            conf=0.25,
            iou=0.7,
            max_det=300,
            save=True,
            save_conf=True,
            save_txt=True,
            project=model_result_dir,
            name='predictions'
        )
        
        logging.info(f"Đã đánh giá xong mô hình: {model_name}")
        logging.info(f"  mAP@0.5: {metrics['map50']:.4f}")
        logging.info(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        logging.info(f"  Precision: {metrics['precision']:.4f}")
        logging.info(f"  Recall: {metrics['recall']:.4f}")
        logging.info(f"  Tốc độ: {metrics['speed_ms']:.2f}ms ({metrics['speed_fps']:.1f} FPS)")
        
        # Gộp kết quả đánh giá và cấu hình (nếu có)
        metrics["model_name"] = model_name
        metrics["model_path"] = model_path
        metrics["config"] = config
        
        return metrics
    
    except Exception as e:
        logging.error(f"Lỗi khi đánh giá mô hình {model_name}: {str(e)}")
        return None

# Chức năng chính
def main():
    model_list = find_models()
    
    if not model_list:
        logging.error("Không tìm thấy mô hình nào để đánh giá!")
        return
    
    results = []
    for model_info in tqdm(model_list, desc="Đánh giá mô hình"):
        result = evaluate_model(model_info)
        if result:
            results.append(result)
    
    if not results:
        logging.error("Không có mô hình nào được đánh giá thành công!")
        return
    
    logging.info(f"Đánh giá thành công {len(results)}/{len(model_list)} mô hình")
    
    # Lưu kết quả tổng hợp
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    summary_path = os.path.join(EVAL_RESULTS_DIR, f'model_comparison_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Tạo DataFrame và lưu bảng so sánh
    df = pd.DataFrame([{
        'Model': r['model_name'],
        'mAP@0.5': r['map50'],
        'mAP@0.5:0.95': r['map50_95'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'Speed (ms)': r['speed_ms'],
        'FPS': r['speed_fps']
    } for r in results])
    
    df = df.sort_values('mAP@0.5:0.95', ascending=False)
    csv_path = os.path.join(EVAL_RESULTS_DIR, f'model_comparison_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("BẢNG SO SÁNH HIỆU SUẤT CÁC MÔ HÌNH TRÊN TẬP TEST")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='pretty', floatfmt='.4f'))
    print("="*80)
    
    # Tạo biểu đồ so sánh
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    metrics_df = df[['Model', 'mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall']]
    metrics_df = pd.melt(metrics_df, id_vars=['Model'], var_name='Metric', value_name='Value')
    sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df)
    plt.title('So sánh hiệu suất các mô hình')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(2, 1, 2)
    sns.barplot(x='Model', y='FPS', data=df, palette='viridis')
    plt.title('So sánh tốc độ inference (FPS)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    chart_path = os.path.join(EVAL_RESULTS_DIR, f'model_comparison_chart_{timestamp}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    create_detailed_report(results, timestamp)
    
    logging.info(f"Đã lưu bảng so sánh tại: {csv_path}")
    logging.info(f"Đã lưu biểu đồ so sánh tại: {chart_path}")
    print(f"\nĐã lưu kết quả đánh giá tại: {EVAL_RESULTS_DIR}")

# Tạo báo cáo chi tiết
def create_detailed_report(results, timestamp):
    report_path = os.path.join(EVAL_RESULTS_DIR, f'detailed_report_{timestamp}.html')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Báo cáo đánh giá mô hình YOLOv12</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .header {{ background-color: #4CAF50; color: white; padding: 10px; }}
            .model-section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; }}
            .metrics {{ margin-top: 10px; }}
            .class-metrics {{ margin-top: 20px; }}
            .highlight {{ background-color: #ffffcc; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Báo cáo đánh giá mô hình YOLOv12</h1>
            <p>Ngày tạo: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
        </div>
        
        <h2>Bảng so sánh hiệu suất</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>mAP@0.5</th>
                <th>mAP@0.5:0.95</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Speed (ms)</th>
                <th>FPS</th>
            </tr>
    """
    
    sorted_results = sorted(results, key=lambda x: x['map50_95'], reverse=True)
    for r in sorted_results:
        html_content += f"""
            <tr>
                <td>{r['model_name']}</td>
                <td>{r['map50']:.4f}</td>
                <td>{r['map50_95']:.4f}</td>
                <td>{r['precision']:.4f}</td>
                <td>{r['recall']:.4f}</td>
                <td>{r['speed_ms']:.2f}</td>
                <td>{r['speed_fps']:.1f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Chi tiết từng mô hình</h2>
    """
    
    for r in sorted_results:
        html_content += f"""
        <div class="model-section">
            <h3>{r['model_name']}</h3>
            <p><strong>Đường dẫn:</strong> {r['model_path']}</p>
            <div class="metrics">
                <h4>Hiệu suất chung:</h4>
                <ul>
                    <li>mAP@0.5: {r['map50']:.4f}</li>
                    <li>mAP@0.5:0.95: {r['map50_95']:.4f}</li>
                    <li>Precision: {r['precision']:.4f}</li>
                    <li>Recall: {r['recall']:.4f}</li>
                    <li>Tốc độ: {r['speed_ms']:.2f}ms ({r['speed_fps']:.1f} FPS)</li>
                </ul>
            </div>
        """
        if r['config']:
            html_content += "<div class='config'><h4>Cấu hình:</h4><ul>"
            for key, value in r['config'].items():
                html_content += f"<li><strong>{key}:</strong> {value}</li>"
            html_content += "</ul></div>"
        
        if r.get('class_metrics'):
            html_content += """
            <div class="class-metrics">
                <h4>Hiệu suất theo lớp:</h4>
                <table>
                    <tr>
                        <th>Lớp</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                    </tr>
            """
            for class_name, metrics in r['class_metrics'].items():
                html_content += f"""
                    <tr>
                        <td>{class_name}</td>
                        <td>{metrics['precision']:.4f}</td>
                        <td>{metrics['recall']:.4f}</td>
                        <td>{metrics['f1']:.4f}</td>
                    </tr>
                """
            html_content += "</table></div>"
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"Đã tạo báo cáo chi tiết tại: {report_path}")

if __name__ == "__main__":
    main()
