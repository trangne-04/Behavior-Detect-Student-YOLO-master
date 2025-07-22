import os
import subprocess
import math

def get_video_duration(file_path):
    """Lấy thời lượng của video sử dụng ffmpeg"""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', file_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def split_video(input_path, output_folder, segment_length_minutes=5):
    """
    Cắt video thành nhiều phần, mỗi phần có độ dài được chỉ định sử dụng ffmpeg
    
    Args:
        input_path: Đường dẫn đến file video đầu vào
        output_folder: Thư mục để lưu các video được cắt
        segment_length_minutes: Độ dài của mỗi đoạn video (tính bằng phút)
    """
    # Chuyển đổi độ dài từ phút sang giây
    segment_length = segment_length_minutes * 60
    
    # Tạo thư mục output nếu nó chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Lấy thời lượng của video
    duration = get_video_duration(input_path)
    
    # Tính số lượng đoạn
    num_segments = math.ceil(duration / segment_length)
    
    print(f"Tổng thời lượng video: {duration:.2f} giây")
    print(f"Chia thành {num_segments} đoạn, mỗi đoạn {segment_length_minutes} phút")
    
    # Cắt và lưu từng đoạn
    for i in range(num_segments):
        start_time = i * segment_length
        # Không cần hạn chế end_time vì ffmpeg sẽ tự dừng khi kết thúc video
        
        # Tạo tên file output
        output_path = os.path.join(output_folder, f"part_{i+1:03d}.mp4")
        
        # Command để cắt video
        cmd = [
            'ffmpeg', '-i', input_path, '-ss', str(start_time),
            '-t', str(segment_length), '-c:v', 'copy', '-c:a', 'copy',
            output_path
        ]
        
        print(f"Đang xử lý đoạn {i+1}/{num_segments}: từ {start_time:.2f}s -> {output_path}")
        
        # Thực thi lệnh
        subprocess.run(cmd)
        
    print(f"Hoàn thành việc cắt video thành {num_segments} đoạn!")

# Đường dẫn đến file video và thư mục đầu ra
input_video = "/hdd2/minhnv/CodingYOLOv12/VideoTesting/D11_20250303093031.mp4"
output_folder = "/hdd2/minhnv/CodingYOLOv12/VideoTesting/OutputVideoD11"

# Gọi hàm để cắt video
split_video(input_video, output_folder, segment_length_minutes=8)