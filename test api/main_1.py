import requests
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

def download_video(url, local_filename):
    # Gửi request để tải video
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Lưu video vào file local
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Video downloaded successfully: {local_filename}")
        return local_filename
    else:
        print("Failed to download video")
        return None

def play_video(video_path):
    model = YOLO('yolov8n.pt')

    # Khởi tạo capture video từ file video
    cap = cv2.VideoCapture(video_path)  # Thay bằng đường dẫn tới video của bạn

    # Thiết lập kích thước khung hình
    cap.set(3, 640)
    cap.set(4, 480)

    # Danh sách các nhãn phương tiện giao thông cần quan tâm (có thể tùy chỉnh)
    traffic_labels = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

    while True:
        # Đọc một frame từ video
        ret, img = cap.read()

        # Kiểm tra nếu video đã kết thúc
        if not ret:
            print("Kết thúc video")
            break

        # Dự đoán các đối tượng trong frame
        results = model.predict(img)

        # Vẽ các bounding box và nhãn cho các đối tượng được phát hiện
        for r in results:
            annotator = Annotator(img)
            boxes = r.boxes

            for box in boxes:
                # Lấy tọa độ bounding box và nhãn lớp
                b = box.xyxy[0]  # Tọa độ của bounding box (left, top, right, bottom)
                c = box.cls  # Chỉ số lớp của đối tượng
                label = model.names[int(c)]  # Lấy tên lớp

                # Kiểm tra nếu đối tượng là phương tiện giao thông
                if label in traffic_labels:
                    annotator.box_label(b, label)

            # Vẽ kết quả lên frame
            img = annotator.result()

        # Thay đổi kích thước frame và hiển thị
        img_resized = cv2.resize(img, (960, 540))
        cv2.imshow('YOLO V8 Traffic Detection', img_resized)

        # Thoát khi nhấn phím ' '
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    
    # Giải phóng các tài nguyên
    cap.release()
    cv2.destroyAllWindows()








if __name__ == "__main__":
    # URL video từ API
    video_url = 'http://127.0.0.1:5000/video/Le Van Viet.mp4'
    
    # Đường dẫn lưu video cục bộ
    local_video_path = 'downloaded_video.mp4'

    # Tải video
    downloaded_video = download_video(video_url, local_video_path)

    # Nếu video được tải thành công, tiến hành phát
    if downloaded_video:
        play_video(downloaded_video)
