import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


def play_video_stream(video_url):
    model = YOLO('yolov8n.pt')

    # Khởi tạo capture video từ URL của video
    cap = cv2.VideoCapture(video_url)

    # Kiểm tra xem có thể mở video hay không
    if not cap.isOpened():
        print(f"Cannot open video from {video_url}")
        return

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
    # URL video từ API (sử dụng URL trực tiếp)
    video_url = 'http://127.0.0.1:5000/video/Le Van Viet.mp4'

    # Phát và xử lý video trực tiếp từ URL
    play_video_stream(video_url)
