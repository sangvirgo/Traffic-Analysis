from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2

# Tải mô hình YOLOv8n
model = YOLO('yolov8n.pt')

# Khởi tạo capture video từ file video
cap = cv2.VideoCapture('traffic.mp4')  # Thay bằng đường dẫn tới video của bạn

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