import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from trafficUI import Ui_MainWindow  # Import giao diện từ file trafficUI.py
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Tải mô hình YOLOv8n
        self.model = YOLO('yolov8n.pt')

        # Khởi tạo video capture
        self.cap = cv2.VideoCapture('../uploads/Le Van Viet.mp4')  # Thay bằng đường dẫn tới video của bạn

        # Cài đặt timer để lấy frame từ video
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Cập nhật mỗi 30ms

        self.video_paths= {
            "Lê Văn Việt": "../uploads/Le Van Viet.mp4",
            "Võ Văn Ngân": "../uploads/Vo Van Ngan.mp4",
            "Võ Văn Kiệt": "../uploads/Vo Van Kiet.mp4",
            "Hai Bà Trưng": "../uploads/Hai Ba Trung.mp4",
            "Võ Chí Công": "../uploads/Vo Chi Cong.mp4",
        }

        # Xử lý khi chọn vị trí từ combobox
        self.ui.locationComboBox.currentIndexChanged.connect(self.on_location_changed)

        # Danh sách các nhãn phương tiện giao thông cần quan tâm (có thể tùy chỉnh)
        self.traffic_labels = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']


    def on_location_changed(self):
        if self.cap:
            self.cap.release()
            self.timer.stop()
        location=self.ui.locationComboBox.currentText()

        if location in self.video_paths:
            self.cap = cv2.VideoCapture(self.video_paths[location])
            self.timer.start(30)

    def update_frame(self):
        # Đọc một frame từ video
        ret, frame = self.cap.read()

        # Kiểm tra nếu video đã kết thúc
        if not ret:
            print("Kết thúc video")
            self.cap.release()
            self.timer.stop()
            return

        # Gọi hàm xuatloi để xử lý frame và nhận về frame đã xử lý
        processed_frame = xuatloi(frame)

        # Chuyển đổi frame thành định dạng QImage để hiển thị trên QLabel
        img_resized = cv2.resize(processed_frame, (791, 571))  # Kích thước phù hợp với QLabel
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.ui.videoLabel.setPixmap(QtGui.QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        # Khi đóng cửa sổ, giải phóng tài nguyên
        self.cap.release()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
