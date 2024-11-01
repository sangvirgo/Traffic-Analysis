import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from trafficUI import Ui_MainWindow
from detect import xuatloi
import cv2
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtWidgets import QListWidgetItem

class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Khởi tạo video và timer
        self.cap = cv2.VideoCapture('../uploads/Le Van Viet.mp4')


        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_skip = 0  # Sử dụng frame_skip để kiểm soát số khung hình bỏ qua
        self.timer.start(10)  # 30 FPS

        # Đường dẫn video cho các địa điểm khác nhau
        self.video_paths = {
            "Lê Văn Việt": "../uploads/Le Van Viet.mp4",
            "Võ Văn Ngân": "../uploads/Vo Van Ngan.mp4",
            "Võ Văn Kiệt": "../uploads/Vo Van Kiet.mp4",
            "Hai Bà Trưng": "../uploads/Hai Ba Trung.mp4",
            "Võ Chí Công": "../uploads/Vo Chi Cong.mp4",
        }

        # Xử lý khi chọn vị trí từ combobox
        self.ui.locationComboBox.currentIndexChanged.connect(self.on_location_changed)

    def on_location_changed(self):
        # Khi thay đổi địa điểm, dừng video hiện tại và chuyển sang video mới
        if self.cap:
            self.cap.release()
            self.timer.stop()
        location = self.ui.locationComboBox.currentText()

        if location in self.video_paths:
            self.cap = cv2.VideoCapture(self.video_paths[location])
            self.timer.start(10)

    def update_frame(self):
        ret, img = self.cap.read()

        if not ret:
            print("Kết thúc video")
            self.cap.release()
            self.timer.stop()
            return

        # Chỉ xử lý mỗi giây một khung hình
        if self.frame_skip % 10 == 0:
            img = cv2.resize(img, (680, 500)) # Giảm độ phân giải
            img = xuatloi(img, self)  # Hàm xử lý khung hình vi phạm

            # Chuyển đổi khung hình để hiển thị
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qt_img = QtGui.QImage(img_rgb.data, img_rgb.shape[1], img_rgb.shape[0], QtGui.QImage.Format_RGB888)
            self.ui.videoLabel.setPixmap(QtGui.QPixmap.fromImage(qt_img))

        self.frame_skip += 1

    def log_message(self, mes):
        self.ui.listWidget.addItem(mes)
        self.ui.listWidget.scrollToBottom()

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