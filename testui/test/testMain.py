import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from Test import Ui_MainWindow

class MainWindow:
    def __init__(self):
        self.main_window = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_window)

        # Chọn tab đầu tiên làm tab mặc định khi ứng dụng khởi chạy
        self.uic.tabWidget.setCurrentIndex(0)

        self.uic.Home_button.clicked.connect(lambda: self.uic.tabWidget.setCurrentIndex(0))
        self.uic.S1.clicked.connect(lambda: self.uic.tabWidget.setCurrentIndex(1))
        self.uic.S2.clicked.connect(lambda: self.uic.tabWidget.setCurrentIndex(2))
        self.uic.S3.clicked.connect(lambda: self.uic.tabWidget.setCurrentIndex(3))
    def show(self):
        self.main_window.show()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())