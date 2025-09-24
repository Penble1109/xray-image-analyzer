from PySide6.QtWidgets import QApplication, QMainWindow,QLabel,QToolBar, QFileDialog,QVBoxLayout, QWidget,QStatusBar
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction, QPixmap, QImage
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-ray Image Analyzer")
        self.setMinimumSize(800,600)
        label = QLabel("Hello, X-Ray!")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        self.setCentralWidget(label)

        self.statusBar().showMessage("Ready") #status bar

        toolbar = QToolBar("ToolBar") #toolbar for options
        self.addToolBar(toolbar)
        open_action = QAction(QIcon.fromTheme("document-open"), "Open File", self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)

    def open_file(self):
        print("Open placeholder")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
