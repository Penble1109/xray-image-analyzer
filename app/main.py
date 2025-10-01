from PySide6.QtWidgets import QApplication, QMainWindow,QLabel,QToolBar, QFileDialog,QVBoxLayout, QWidget,QStatusBar, QSlider, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction, QPixmap, QImage
import sys
import cv2
import numpy as np
from .processing import apply_window_level

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-ray Image Analyzer")
        self.setMinimumSize(800,600)
        

        #status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")
        
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)

        #sliders
        self.brightness = QSlider(Qt.Orientation.Horizontal); self.brightness.setRange(-50, 50); self.brightness.setValue(0)
        self.contrast   = QSlider(Qt.Orientation.Horizontal); self.contrast.setRange(-50, 50); self.contrast.setValue(0)
        self.brightness.valueChanged.connect(self.refresh)
        self.contrast.valueChanged.connect(self.refresh)
        sliders = QHBoxLayout()
        sliders.addWidget(QLabel("Brightness"))
        sliders.addWidget(self.brightness)
        sliders.addWidget(QLabel("Contrast"))
        sliders.addWidget(self.contrast)

        lay = QVBoxLayout(); lay.addWidget(self.view)
        lay.addLayout(sliders)
        root = QWidget(); root.setLayout(lay)
        self.setCentralWidget(root)

        #toolbar
        toolbar = QToolBar("ToolBar") #toolbar for options
        self.addToolBar(toolbar)
        open_action = QAction(QIcon.fromTheme("document-open"), "Open File", self)
        open_action.triggered.connect(self.open_file)
        toolbar.addAction(open_action)


        self.img: np.ndarray | None = None

        


    def open_file(self):
        path, _ = QFileDialog.getOpenFileName( self, "Open Image", filter="Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")

        if not path :
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.status.showMessage("Failed to load image", 2000)
            return
        
        if img.ndim == 3:
            #converts to grayscale if needed
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.dtype == np.uint16:
            #16-bit grayscale tto 8 bit for display
            img = cv2.convertScaleAbs(img,alpha= 255.0/65535.0)
        
        img = np.ascontiguousarray(img)
        self.img = img
        self.refresh()
    def current_view(self):
        if self.img is None:
            return None
        return apply_window_level(self.img, self.brightness.value(), self.contrast.value())
    def refresh(self):
        img = self.current_view()
        if img is None:
            return
        h, w = img.shape
        #qimg = np.ascontiguousarray(img)
        qimg = QImage(img.data, w ,h, img.strides[0], QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self.view.setPixmap(
            pix.scaled(self.view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.status.showMessage(f"Loaded {w}x{h} | B:{self.brightness.value()}  C:{self.contrast.value()}") 

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
