from PySide6.QtWidgets import QApplication, QMainWindow,QLabel,QToolBar, QFileDialog,QVBoxLayout, QWidget,QStatusBar, QSlider, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QAction, QPixmap, QImage
from.io_utils import save_outputs

import sys
import cv2
import numpy as np
from .processing import apply_window_level, apply_clahe, apply_blur, apply_edges
from .inference import load_session, slide_and_heatmap, nms

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

        act_export = QAction("Export", self)
        act_export.triggered.connect(self.on_export)
        toolbar.addAction(act_export)
        
        self.act_analyze = QAction("Analyze", self)
        self.act_analyze.triggered.connect(self.on_analyze)
        toolbar.addAction(self.act_analyze)

        toolbar.addSeparator()
        # checkable toggles
        self.act_clahe = QAction("CLAHE", self); self.act_clahe.setCheckable(True); self.act_clahe.toggled.connect(self.refresh); toolbar.addAction(self.act_clahe)
        self.act_blur  = QAction("Blur",   self); self.act_blur.setCheckable(True);  self.act_blur.toggled.connect(self.refresh);  toolbar.addAction(self.act_blur)
        self.act_edges = QAction("Edges",  self); self.act_edges.setCheckable(True); self.act_edges.toggled.connect(self.refresh); toolbar.addAction(self.act_edges)

        self.act_heatmap = QAction("Heatmap", self); self.act_heatmap.setCheckable(True); self.act_heatmap.toggled.connect(self.refresh); toolbar.addAction(self.act_heatmap)

        


        self.ort_sess = None           # lazy-load on first analyze
        self.heatmap: np.ndarray | None = None
        self.boxes: list[tuple[int,int,int,int,float]] = []
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
        #window level
        out = apply_window_level(self.img, self.brightness.value(), self.contrast.value())

        if self.act_clahe.isChecked():
            out = apply_clahe(out)
        if self.act_blur.isChecked():
            out = apply_blur(out)
        if self.act_edges.isChecked():
            edges = apply_edges(out)                    # 0/255
            out = np.maximum(out, (edges > 0) * 255).astype(np.uint8)  # overlay in white
        return out
    def refresh(self):
        img = self.current_view()
        if img is None:
            return

        # start from the processed view
        overlay = img.copy()

        # 1) heatmap overlay (brighten hot regions)
        if getattr(self, "act_heatmap", None) and self.act_heatmap.isChecked() and self.heatmap is not None:
            hm = (np.clip(self.heatmap, 0, 1) * 255).astype(np.uint8)
            overlay = cv2.max(overlay, hm)

        # 2) boxes + scores
        if getattr(self, "boxes", None):
            for (x1, y1, x2, y2, s) in self.boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), 255, 2)
                cv2.putText(overlay, f"{s:.2f}", (x1 + 2, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA)

        # 3) draw to QLabel (use overlay, not the original)
        overlay = np.ascontiguousarray(overlay)       # ensure contiguous memory
        h, w = overlay.shape
        qimg = QImage(overlay.data, w, h, overlay.strides[0], QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)

        self.view.setPixmap(
            pix.scaled(
                self.view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        # status
        b = self.brightness.value() if hasattr(self, "brightness") else 0
        c = self.contrast.value()   if hasattr(self, "contrast")   else 0
        self.status.showMessage(
            f"Loaded {w}x{h} | B:{b} C:{c} | "
            f"CLAHE:{getattr(self, 'act_clahe', None) and self.act_clahe.isChecked()} "
            f"Blur:{getattr(self, 'act_blur', None) and self.act_blur.isChecked()} "
            f"Edges:{getattr(self, 'act_edges', None) and self.act_edges.isChecked()}"
        )

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh()
    
    def on_export(self):
        img = self.current_view() if hasattr(self, "current_view") else None

        if img is None:
            self.status.showMessage("Nothing to export - opan an image first")
            return 
        flags = {
        "clahe":  bool(getattr(self, "act_clahe", None) and self.act_clahe.isChecked()),
        "blur":   bool(getattr(self, "act_blur", None) and self.act_blur.isChecked()),
        "edges":  bool(getattr(self, "act_edges", None) and self.act_edges.isChecked()),
         }

        try:
            img_path, json_path = save_outputs(
                img=img,
                flags=flags,
                brightness=int(self.brightness.value()) if hasattr(self, "brightness") else 0,
                contrast=int(self.contrast.value())     if hasattr(self, "contrast") else 0,
         )
            self.status.showMessage(f"Exported → {img_path.split('/')[-1]} + report.json", 4000)
        except Exception as e:
            self.status.showMessage(f"Export failed: {e}", 5000)
    def on_analyze(self):
        if self.img is None:
            self.status.showMessage("Open an image first.", 3000); return
        try:
            if self.ort_sess is None:
                self.ort_sess = load_session()
        except Exception as e:
            self.status.showMessage(f"Model load failed: {e}", 5000); return

    # Run sliding-window on the **current processed** view (so user controls matter)
        proc = self.img.copy()
        if proc is None:
            self.status.showMessage("No processed image to analyze.", 3000); return

        heat, boxes = slide_and_heatmap(proc, self.ort_sess, win=64, stride=32) #prev 16
        picks = nms(boxes, iou_thr=0.35, score_thr=0.95, top_k=3)

        self.heatmap = heat
        self.boxes = picks
        self.status.showMessage(
            f"Analyze: top score {max([b[4] for b in picks], default=0):.2f} | {len(picks)} boxes"
        )
        self.refresh()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
