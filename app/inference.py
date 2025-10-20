from __future__ import annotations
import numpy as np
import onnxruntime as ort
from pathlib import Path
import cv2

_MODEL_PATH = Path("models/model.onnx")

def load_session(model_path: str | Path = _MODEL_PATH) -> ort.InferenceSession:
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(model_path), providers=providers)
    return sess

def infer_score(sess: ort.InferenceSession, img64: np.ndarray) -> float:
    """img64: uint8 (64x64) grayscale → returns float score in [0,1]."""
    x = img64.astype(np.float32) / 255.0
    x = x[None, None, :, :]  # [1,1,64,64]
    outputs = sess.run(["output"], {"input": x})
    y: np.ndarray = np.asarray(outputs[0])
    return float(y.squeeze().item())

def slide_and_heatmap(
    img: np.ndarray, sess: ort.InferenceSession, win: int = 64, stride: int = 16
) -> tuple[np.ndarray, list[tuple[int,int,int,int,float]]]:
    """
    Returns (heatmap float32 in [0,1] at image resolution, boxes [(x1,y1,x2,y2,score)])
    """
    h, w = img.shape
    ny = (h - win) // stride + 1
    nx = (w - win) // stride + 1
    heat = np.zeros((ny, nx), np.float32)
    boxes: list[tuple[int,int,int,int,float]] = []

    for iy, y in enumerate(range(0, h - win + 1, stride)):
        for ix, x in enumerate(range(0, w - win + 1, stride)):
            patch = img[y:y + win, x:x + win]
            s = infer_score(sess, cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA))
            heat[iy, ix] = s
            boxes.append((x, y, x + win, y + win, float(s)))

    # upscale heatmap to image size
    heat_up = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    return heat_up, boxes

def nms(boxes: list[tuple[int,int,int,int,float]], iou_thr: float = 0.3, score_thr: float = 0.8, top_k: int = 5):
    """Greedy non-max suppression."""
    boxes = [b for b in boxes if b[4] >= score_thr]
    boxes.sort(key=lambda b: b[4], reverse=True)
    keep: list[tuple[int,int,int,int,float]] = []
    def iou(a, b):
        ax1, ay1, ax2, ay2, _ = a; bx1, by1, bx2, by2, _ = b
        inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    for b in boxes:
        if all(iou(b, k) <= iou_thr for k in keep):
            keep.append(b)
        if len(keep) >= top_k:
            break
    return keep