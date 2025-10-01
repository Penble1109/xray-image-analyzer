import numpy as np
import cv2

def apply_window_level(img: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    img32 = img.astype(np.float32)
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness * 2.0
    out = img32 * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_clahe(img: np.ndarray) -> np.ndarray:
    #Local contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)

def apply_blur(img: np.ndarray) -> np.ndarray:
    #Denoise a bit; helps suppress speckle before edges.
    return cv2.GaussianBlur(img, (5, 5), 0)

def apply_edges(img: np.ndarray) -> np.ndarray:
    #Canny edges (uint8 mask 0/255)
    return cv2.Canny(img, 50, 150)