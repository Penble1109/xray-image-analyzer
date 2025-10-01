import numpy as np

def apply_window_level(img: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    img32 = img.astype(np.float32)
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness * 2.0
    out = img32 * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)