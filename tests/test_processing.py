import numpy as np
import cv2
import math

from app.processing import (
    apply_window_level,
    apply_clahe,
    apply_blur,
    apply_edges,
)

def solid_img(value: int = 128, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h,w), value, np.uint8)

def checkerboard(h: int = 64, w: int = 64, block: int = 4) -> np.ndarray:
    y, x= np.indices((h, w))
    pattern = ((x// block) + (y//block)) % 2
    return (pattern * 255).astype(np.uint8)

def lap_energy(img: np.ndarray) -> float:
    k = cv2.Laplacian(img, cv2.CV_32F)
    return float(np.mean(np.square(k)))

#tests

def test_window_level_dtype_and_range():
    img = solid_img(128)
    out = apply_window_level(img, brightness=20, contrast=20)
    assert out.dtype == np.uint8
    assert out.shape == img.shape
    assert 0 <= int(out.min()) and int(out.max()) <= 255

def test_window_level_monotonic_brightness_increases_mean():
    img = solid_img(100)
    out_low  = apply_window_level(img, brightness=-20, contrast=0)
    out_high = apply_window_level(img, brightness=+20, contrast=0)
    assert out_high.mean() > out_low.mean()

def test_window_level_contrast_expands_range():
    ramp = np.tile(np.arange(0, 256, dtype=np.uint8), (64, 1))
    out = apply_window_level(ramp, brightness=0, contrast=+30)
    # Contrast should increase variance vs original
    assert out.var() > ramp.var()

def test_blur_reduces_high_frequency_energy():
    img = checkerboard()
    e_before = lap_energy(img)
    out = apply_blur(img)
    e_after = lap_energy(out)
    assert e_after < e_before  # blur should smooth edges

def test_edges_binary_mask_and_shape():
    img = checkerboard()
    e = apply_edges(img)
    assert e.shape == img.shape
    uniq = np.unique(e)
    assert set(uniq.tolist()).issubset({0, 255})

def test_clahe_increases_local_contrast_on_low_contrast_input():
    rng = np.random.default_rng(0)
    base = solid_img(100)
    noisy = np.clip(base.astype(np.int16) + rng.integers(-2, 3, base.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    before_std = noisy.std()
    out = apply_clahe(noisy)
    after_std = out.std()
    assert after_std >= before_std

def test_pipeline_is_clamped_and_uint8():
    img = checkerboard()
    wl = apply_window_level(img, brightness=30, contrast=30)
    cl = apply_clahe(wl)
    bl = apply_blur(cl)
    ed = apply_edges(bl)
    # overlay edges in white
    out = np.maximum(bl, (ed > 0) * 255).astype(np.uint8)
    assert out.dtype == np.uint8
    assert 0 <= int(out.min()) and int(out.max()) <= 255