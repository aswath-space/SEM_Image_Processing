"""
semx/grains.py
--------------
Core functions for Stage 1 grain analysis.

Design goals:
- Accuracy over speed
- Clear, testable functions
- Rich comments for maintainability
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import re
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from skimage import exposure, filters, measure, morphology, segmentation
import matplotlib.pyplot as plt


# ---------------- Metadata parsing ----------------

def parse_metadata(txt: str) -> Dict[str, Any]:
    """
    Parse SEM metadata text for fields needed by Stage 1.
    Expected keys:
      - image_name (str)
      - data_w, data_h (int)  # DataSize=WxH
      - px_um (float)         # µm / pixel
      - micron_marker_um (float|None)

    Raises: ValueError if any required field is missing.
    """
    m_img = re.search(r'(?im)^\s*ImageName\s*=\s*(\S+)\s*$', txt)
    if not m_img:
        raise ValueError("Metadata missing 'ImageName='.")
    image_name = m_img.group(1).strip()

    m_sz = re.search(r'(?im)^\s*DataSize\s*=\s*(\d+)\s*x\s*(\d+)\s*$', txt)
    if not m_sz:
        raise ValueError("Metadata missing 'DataSize=WxH'.")
    data_w, data_h = int(m_sz.group(1)), int(m_sz.group(2))

    m_px = re.search(r'(?im)^\s*PixelSize\s*=\s*([0-9]*\.?[0-9]+)\s*$', txt)
    if not m_px:
        raise ValueError("Metadata missing 'PixelSize='.")
    px_nm = float(m_px.group(1))
    px_um = px_nm / 1000.0

    m_marker = re.search(r'(?im)^\s*MicronMarker\s*=\s*([0-9]*\.?[0-9]+)\s*$', txt)
    micron_marker_um = float(m_marker.group(1))/1000.0 if m_marker else None

    return {
        "image_name": image_name,
        "data_w": data_w,
        "data_h": data_h,
        "px_um": px_um,
        "micron_marker_um": micron_marker_um,
    }


# ---------------- Image I/O ----------------

def read_image_gray_float(file_bytes: bytes) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Decode image bytes, return grayscale float32 in [0,1] *after* normalization? No.
    We return raw grayscale in float32 (0..max) and the (H,W) shape.
    Normalization is handled by preprocess_gray.
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError("Could not decode image.")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)
    h, w = im.shape
    return im, (h, w)


# ---------------- Failsafes ----------------

def verify_pairing_or_raise(uploaded_name: str, image_shape: Tuple[int,int], meta: Dict[str, Any]) -> None:
    """
    Enforce that uploaded image ↔ metadata belong together.
    Conditions:
      - filenames must match (case-insensitive)
      - resolution must match DataSize
    Raise ValueError with clear message if any check fails.
    """
    if uploaded_name.lower() != meta["image_name"].lower():
        raise ValueError(
            f"Filename mismatch:\n"
            f"- Uploaded image: {uploaded_name}\n"
            f"- Metadata ImageName=: {meta['image_name']}"
        )
    h, w = image_shape
    if (w != meta["data_w"]) or (h != meta["data_h"]):
        raise ValueError(
            f"Resolution mismatch:\n"
            f"- Actual image: {w}x{h}\n"
            f"- Metadata DataSize=: {meta['data_w']}x{meta['data_h']}"
        )


# ---------------- Preprocessing ----------------

def preprocess_gray(img: np.ndarray) -> np.ndarray:
    """
    Accuracy-first preprocessing:
      - normalize to [0,1] per-image
      - CLAHE to enhance boundaries (clip_limit ~0.02)
      - bilateral filter for edge-preserving denoise
    Returns gray float32 in [0,1].
    """
    x = (img - img.min()) / max(1e-6, (img.max() - img.min()))
    x = exposure.equalize_adapthist(x, clip_limit=0.02)
    x = cv2.bilateralFilter((x*255).astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=7).astype(np.float32)/255.0
    return x


# ---------------- Deep QC: scale-bar detection ----------------

def _detect_scale_bar_roi(gray01: np.ndarray, roi_bottom_pct: int) -> Tuple[np.ndarray, int]:
    """Crop bottom ROI for scale-bar search; return (roi_gray, top_row_index)."""
    H, W = gray01.shape
    top = max(0, H - int(H * roi_bottom_pct / 100))
    return gray01[top:H, :].copy(), top


def _find_scale_bar_component(bin_roi: np.ndarray, expected_px: float | None, min_aspect: int) -> Dict[str, any]:
    """
    Search labeled components for a long, horizontal bar.
    Return best candidate dict: {found, width, bbox, score}
    """
    best = {"found": False, "width": None, "bbox": None, "score": None}
    lab = measure.label(bin_roi)
    for p in measure.regionprops(lab):
        minr, minc, maxr, maxc = p.bbox
        height = maxr - minr
        width = maxc - minc
        if width < 10 or height < 2:
            continue
        aspect = width / max(height, 1e-6)
        if aspect < min_aspect:
            continue
        if expected_px and expected_px > 0:
            closeness = abs(width - expected_px) / expected_px
        else:
            closeness = 1.0 / (1.0 + width)
        score = closeness + (0.25 if aspect < (min_aspect * 1.5) else 0.0)
        if (not best["found"]) or (score < best["score"]):
            best.update({"found": True, "width": float(width), "bbox": (int(minr), int(minc), int(maxr), int(maxc)), "score": float(score)})
    return best


def run_deep_qc_scale_bar(
    gray01: np.ndarray,
    px_um: float,
    micron_marker_um: float | None,
    roi_bottom_pct: int = 16,
    min_aspect: int = 12,
) -> Dict[str, any]:
    """
    Detect the scale bar near the bottom of the image, measure its pixel length,
    and compare to expected length from MicronMarker/PixelSize.

    Returns dict:
      - found (bool)
      - expected_px (float|None)
      - measured_px (float|None)
      - error_pct (float|None)
      - overlay (H,W,3) uint8, green box drawn around detected bar (if any)
    """
    H, W = gray01.shape
    base_overlay = (gray01 * 255).astype(np.uint8)
    overlay = cv2.cvtColor(base_overlay, cv2.COLOR_GRAY2BGR)

    expected_px = None
    if micron_marker_um is not None and px_um > 0:
        expected_px = micron_marker_um / px_um

    # Crop ROI
    roi, top = _detect_scale_bar_roi(gray01, roi_bottom_pct)
    roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)

    best = {"found": False}
    # Try both polarities (white/black bars)
    for polarity in ("white", "black"):
        work = roi_blur if polarity == "white" else 1.0 - roi_blur
        thr = filters.threshold_otsu(work)
        bin_roi = work > thr
        # Morphology that favors long horizontal rectangles
        bin_roi = morphology.binary_closing(bin_roi, footprint=morphology.rectangle(3, 25))
        bin_roi = morphology.binary_opening(bin_roi, footprint=morphology.rectangle(3, 7))
        candidate = _find_scale_bar_component(bin_roi, expected_px=expected_px, min_aspect=min_aspect)
        if candidate["found"] and (not best.get("found") or candidate["score"] < best.get("score", 1e9)):
            best = candidate

    report = {"found": False, "expected_px": expected_px, "measured_px": None, "error_pct": None, "overlay": overlay}
    if best.get("found"):
        minr, minc, maxr, maxc = best["bbox"]
        # Draw bbox on overlay in full-coordinates
        cv2.rectangle(overlay, (minc, top + minr), (maxc, top + maxr), (0, 255, 0), 2)
        report["found"] = True
        report["measured_px"] = float(best["width"])
        if expected_px:
            report["error_pct"] = abs(best["width"] - expected_px) / expected_px * 100.0
        report["overlay"] = overlay
    return report


# ---------------- Segmentation ----------------

def segment_grains(
    gray01: np.ndarray,
    mask_bottom_pct: int = 16,
    min_area_px: int = 80,
    hole_area_px: int = 80,
    h_prominence: float = 0.2,
) -> np.ndarray:
    """
    Segment grains from a preprocessed [0,1] grayscale SEM image.

    Steps:
      - mask bottom legend strip
      - Otsu threshold
      - remove small objects & fill small holes
      - EDT + watershed with h-maxima to separate touching grains
    Returns integer label map (0=background).
    """
    H, W = gray01.shape
    keep = np.ones_like(gray01, dtype=bool)
    bb = int(H * mask_bottom_pct / 100)
    if bb > 0:
        keep[-bb:, :] = False

    x_masked = gray01.copy()
    x_masked[~keep] = np.median(gray01[keep])

    thr = filters.threshold_otsu(x_masked)
    mask = x_masked > thr

    if hole_area_px > 0:
        mask = morphology.remove_small_holes(mask, area_threshold=int(hole_area_px))
    if min_area_px > 0:
        mask = morphology.remove_small_objects(mask, min_size=int(min_area_px))

    dist = ndi.distance_transform_edt(mask)
    hmax = morphology.h_maxima(dist, h_prominence) if h_prominence > 0 else (dist == dist)
    markers = measure.label(hmax)
    labels = segmentation.watershed(-dist, markers, mask=mask)
    return labels


# ---------------- Measurement ----------------

def measure_grains(labels: np.ndarray, px_um: float, min_area_px: int = 80) -> pd.DataFrame:
    """
    Measure per-grain features and convert to physical units.
    """
    rows = []
    for p in measure.regionprops(labels):
        if p.area < min_area_px:
            continue
        rows.append({
            "label": int(p.label),
            "area_px": int(p.area),
            "equiv_diam_um": float(p.equivalent_diameter * px_um),
            "major_axis_um": float(p.major_axis_length * px_um),
            "minor_axis_um": float(p.minor_axis_length * px_um),
            "aspect_ratio": float(p.major_axis_length / max(p.minor_axis_length, 1e-6)),
            "orientation_rad": float(p.orientation),
        })
    return pd.DataFrame(rows)


# ---------------- Overlay ----------------

def make_size_overlay(
    gray01: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    metric: str,
    p_lo: int = 5,
    p_hi: int = 95,
    alpha: float = 0.5,
):
    """
    Colorize grains by a chosen metric and blend on top of grayscale.

    Returns:
      overlay_rgb: (H,W,3) uint8 in RGB order
      vmin, vmax: color scale bounds after percentile clipping
    """
    H, W = labels.shape
    size_map = np.zeros((H, W), dtype=np.float32)

    if metric not in df.columns:
        # No data to colorize; return grayscale
        base = (gray01 * 255).astype(np.uint8)
        return cv2.cvtColor(base, cv2.COLOR_GRAY2RGB), 0.0, 1.0

    # Fill per-grain value
    val_by_label = dict(zip(df["label"].astype(int), df[metric].astype(float)))
    for lbl, val in val_by_label.items():
        size_map[labels == lbl] = val

    vals = size_map[size_map > 0]
    if vals.size == 0:
        base = (gray01 * 255).astype(np.uint8)
        return cv2.cvtColor(base, cv2.COLOR_GRAY2RGB), 0.0, 1.0

    vmin = float(np.percentile(vals, p_lo))
    vmax = float(np.percentile(vals, p_hi))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    cmap = plt.cm.get_cmap()  # perceptually uniform default
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colored = cmap(norm(size_map))[:, :, :3]  # drop alpha

    base = np.stack([gray01, gray01, gray01], axis=-1)
    blended = (1 - alpha) * base + alpha * colored
    overlay_rgb = (blended * 255).astype(np.uint8)
    return overlay_rgb, vmin, vmax