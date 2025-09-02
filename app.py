import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from semx.grains import (
    parse_metadata,
    read_image_gray_float,
    verify_pairing_or_raise,
    preprocess_gray,
    run_deep_qc_scale_bar,
    segment_grains,
    measure_grains,
    make_size_overlay,
)

st.set_page_config(page_title="SEM Grain Analyzer", layout="wide")
st.title("🔬 SEM Grain Size Analyzer — Stage 1 (GitHub/Cloud)")

# ---------- Sidebar: QC + segmentation + overlay ----------
with st.sidebar:
    st.header("Deep QC (Scale Bar)")
    deep_qc_enabled = st.checkbox("Enable Deep QC", value=True)
    block_on_qc_fail = st.checkbox("Block on Deep QC fail", value=True)
    qc_error_threshold_pct = st.slider("Error threshold (%)", 0.5, 5.0, 2.0, 0.5)
    qc_roi_bottom_pct = st.slider("Bottom ROI height (%)", 8, 30, 16, 1)
    qc_min_aspect = st.slider("Min bar aspect (W/H)", 6, 30, 12, 1)

    st.header("Segmentation")
    mask_bottom_pct = st.slider("Mask bottom legend (%)", 8, 30, 16, 1)
    min_area_px = st.number_input("Min object area (px)", 10, 5000, 80, 10)
    hole_area_px = st.number_input("Fill holes up to (px)", 0, 5000, 80, 10)
    h_prominence = st.slider("Watershed h-max prominence", 0.0, 3.0, 0.2, 0.05)

    st.header("Overlay")
    overlay_metric = st.selectbox(
        "Colorize by",
        ["equiv_diam_um", "major_axis_um", "minor_axis_um", "aspect_ratio"],
        index=0,
    )
    clip_lo, clip_hi = st.slider("Percentile clip", 0, 100, (5, 95), 1)
    overlay_alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, 0.05)

    st.header("Plots")
    hist_bins = st.slider("Histogram bins", 10, 200, 40, 5)
    hist_logy = st.checkbox("Histogram log-y", value=False)
    show_box = st.checkbox("Show box plot", value=True)

# ---------- Uploads ----------
st.header("1) Upload SEM image + metadata")
img_file = st.file_uploader("SEM image (.jpg/.jpeg/.png/.tif/.tiff)", type=["jpg","jpeg","png","tif","tiff"])
meta_file = st.file_uploader("Metadata (.txt)", type=["txt"])

if not img_file or not meta_file:
    st.info("Upload BOTH an SEM image and its corresponding metadata (.txt).")
    st.stop()

# ---------- Parse metadata & load image ----------
meta_txt = meta_file.read().decode(errors="ignore")
meta = parse_metadata(meta_txt)  # {image_name, data_w, data_h, px_um, micron_marker_um}

img_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
img_gray, (h, w) = read_image_gray_float(img_bytes)

# ---------- Failsafes ----------
try:
    verify_pairing_or_raise(
        uploaded_name=Path(img_file.name).name,
        image_shape=(h, w),
        meta=meta,
    )
except ValueError as e:
    st.error(f"❌ Failsafe failed:\n\n{e}")
    st.stop()

st.success(
    f"Pair verified ✔  |  {meta['image_name']}  |  {w}x{h}px  |  {meta['px_um']*1000:.2f} nm/px"
    + (f"  |  Scale bar (metadata): {meta['micron_marker_um']:.3f} µm" if meta.get("micron_marker_um") else "")
)

# ---------- Preprocess ----------
x = preprocess_gray(img_gray)  # normalized [0,1], CLAHE, denoise

# ---------- Deep QC (optional) ----------
if deep_qc_enabled:
    qc = run_deep_qc_scale_bar(
        gray01=x,
        px_um=meta["px_um"],
        micron_marker_um=meta.get("micron_marker_um"),
        roi_bottom_pct=qc_roi_bottom_pct,
        min_aspect=qc_min_aspect,
    )
    st.subheader("Deep QC — Scale Bar")
    cqc1, cqc2 = st.columns(2)
    with cqc1: st.image(x, clamp=True, caption="Preprocessed image", use_container_width=True)
    with cqc2: st.image(qc["overlay"], clamp=True, caption="Detected bar (green box)", use_container_width=True)

    if qc["found"]:
        st.metric("Expected length (px)", f"{qc['expected_px']:.1f}")
        st.metric("Measured length (px)", f"{qc['measured_px']:.1f}")
        if qc["error_pct"] is not None:
            st.metric("Scale-bar error (%)", f"{qc['error_pct']:.2f}%")
            if block_on_qc_fail and qc["error_pct"] > qc_error_threshold_pct:
                st.error(f"❌ Deep QC failed: {qc['error_pct']:.2f}% > {qc_error_threshold_pct:.2f}% — Blocking.")
                st.stop()
    else:
        if block_on_qc_fail:
            st.error("❌ Deep QC failed: could not detect a plausible scale bar — Blocking.")
            st.stop()
        else:
            st.warning("⚠️ Deep QC could not detect a scale bar — Proceeding per your settings.")

# ---------- Segmentation & measurement ----------
labels = segment_grains(
    gray01=x,
    mask_bottom_pct=max(mask_bottom_pct, qc_roi_bottom_pct),
    min_area_px=min_area_px,
    hole_area_px=hole_area_px,
    h_prominence=h_prominence,
)
df = measure_grains(labels=labels, px_um=meta["px_um"], min_area_px=min_area_px)

# ---------- Overlay ----------
overlay_rgb, vmin, vmax = make_size_overlay(
    gray01=x,
    labels=labels,
    df=df,
    metric=overlay_metric,
    p_lo=clip_lo,
    p_hi=clip_hi,
    alpha=overlay_alpha,
)

st.header("2) Results")
c1, c2 = st.columns(2)
with c1:
    st.caption("Preprocessed (contrast-enhanced)")
    st.image(x, clamp=True, use_container_width=True)
with c2:
    st.caption(f"Size overlay: {overlay_metric} (clipped {clip_lo}–{clip_hi}%)")
    st.image(overlay_rgb, use_container_width=True)
st.caption(f"Color scale: vmin={vmin:.4g}  vmax={vmax:.4g}  ({'µm' if overlay_metric!='aspect_ratio' else 'a.u.'})")

st.subheader("Per-grain table (preview)")
st.dataframe(df.head(12), use_container_width=True)

if not df.empty:
    st.subheader("Summary")
    st.metric("Grain count", len(df))
    st.metric("Mean equiv. diameter (µm)", f"{df['equiv_diam_um'].mean():.3f}")
    st.metric("Median equiv. diameter (µm)", f"{df['equiv_diam_um'].median():.3f}")

# ---------- Plots ----------
st.subheader("3) Distribution plots")
if df.empty or overlay_metric not in df.columns:
    st.info("No data to plot yet. Try adjusting segmentation parameters.")
else:
    series = df[overlay_metric].dropna().astype(float)
    unit = "µm" if overlay_metric != "aspect_ratio" else "a.u."
    nice_name = {
        "equiv_diam_um": "Equivalent diameter",
        "major_axis_um": "Major axis length",
        "minor_axis_um": "Minor axis length",
        "aspect_ratio": "Aspect ratio",
    }.get(overlay_metric, overlay_metric)
    if not series.empty:
        # Histogram
        fig_h, ax_h = plt.subplots(figsize=(7, 4), dpi=160)
        ax_h.hist(series.values, bins=hist_bins, edgecolor="black")
        ax_h.set_xlabel(f"{nice_name} ({unit})")
        ax_h.set_ylabel("Count")
        ax_h.set_title(f"{nice_name} — Histogram")
        if hist_logy:
            ax_h.set_yscale("log")
        mean_v, median_v = float(series.mean()), float(series.median())
        ax_h.axvline(mean_v, linestyle="--", linewidth=1, label=f"Mean = {mean_v:.3g} {unit}")
        ax_h.axvline(median_v, linestyle=":",  linewidth=1, label=f"Median = {median_v:.3g} {unit}")
        ax_h.legend()
        st.pyplot(fig_h, clear_figure=True)
        buf_h = io.BytesIO()
        fig_h.tight_layout(); fig_h.savefig(buf_h, format="png", bbox_inches="tight")
        st.download_button(
            "⬇️ Download histogram (PNG)",
            data=buf_h.getvalue(),
            file_name=f"{Path(meta['image_name']).stem}_{overlay_metric}_hist.png",
            mime="image/png",
        )
        plt.close(fig_h)

        # Box plot
        if show_box:
            fig_b, ax_b = plt.subplots(figsize=(6, 3.5), dpi=160)
            ax_b.boxplot(series.values, vert=False, showmeans=True, meanline=True)
            ax_b.set_xlabel(f"{nice_name} ({unit})")
            ax_b.set_yticklabels([""])
            ax_b.set_title(f"{nice_name} — Box plot")
            st.pyplot(fig_b, clear_figure=True)
            buf_b = io.BytesIO()
            fig_b.tight_layout(); fig_b.savefig(buf_b, format="png", bbox_inches="tight")
            st.download_button(
                "⬇️ Download box plot (PNG)",
                data=buf_b.getvalue(),
                file_name=f"{Path(meta['image_name']).stem}_{overlay_metric}_box.png",
                mime="image/png",
            )
            plt.close(fig_b)

with st.expander("QC enforced"):
    st.markdown(
        "- **Filename pairing**: uploaded image must equal `ImageName=` in metadata (case-insensitive).\n"
        "- **Resolution pairing**: actual pixels must equal `DataSize=` in metadata.\n"
        "- **Scale (Deep QC, optional)**: detect scale bar and compare measured px to `MicronMarker/PixelSize`.\n"
        "- **Legend masking**: bottom strip masked before segmentation to avoid artifacts.\n"
        "- **Overlay**: robust percentile clipping for stable color mapping."
    )
