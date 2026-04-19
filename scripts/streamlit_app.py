from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch

# Add project src directory for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdr_project.model import SmallUNetHDR
from hdr_project.utils import tonemap_for_display


@st.cache_resource
def load_single_shot_model(weights_path: str, base_channels: int, device_str: str) -> SmallUNetHDR:
    """Load single-shot model once and cache it across reruns."""
    device = torch.device(device_str)
    model = SmallUNetHDR(base=base_channels).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def read_uploaded_image(uploaded_file) -> np.ndarray:
    """Decode uploaded bytes into BGR uint8 image."""
    arr = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {uploaded_file.name}")
    return img


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG output")
    return bytes(buf)


def to_hdr_bytes(hdr: np.ndarray) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "result.hdr"
        cv2.imwrite(str(p), hdr.astype(np.float32))
        return p.read_bytes()


def infer_single_shot(model: SmallUNetHDR, image_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Run AI single-shot model and return predicted HDR (float32)."""
    h, w = image_bgr.shape[:2]

    # Keep dimensions safe for UNet down/up sampling.
    safe_h = max(4, ((h + 3) // 4) * 4)
    safe_w = max(4, ((w + 3) // 4) * 4)
    if (safe_h, safe_w) != (h, w):
        resized = cv2.resize(image_bgr, (safe_w, safe_h), interpolation=cv2.INTER_AREA)
    else:
        resized = image_bgr

    x = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_log = model(x).squeeze(0).cpu().numpy()

    pred_log = np.transpose(pred_log, (1, 2, 0))
    pred_hdr = np.expm1(np.clip(pred_log, 0.0, None)).astype(np.float32)

    if (safe_h, safe_w) != (h, w):
        pred_hdr = cv2.resize(pred_hdr, (w, h), interpolation=cv2.INTER_LINEAR)

    return pred_hdr


def estimate_linear_luminance(image_bgr: np.ndarray) -> float:
    """Estimate scene-average luminance in linear light."""
    x = np.power(np.clip(image_bgr.astype(np.float32) / 255.0, 0.0, 1.0), 2.2)
    lum = 0.0722 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.2126 * x[:, :, 2]
    return float(np.mean(lum))


def normalize_single_shot_input(
    image_bgr: np.ndarray,
    target_luma: float,
    min_gain: float = 0.35,
    max_gain: float = 1.35,
) -> tuple[np.ndarray, float, float]:
    """Shift input exposure toward the training distribution before inference."""
    mean_luma = estimate_linear_luminance(image_bgr)
    gain = float(np.clip(target_luma / (mean_luma + 1e-6), min_gain, max_gain))

    x = np.power(np.clip(image_bgr.astype(np.float32) / 255.0, 0.0, 1.0), 2.2)
    x = np.clip(x * gain, 0.0, 1.0)
    normalized = np.power(x, 1.0 / 2.2)
    normalized_u8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    return normalized_u8, gain, mean_luma


def tonemap_for_display_robust(
    hdr: np.ndarray,
    gamma: float = 2.2,
    white_percentile: float = 99.5,
    black_percentile: float = 2.0,
) -> np.ndarray:
    """Tone-map HDR with percentile black/white points for better contrast."""
    hdr = np.clip(hdr, 0.0, None)
    mapped = np.log1p(hdr)
    black_point = float(np.percentile(mapped, black_percentile))
    white_point = float(np.percentile(mapped, white_percentile))
    mapped = (mapped - black_point) / (white_point - black_point + 1e-8)
    mapped = np.clip(mapped, 0.0, 1.0)
    mapped = np.power(mapped, 1.0 / gamma)
    return (mapped * 255.0).astype(np.uint8)


def neutralize_color_cast(image_bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Reduce global color cast with robust gray-world balancing."""
    if strength <= 0.0:
        return image_bgr

    img = image_bgr.astype(np.float32)
    means = np.array([
        np.mean(img[:, :, 0]),
        np.mean(img[:, :, 1]),
        np.mean(img[:, :, 2]),
    ], dtype=np.float32)

    target = float(np.mean(means))
    gains = target / (means + 1e-6)
    gains = np.clip(gains, 0.75, 1.35)

    # Blend toward neutral gains so correction stays natural.
    gains = 1.0 + (gains - 1.0) * float(np.clip(strength, 0.0, 1.0))

    balanced = img.copy()
    balanced[:, :, 0] *= gains[0]
    balanced[:, :, 1] *= gains[1]
    balanced[:, :, 2] *= gains[2]
    return np.clip(balanced, 0.0, 255.0).astype(np.uint8)


def apply_warm_temperature(image_bgr: np.ndarray, strength: float = 0.12) -> np.ndarray:
    """Apply a subtle warm shift (+red, +yellow) to reduce cool/blue appearance."""
    if strength <= 0.0:
        return image_bgr

    s = float(np.clip(strength, 0.0, 0.35))
    img = image_bgr.astype(np.float32)

    # BGR order: reduce blue, slightly boost green, boost red.
    img[:, :, 0] *= (1.0 - 0.22 * s)
    img[:, :, 1] *= (1.0 + 0.10 * s)
    img[:, :, 2] *= (1.0 + 0.25 * s)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def denoise_low_light(image_bgr: np.ndarray, strength: int) -> np.ndarray:
    if strength <= 0:
        return image_bgr
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, strength, strength, 7, 21)


def recover_details(image_bgr: np.ndarray, amount: float, sigma: float = 1.1) -> np.ndarray:
    if amount <= 0.0:
        return image_bgr
    src = image_bgr.astype(np.float32)
    blur = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(src, 1.0 + amount, blur, -amount, 0.0)
    return np.clip(sharp, 0.0, 255.0).astype(np.uint8)


def model_retinex_lime_plus(image_bgr: np.ndarray) -> np.ndarray:
    """Retinex/LIME-like single image model (fixed best preset)."""
    x = denoise_low_light(image_bgr, strength=10).astype(np.float32) / 255.0

    illum0 = np.max(x, axis=2)
    illum = cv2.bilateralFilter((illum0 * 255.0).astype(np.uint8), 9, 40, 25).astype(np.float32) / 255.0
    illum = np.clip(illum, 0.04, 1.0)

    reflectance = x / illum[:, :, None]
    target_illum = np.power(illum, 0.65)
    enhanced = np.clip(reflectance * target_illum[:, :, None], 0.0, 1.0)

    shadow_mask = np.power(1.0 - illum, 1.8)
    alpha = np.clip(0.85 * shadow_mask, 0.0, 1.0)[:, :, None]
    blended = np.clip((1.0 - alpha) * x + alpha * enhanced, 0.0, 1.0)

    out = np.maximum(blended, x * 0.94)
    out_u8 = (out * 255.0).astype(np.uint8)
    return recover_details(out_u8, amount=0.18)


def model_msrcp(image_bgr: np.ndarray) -> np.ndarray:
    """MSRCP-style enhancement with stronger visibility and color preservation."""
    x = denoise_low_light(image_bgr, strength=8).astype(np.float32) / 255.0
    x = np.clip(x, 0.0, 1.0)

    intensity = np.mean(x, axis=2)
    retinex = np.zeros_like(intensity)
    for sigma in (15.0, 80.0, 250.0):
        blur = cv2.GaussianBlur(intensity, (0, 0), sigmaX=sigma, sigmaY=sigma)
        retinex += np.log1p(intensity) - np.log1p(blur)
    retinex /= 3.0

    low, high = np.percentile(retinex, [2.0, 98.0])
    retinex = np.clip((retinex - low) / (high - low + 1e-6), 0.0, 1.0)
    retinex = np.power(retinex, 0.88)

    color = x / (np.sum(x, axis=2, keepdims=True) + 1e-6)
    out = np.clip(retinex[:, :, None] * color * 3.0, 0.0, 1.0)
    out_u8 = (out * 255.0).astype(np.uint8)
    return recover_details(out_u8, amount=0.14)


def model_single_fusion_hdr(image_bgr: np.ndarray) -> np.ndarray:
    """Stronger synthetic bracket + Mertens fusion model (fixed preset)."""
    x = denoise_low_light(image_bgr, strength=12).astype(np.float32) / 255.0
    lin = np.power(x, 2.2)

    ev_steps = [-3.0, -1.5, 0.0, 1.5, 3.0]
    stack = []
    for ev in ev_steps:
        exp_lin = np.clip(lin * (2.0 ** ev), 0.0, 1.0)
        exp_srgb = np.power(exp_lin, 1.0 / 2.2)
        stack.append((exp_srgb * 255.0).astype(np.uint8))

    fusion = cv2.createMergeMertens().process(stack)
    fusion = np.clip(np.nan_to_num(fusion, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    fusion = np.power(fusion, 0.88)

    y = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
    shadow_mask = np.power(1.0 - y, 1.45)
    alpha = np.clip(0.15 + 0.95 * shadow_mask, 0.0, 1.0)[:, :, None]

    blended = np.clip((1.0 - alpha) * x + alpha * fusion * 1.05, 0.0, 1.0)
    out = np.maximum(blended, x * 0.90)
    out_u8 = (out * 255.0).astype(np.uint8)
    return recover_details(out_u8, amount=0.22)


def model_cinematic_grade(image_bgr: np.ndarray) -> np.ndarray:
    """Cinematic photo grade with neutral color balance (less yellow cast)."""
    x_u8 = denoise_low_light(image_bgr, strength=8)

    # Local contrast shaping on luminance channel.
    lab = cv2.cvtColor(x_u8, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_enh = clahe.apply(l_chan)
    graded = cv2.cvtColor(cv2.merge([l_enh, a_chan, b_chan]), cv2.COLOR_LAB2BGR).astype(np.float32)

    # Subtle warmth only (previous values were too warm/yellow on many scenes).
    graded[:, :, 0] *= 0.98
    graded[:, :, 1] *= 1.00
    graded[:, :, 2] *= 1.03
    graded = np.clip(graded, 0.0, 255.0).astype(np.uint8)

    # Mild saturation boost while keeping tones natural.
    hsv = cv2.cvtColor(graded, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.06, 0.0, 255.0)
    graded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Gray-world balancing to reduce global yellow/green tint.
    b_mean = float(np.mean(graded[:, :, 0]))
    g_mean = float(np.mean(graded[:, :, 1]))
    r_mean = float(np.mean(graded[:, :, 2]))
    mean_gray = (b_mean + g_mean + r_mean) / 3.0
    if b_mean > 1e-6 and g_mean > 1e-6 and r_mean > 1e-6:
        graded_f = graded.astype(np.float32)
        graded_f[:, :, 0] *= (mean_gray / b_mean)
        graded_f[:, :, 1] *= (mean_gray / g_mean)
        graded_f[:, :, 2] *= (mean_gray / r_mean)
        graded = np.clip(graded_f, 0.0, 255.0).astype(np.uint8)

    return recover_details(graded, amount=0.14)


def run_model(
    model_name: str,
    image_bgr: np.ndarray,
    device: torch.device,
    weights_path: Path,
    base_channels: int,
    single_shot_target_luma: float,
    single_shot_exposure_comp: float,
    single_shot_gamma: float,
    single_shot_white_percentile: float,
    single_shot_black_percentile: float,
    single_shot_denoise: int,
    single_shot_detail_boost: float,
    single_shot_color_neutralize: float,
    single_shot_warm_shift: float,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float] | None]:
    """Run selected model and return preview image, optional HDR tensor, and debug metadata."""
    if model_name == "single_shot_ai":
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        model = load_single_shot_model(str(weights_path), base_channels, device.type)
        normalized, exposure_gain, input_luma = normalize_single_shot_input(
            image_bgr,
            target_luma=single_shot_target_luma,
        )
        pred_hdr = infer_single_shot(model, normalized, device)

        # Partially compensate for preprocessing gain so output does not become over-bright.
        pred_hdr = pred_hdr / (max(exposure_gain, 1e-6) ** single_shot_exposure_comp)

        preview = tonemap_for_display_robust(
            pred_hdr,
            gamma=single_shot_gamma,
            white_percentile=single_shot_white_percentile,
            black_percentile=single_shot_black_percentile,
        )

        # Requested order: denoise before enhancement, then white-balance, then slight warm shift.
        preview = denoise_low_light(preview, strength=single_shot_denoise)
        if single_shot_detail_boost > 0.0:
            preview = recover_details(preview, amount=single_shot_detail_boost)
        preview = neutralize_color_cast(preview, strength=single_shot_color_neutralize)
        preview = apply_warm_temperature(preview, strength=single_shot_warm_shift)

        meta = {
            "input_luma": input_luma,
            "preprocess_gain": exposure_gain,
        }
        return preview, pred_hdr, meta

    if model_name == "retinex_lime_plus":
        return model_retinex_lime_plus(image_bgr), None, None

    if model_name == "msrcp":
        return model_msrcp(image_bgr), None, None

    if model_name == "single_fusion_hdr":
        return model_single_fusion_hdr(image_bgr), None, None

    if model_name == "cinematic_grade":
        return model_cinematic_grade(image_bgr), None, None

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    st.set_page_config(page_title="HDR Image Converter", layout="wide")
    st.title("HDR Image Converter")
    st.caption("Upload one image, choose one model, and compare original vs enhanced output.")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    with st.sidebar:
        st.subheader("Model")
        model_name = st.selectbox(
            "Choose HDR model",
            [
                "single_shot_ai",
                "retinex_lime_plus",
                "msrcp",
                "single_fusion_hdr",
                "cinematic_grade",
            ],
        )

        single_shot_target_luma = 0.07
        single_shot_exposure_comp = 0.90
        single_shot_gamma = 2.0
        single_shot_white_percentile = 98.8
        single_shot_black_percentile = 3.0
        single_shot_denoise = 5
        single_shot_detail_boost = 0.12
        single_shot_color_neutralize = 0.65
        single_shot_warm_shift = 0.12

        if model_name == "single_shot_ai":
            st.subheader("Single-Shot Quality")
            single_shot_target_luma = st.slider("Target input luminance", 0.04, 0.16, 0.07, 0.01)
            single_shot_exposure_comp = st.slider("Exposure compensation", 0.0, 1.0, 0.90, 0.05)
            single_shot_gamma = st.slider("Preview gamma", 1.6, 2.8, 2.0, 0.1)
            single_shot_black_percentile = st.slider("Preview black percentile", 0.0, 10.0, 3.0, 0.5)
            single_shot_white_percentile = st.slider("Preview white percentile", 95.0, 99.9, 98.8, 0.1)
            single_shot_denoise = st.slider("Denoise before enhancement", 0, 15, 5, 1)
            single_shot_detail_boost = st.slider("Preview detail boost", 0.0, 0.30, 0.12, 0.01)
            single_shot_color_neutralize = st.slider("Color neutralize", 0.0, 1.0, 0.65, 0.05)
            single_shot_warm_shift = st.slider("Warm shift (+red,+yellow)", 0.0, 0.30, 0.12, 0.01)

        st.subheader("Runtime")
        st.write(f"Device: {'CUDA' if use_cuda else 'CPU'}")
        if use_cuda:
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")

    up = st.file_uploader("Input image", type=["png", "jpg", "jpeg"])

    default_weights = PROJECT_ROOT / "outputs" / "train_runs" / "default" / "best_model.pt"

    if st.button("Generate HDR", type="primary"):
        if up is None:
            st.error("Please upload an image.")
            return

        try:
            image = read_uploaded_image(up)
            out_preview, out_hdr, meta = run_model(
                model_name,
                image,
                device,
                default_weights,
                32,
                single_shot_target_luma,
                single_shot_exposure_comp,
                single_shot_gamma,
                single_shot_white_percentile,
                single_shot_black_percentile,
                single_shot_denoise,
                single_shot_detail_boost,
                single_shot_color_neutralize,
                single_shot_warm_shift,
            )

            if model_name == "single_shot_ai" and meta is not None:
                st.caption(
                    f"Single-shot diagnostics: input_luma={meta['input_luma']:.4f}, "
                    f"preprocess_gain={meta['preprocess_gain']:.3f}"
                )

            st.subheader("Comparison")
            c1, c2 = st.columns(2)
            with c1:
                st.image(bgr_to_rgb(image), caption="Original", use_column_width=True)
            with c2:
                st.image(bgr_to_rgb(out_preview), caption="Enhanced", use_column_width=True)

            st.subheader("Downloads")
            st.download_button(
                "Download Original (.png)",
                data=to_png_bytes(image),
                file_name=f"{Path(up.name).stem}_original.png",
                mime="image/png",
            )
            st.download_button(
                "Download Enhanced (.png)",
                data=to_png_bytes(out_preview),
                file_name=f"{Path(up.name).stem}_{model_name}.png",
                mime="image/png",
            )

            if out_hdr is not None:
                st.download_button(
                    "Download HDR (.hdr)",
                    data=to_hdr_bytes(out_hdr),
                    file_name=f"{Path(up.name).stem}_{model_name}.hdr",
                    mime="application/octet-stream",
                )

        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    main()
