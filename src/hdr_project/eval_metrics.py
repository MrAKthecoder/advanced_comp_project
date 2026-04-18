from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def quality_metrics(pred_bgr_u8: np.ndarray, ref_bgr_u8: np.ndarray) -> Dict[str, float]:
    """Compute PSNR and SSIM in RGB space on uint8 images."""
    pred = cv2.cvtColor(pred_bgr_u8, cv2.COLOR_BGR2RGB)
    ref = cv2.cvtColor(ref_bgr_u8, cv2.COLOR_BGR2RGB)

    psnr = float(peak_signal_noise_ratio(ref, pred, data_range=255))
    ssim = float(structural_similarity(ref, pred, channel_axis=2, data_range=255))
    return {"psnr": psnr, "ssim": ssim}
