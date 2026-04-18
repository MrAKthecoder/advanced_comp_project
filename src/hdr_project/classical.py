from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np


def debevec_hdr(images: List[np.ndarray], times: np.ndarray) -> Dict[str, np.ndarray]:
    """Run classical Debevec HDR merge and tone mapping."""
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, times)

    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, times, response)

    tonemap = cv2.createTonemapDrago(2.2)
    ldr = tonemap.process(hdr)
    # OpenCV tonemap can produce NaN/Inf in extreme highlights, so sanitize first.
    ldr = np.nan_to_num(ldr, nan=0.0, posinf=1.0, neginf=0.0)
    ldr_uint8 = (np.clip(ldr, 0.0, 1.0) * 255.0).astype(np.uint8)

    return {"hdr": hdr.astype(np.float32), "ldr": ldr_uint8}


def mertens_fusion(images: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Run exposure fusion from bracketed LDR images."""
    merge = cv2.createMergeMertens()
    fusion = merge.process(images)
    fusion_uint8 = (np.clip(fusion, 0.0, 1.0) * 255.0).astype(np.uint8)
    return {"fusion": fusion_uint8}
