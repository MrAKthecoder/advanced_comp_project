from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def ensure_dir(path: str | Path) -> None:
    """Create a directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def read_hdr_image(path: str | Path) -> np.ndarray:
    """Read HDR/EXR image as float32 BGR with shape HxWx3."""
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read HDR image: {path}")
    return image.astype(np.float32)


def write_hdr_image(path: str | Path, image: np.ndarray) -> None:
    """Write float32 HDR image to disk in Radiance HDR format."""
    cv2.imwrite(str(path), image.astype(np.float32))


def tonemap_for_display(hdr: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Convert linear HDR to displayable uint8 image."""
    hdr = np.clip(hdr, 0.0, None)
    mapped = np.log1p(hdr)
    mapped /= mapped.max() + 1e-8
    mapped = np.power(mapped, 1.0 / gamma)
    return (np.clip(mapped, 0.0, 1.0) * 255.0).astype(np.uint8)


def read_exposure_list(scene_dir: str | Path) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load exposure stack and times from exposures.txt in a scene folder.

    Expected line format in exposures.txt:
        image_0.png 0.033333
    """
    scene_dir = Path(scene_dir)
    exp_file = scene_dir / "exposures.txt"
    if not exp_file.exists():
        raise FileNotFoundError(f"Missing exposure file: {exp_file}")

    images: List[np.ndarray] = []
    times: List[float] = []

    for line in exp_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        file_name, time_val = line.split()
        image = cv2.imread(str(scene_dir / file_name), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read bracket image: {scene_dir / file_name}")
        images.append(image)
        times.append(float(time_val))

    return images, np.array(times, dtype=np.float32)
