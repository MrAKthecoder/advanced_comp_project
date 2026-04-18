from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SingleShotHDRDataset(Dataset):
    """LDR-to-HDR paired dataset used for single-shot training.

    Folder layout:
      root/split/ldr/*.png
      root/split/hdr/*.npy
    """

    def __init__(self, root: str, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.ldr_dir = self.root / split / "ldr"
        self.hdr_dir = self.root / split / "hdr"

        if not self.ldr_dir.exists() or not self.hdr_dir.exists():
            raise FileNotFoundError(
                f"Missing split directories for '{split}': {self.ldr_dir} and {self.hdr_dir}"
            )

        self.ids = sorted([p.stem for p in self.ldr_dir.glob("*.png")])
        if not self.ids:
            raise RuntimeError(f"No samples found in {self.ldr_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.ids[idx]

        ldr_path = self.ldr_dir / f"{sample_id}.png"
        hdr_path = self.hdr_dir / f"{sample_id}.npy"

        ldr = cv2.imread(str(ldr_path), cv2.IMREAD_COLOR)
        if ldr is None:
            raise FileNotFoundError(f"Could not read LDR image: {ldr_path}")

        ldr = ldr.astype(np.float32) / 255.0
        hdr_log = np.load(hdr_path).astype(np.float32)

        # Convert to CHW tensors for PyTorch.
        ldr_t = torch.from_numpy(np.transpose(ldr, (2, 0, 1)))
        hdr_t = torch.from_numpy(np.transpose(hdr_log, (2, 0, 1)))

        return {"id": sample_id, "ldr": ldr_t, "hdr_log": hdr_t}


def count_samples(root: str) -> Dict[str, int]:
    """Quick helper to inspect dataset split sizes."""
    out: Dict[str, int] = {}
    for split in ["train", "val", "test"]:
        ldr_dir = Path(root) / split / "ldr"
        out[split] = len(list(ldr_dir.glob("*.png"))) if ldr_dir.exists() else 0
    return out
