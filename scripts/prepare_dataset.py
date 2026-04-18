from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Add project src directory for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdr_project.utils import ensure_dir, read_hdr_image, tonemap_for_display


def list_hdr_files(raw_hdr_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.hdr", "*.exr"):
        files.extend(sorted(raw_hdr_dir.glob(ext)))
    return files


def split_ids(ids: List[str], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(seed)
    ids = ids[:]
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    return train_ids, val_ids, test_ids


def make_low_ldr(hdr: np.ndarray, exposure_scale: float) -> np.ndarray:
    """Create low-exposure LDR image from linear HDR radiance."""
    ldr = np.power(np.clip(hdr * exposure_scale, 0.0, 1.0), 1.0 / 2.2)
    return (ldr * 255.0).astype(np.uint8)


def build_pairs(hdr: np.ndarray, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    # Train split gets random low exposures for better augmentation.
    if split_name == "train":
        exposure = np.random.uniform(0.05, 0.25)
    elif split_name == "val":
        exposure = 0.12
    else:
        exposure = 0.08

    ldr_low = make_low_ldr(hdr, exposure)
    hdr_log = np.log1p(np.clip(hdr, 0.0, None)).astype(np.float32)
    return ldr_low, hdr_log


def create_bracket_scene(scene_dir: Path, hdr: np.ndarray) -> None:
    """Create 3-image exposure stack from HDR image for classical methods."""
    ensure_dir(scene_dir)

    exposure_times = [1.0 / 30.0, 1.0 / 8.0, 1.0 / 2.0]
    scale_factors = [0.08, 0.2, 0.6]

    lines = []
    for idx, (t, scale) in enumerate(zip(exposure_times, scale_factors, strict=True)):
        frame = make_low_ldr(hdr, scale)
        name = f"img_{idx}.png"
        cv2.imwrite(str(scene_dir / name), frame)
        lines.append(f"{name} {t:.8f}")

    (scene_dir / "exposures.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Save reference HDR for quantitative benchmarking.
    np.save(scene_dir / "reference_hdr.npy", hdr.astype(np.float32))
    cv2.imwrite(str(scene_dir / "reference_tonemap.png"), tonemap_for_display(hdr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare single-shot and bracketed HDR dataset.")
    parser.add_argument("--raw-hdr-dir", type=Path, default=Path("data/raw_hdr"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--bracketed-root", type=Path, default=Path("data/bracketed"))
    parser.add_argument("--resize", type=int, default=512, help="Resize shorter side to this value.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    hdr_files = list_hdr_files(args.raw_hdr_dir)
    if not hdr_files:
        raise RuntimeError(
            f"No HDR files found in {args.raw_hdr_dir}. Add .hdr/.exr files from a public dataset first."
        )

    ids = [p.stem for p in hdr_files]
    split_train, split_val, split_test = split_ids(ids, args.train_ratio, args.val_ratio, args.seed)
    split_map = {
        "train": set(split_train),
        "val": set(split_val),
        "test": set(split_test),
    }

    for split in ["train", "val", "test"]:
        ensure_dir(args.processed_root / split / "ldr")
        ensure_dir(args.processed_root / split / "hdr")

    ensure_dir(args.bracketed_root)

    for hdr_path in hdr_files:
        sample_id = hdr_path.stem
        hdr = read_hdr_image(hdr_path)

        # Keep aspect ratio, resize the shorter side for manageable GPU memory.
        h, w = hdr.shape[:2]
        scale = args.resize / min(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        hdr = cv2.resize(hdr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        split_name = "test"
        if sample_id in split_map["train"]:
            split_name = "train"
        elif sample_id in split_map["val"]:
            split_name = "val"

        ldr_low, hdr_log = build_pairs(hdr, split_name)
        cv2.imwrite(str(args.processed_root / split_name / "ldr" / f"{sample_id}.png"), ldr_low)
        np.save(args.processed_root / split_name / "hdr" / f"{sample_id}.npy", hdr_log)

        # Bracketing scenes are created only from test split to keep eval clean.
        if split_name == "test":
            create_bracket_scene(args.bracketed_root / sample_id, hdr)

    print("Dataset preparation complete.")
    print(f"Train: {len(split_train)}, Val: {len(split_val)}, Test: {len(split_test)}")


if __name__ == "__main__":
    main()
