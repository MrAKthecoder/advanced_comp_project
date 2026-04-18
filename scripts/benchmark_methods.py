from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project src directory for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdr_project.classical import debevec_hdr, mertens_fusion
from hdr_project.eval_metrics import quality_metrics
from hdr_project.model import SmallUNetHDR
from hdr_project.utils import read_exposure_list, tonemap_for_display


def load_single_shot(weights: Path, device: torch.device, base_channels: int) -> SmallUNetHDR:
    model = SmallUNetHDR(base=base_channels).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model


def infer_single_shot(model: SmallUNetHDR, image_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.transpose(image_bgr.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_log = model(x).squeeze(0).cpu().numpy()
    pred_log = np.transpose(pred_log, (1, 2, 0))
    return np.expm1(np.clip(pred_log, 0.0, None)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare bracketing and single-shot HDR methods.")
    parser.add_argument("--bracketed-root", type=Path, default=Path("data/bracketed"))
    parser.add_argument("--single-shot-weights", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("reports/benchmark.csv"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base-channels", type=int, default=32)
    args = parser.parse_args()

    use_cuda = args.device.lower() == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_single_shot(args.single_shot_weights, device, args.base_channels)

    scene_dirs = sorted([p for p in args.bracketed_root.iterdir() if p.is_dir()])
    if not scene_dirs:
        raise RuntimeError(f"No scenes found in {args.bracketed_root}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "method", "latency_ms", "psnr", "ssim", "peak_vram_mb"])

        for scene_dir in scene_dirs:
            images, times = read_exposure_list(scene_dir)
            ref_hdr = np.load(scene_dir / "reference_hdr.npy").astype(np.float32)
            ref_tm = tonemap_for_display(ref_hdr)

            # Debevec bracketing.
            t0 = time.perf_counter()
            deb = debevec_hdr(images, times)
            t1 = time.perf_counter()
            m = quality_metrics(deb["ldr"], ref_tm)
            writer.writerow([scene_dir.name, "debevec", (t1 - t0) * 1000.0, m["psnr"], m["ssim"], 0.0])

            # Mertens bracketing.
            t0 = time.perf_counter()
            mer = mertens_fusion(images)
            t1 = time.perf_counter()
            m = quality_metrics(mer["fusion"], ref_tm)
            writer.writerow([scene_dir.name, "mertens", (t1 - t0) * 1000.0, m["psnr"], m["ssim"], 0.0])

            # Single-shot from darkest bracket image.
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            pred_hdr = infer_single_shot(model, images[0], device)
            t1 = time.perf_counter()
            pred_tm = tonemap_for_display(pred_hdr)
            m = quality_metrics(pred_tm, ref_tm)

            peak_vram_mb = 0.0
            if device.type == "cuda":
                peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

            writer.writerow([
                scene_dir.name,
                "single_shot",
                (t1 - t0) * 1000.0,
                m["psnr"],
                m["ssim"],
                peak_vram_mb,
            ])

    print(f"Benchmark saved to {args.output_csv}")


if __name__ == "__main__":
    main()
