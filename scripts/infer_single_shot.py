from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project src directory for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdr_project.model import SmallUNetHDR
from hdr_project.utils import ensure_dir, tonemap_for_display, write_hdr_image


def load_model(weights_path: Path, device: torch.device, base_channels: int) -> SmallUNetHDR:
    model = SmallUNetHDR(base=base_channels).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_image(model: SmallUNetHDR, image_path: Path, device: torch.device) -> np.ndarray:
    ldr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if ldr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    x = torch.from_numpy(np.transpose(ldr.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_log = model(x).squeeze(0).cpu().numpy()

    pred_log = np.transpose(pred_log, (1, 2, 0))
    pred_hdr = np.expm1(np.clip(pred_log, 0.0, None)).astype(np.float32)
    return pred_hdr


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-shot HDR inference on low-LDR images.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Input image or folder with PNG/JPG files.")
    parser.add_argument("--output", type=Path, default=Path("outputs/single_shot"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base-channels", type=int, default=32)
    args = parser.parse_args()

    use_cuda = args.device.lower() == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(args.weights, device, args.base_channels)
    ensure_dir(args.output)

    if args.input.is_dir():
        image_paths = sorted(list(args.input.glob("*.png")) + list(args.input.glob("*.jpg")))
    else:
        image_paths = [args.input]

    if not image_paths:
        raise RuntimeError("No input images found.")

    for image_path in image_paths:
        pred_hdr = infer_image(model, image_path, device)
        out_name = image_path.stem

        write_hdr_image(args.output / f"{out_name}.hdr", pred_hdr)
        cv2.imwrite(str(args.output / f"{out_name}_tonemap.png"), tonemap_for_display(pred_hdr))

    print(f"Inference complete. Outputs saved to {args.output}")


if __name__ == "__main__":
    main()
