from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

# Add project src directory for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdr_project.classical import debevec_hdr, mertens_fusion
from hdr_project.model import SmallUNetHDR
from hdr_project.utils import ensure_dir, read_exposure_list, tonemap_for_display, write_hdr_image


def load_single_shot_model(weights_path: Path, device: torch.device, base_channels: int) -> SmallUNetHDR:
    """Load trained single-shot model weights."""
    model = SmallUNetHDR(base=base_channels).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def single_shot_predict(model: SmallUNetHDR, image_path: Path, device: torch.device) -> np.ndarray:
    """Run one-image LDR to HDR prediction and return linear HDR float32."""
    ldr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if ldr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    x = torch.from_numpy(np.transpose(ldr.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_log = model(x).squeeze(0).cpu().numpy()

    pred_log = np.transpose(pred_log, (1, 2, 0))
    pred_hdr = np.expm1(np.clip(pred_log, 0.0, None)).astype(np.float32)
    return pred_hdr


def clahe_enhance(image_path: Path) -> np.ndarray:
    """Fast baseline enhancement using CLAHE on luminance channel."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # CLAHE improves local contrast in dark/low dynamic-range regions.
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_new = clahe.apply(l_chan)

    merged = cv2.merge([l_new, a_chan, b_chan])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return out


def method_menu() -> str:
    """Prompt user for method choice."""
    print("Choose conversion method:")
    print("  1) single_shot      - one low image -> HDR via trained model")
    print("  2) debevec          - bracket folder -> HDR merge + tonemap")
    print("  3) mertens          - bracket folder -> exposure fusion")
    print("  4) bracketing_both  - run both Debevec and Mertens")
    print("  5) clahe_fast       - one image fast classical enhancement")

    mapping = {
        "1": "single_shot",
        "2": "debevec",
        "3": "mertens",
        "4": "bracketing_both",
        "5": "clahe_fast",
    }

    while True:
        choice = input("Enter number (1-5): ").strip()
        if choice in mapping:
            return mapping[choice]
        print("Invalid choice. Please select 1 to 5.")


def prompt_existing_path(prompt_text: str) -> Path:
    """Prompt until user provides an existing file/folder path."""
    while True:
        raw = input(prompt_text).strip().strip('"')
        p = Path(raw)
        if p.exists():
            return p
        print("Path does not exist. Try again.")


def resolve_device(device_arg: str) -> torch.device:
    """Resolve runtime device from argument with auto fallback."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def run_single_shot_flow(input_image: Path, output_dir: Path, weights: Path, device: torch.device, base_channels: int) -> None:
    """Run single-shot conversion and save HDR plus tone-mapped preview."""
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights file: {weights}")

    model = load_single_shot_model(weights, device, base_channels)
    pred_hdr = single_shot_predict(model, input_image, device)

    stem = input_image.stem
    out_hdr = output_dir / f"{stem}_single_shot.hdr"
    out_png = output_dir / f"{stem}_single_shot_tonemap.png"

    write_hdr_image(out_hdr, pred_hdr)
    cv2.imwrite(str(out_png), tonemap_for_display(pred_hdr))

    print(f"Saved HDR: {out_hdr}")
    print(f"Saved preview: {out_png}")


def run_bracketing_flow(scene_dir: Path, output_dir: Path, method: str) -> None:
    """Run bracketing methods on one scene folder with exposures.txt."""
    images, times = read_exposure_list(scene_dir)

    if method in {"debevec", "bracketing_both"}:
        deb = debevec_hdr(images, times)
        write_hdr_image(output_dir / "debevec_hdr.hdr", deb["hdr"])
        cv2.imwrite(str(output_dir / "debevec_tonemap.png"), deb["ldr"])
        print(f"Saved Debevec outputs in: {output_dir}")

    if method in {"mertens", "bracketing_both"}:
        mer = mertens_fusion(images)
        cv2.imwrite(str(output_dir / "mertens_fusion.png"), mer["fusion"])
        print(f"Saved Mertens output in: {output_dir}")


def run_clahe_flow(input_image: Path, output_dir: Path) -> None:
    """Run fast classical enhancement flow for one low image."""
    out = clahe_enhance(input_image)
    out_path = output_dir / f"{input_image.stem}_clahe_fast.png"
    cv2.imwrite(str(out_path), out)
    print(f"Saved CLAHE output: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive HDR converter: choose single-shot, bracketing, or fast classical enhancement."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="interactive",
        choices=["interactive", "single_shot", "debevec", "mertens", "bracketing_both", "clahe_fast"],
    )
    parser.add_argument("--input", type=Path, default=None, help="Image path for single_shot/clahe_fast, or scene folder for bracketing.")
    parser.add_argument("--weights", type=Path, default=Path("outputs/train_runs/default/best_model.pt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/app_runs"))
    return parser.parse_args()


def interactive_collect(method: str, default_weights: Path) -> Tuple[str, Path, Path]:
    """Collect user inputs in interactive mode."""
    chosen_method = method_menu() if method == "interactive" else method

    if chosen_method in {"single_shot", "clahe_fast"}:
        input_path = prompt_existing_path("Enter path of low image (png/jpg): ")
    else:
        input_path = prompt_existing_path("Enter path of bracketing scene folder (must contain exposures.txt): ")

    weights = default_weights
    if chosen_method == "single_shot":
        print(f"Default model weights: {default_weights}")
        raw = input("Press Enter to use default, or type custom weights path: ").strip().strip('"')
        if raw:
            weights = Path(raw)

    return chosen_method, input_path, weights


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU mode")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / timestamp
    ensure_dir(out_dir)

    method = args.method
    input_path = args.input
    weights = args.weights

    if method == "interactive" or input_path is None:
        method, input_path, weights = interactive_collect(method, args.weights)

    if method == "single_shot":
        if input_path.is_dir():
            raise ValueError("single_shot expects an image file path, not a folder.")
        run_single_shot_flow(input_path, out_dir, weights, device, args.base_channels)

    elif method in {"debevec", "mertens", "bracketing_both"}:
        if not input_path.is_dir():
            raise ValueError("Bracketing methods expect a scene folder containing exposures.txt.")
        run_bracketing_flow(input_path, out_dir, method)

    elif method == "clahe_fast":
        if input_path.is_dir():
            raise ValueError("clahe_fast expects an image file path, not a folder.")
        run_clahe_flow(input_path, out_dir)

    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Done. Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
