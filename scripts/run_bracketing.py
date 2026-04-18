from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Add project src directory for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdr_project.classical import debevec_hdr, mertens_fusion
from hdr_project.utils import ensure_dir, read_exposure_list, tonemap_for_display, write_hdr_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bracketing HDR methods on all test scenes.")
    parser.add_argument("--bracketed-root", type=Path, default=Path("data/bracketed"))
    parser.add_argument("--output", type=Path, default=Path("outputs/bracketing"))
    args = parser.parse_args()

    ensure_dir(args.output)
    scene_dirs = sorted([p for p in args.bracketed_root.iterdir() if p.is_dir()])
    if not scene_dirs:
        raise RuntimeError(f"No bracketed scenes found in {args.bracketed_root}")

    for scene_dir in scene_dirs:
        images, times = read_exposure_list(scene_dir)
        out_scene = args.output / scene_dir.name
        ensure_dir(out_scene)

        deb = debevec_hdr(images, times)
        mer = mertens_fusion(images)

        write_hdr_image(out_scene / "debevec_hdr.hdr", deb["hdr"])
        cv2.imwrite(str(out_scene / "debevec_tonemap.png"), deb["ldr"])
        cv2.imwrite(str(out_scene / "mertens_fusion.png"), mer["fusion"])

        # Save an extra display conversion of HDR merge for consistency.
        cv2.imwrite(str(out_scene / "debevec_tonemap_log.png"), tonemap_for_display(deb["hdr"]))

    print(f"Bracketing outputs saved to {args.output}")


if __name__ == "__main__":
    main()
