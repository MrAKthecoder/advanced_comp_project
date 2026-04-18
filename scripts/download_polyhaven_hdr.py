from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Add project src directory for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdr_project.utils import ensure_dir

ASSETS_API = "https://api.polyhaven.com/assets?t=hdris"
FILES_API = "https://api.polyhaven.com/files/{asset_id}"


def fetch_json(url: str) -> Dict:
    """Fetch JSON from URL with a user-agent to avoid blocked anonymous calls."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def iter_urls(node: object) -> Iterable[str]:
    """Recursively yield all URL strings from nested JSON payload."""
    if isinstance(node, dict):
        for value in node.values():
            yield from iter_urls(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_urls(value)
    elif isinstance(node, str) and node.startswith("http"):
        yield node


def pick_hdr_url(files_json: Dict, prefer_exr: bool = False) -> Optional[str]:
    """Select one download URL, preferring small resolutions to keep dataset setup fast."""
    urls = [u for u in iter_urls(files_json) if u.lower().endswith((".hdr", ".exr"))]
    if not urls:
        return None

    # Prefer 1k/2k assets first for faster downloads and one-week project timeline.
    def score(u: str) -> Tuple[int, int, int]:
        low = u.lower()
        res = 2
        if "1k" in low:
            res = 0
        elif "2k" in low:
            res = 1

        ext_pref = 0
        if prefer_exr:
            ext_pref = 0 if low.endswith(".exr") else 1
        else:
            ext_pref = 0 if low.endswith(".hdr") else 1

        # Shorter path often maps to base downloadable variant.
        return (res, ext_pref, len(u))

    urls.sort(key=score)
    return urls[0]


def download_file(url: str, out_path: Path) -> None:
    """Download one HDR file to target path."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        out_path.write_bytes(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public HDR files from Poly Haven into data/raw_hdr.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_hdr"))
    parser.add_argument("--limit", type=int, default=40, help="How many HDR files to download.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset in sorted asset list.")
    parser.add_argument("--prefer-exr", action="store_true", help="Prefer EXR over HDR if available.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw_hdr/download_manifest.csv"),
        help="CSV log of downloaded assets.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(args.manifest.parent)

    assets = fetch_json(ASSETS_API)
    asset_ids = sorted(list(assets.keys()))
    chosen = asset_ids[args.start_index : args.start_index + args.limit]

    if not chosen:
        raise RuntimeError("No assets selected. Increase --limit or change --start-index.")

    downloaded = 0
    with args.manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["asset_id", "url", "saved_path"])

        for idx, asset_id in enumerate(chosen, start=1):
            try:
                files_json = fetch_json(FILES_API.format(asset_id=asset_id))
                url = pick_hdr_url(files_json, prefer_exr=args.prefer_exr)
                if not url:
                    print(f"[{idx}/{len(chosen)}] skip {asset_id}: no HDR/EXR url found")
                    continue

                ext = ".exr" if url.lower().endswith(".exr") else ".hdr"
                out_path = args.output_dir / f"{asset_id}{ext}"
                if out_path.exists() and out_path.stat().st_size > 0:
                    print(f"[{idx}/{len(chosen)}] exists {out_path.name}")
                    writer.writerow([asset_id, url, str(out_path)])
                    downloaded += 1
                    continue

                print(f"[{idx}/{len(chosen)}] downloading {asset_id} -> {out_path.name}")
                download_file(url, out_path)
                writer.writerow([asset_id, url, str(out_path)])
                downloaded += 1
            except Exception as exc:
                print(f"[{idx}/{len(chosen)}] failed {asset_id}: {exc}")

    print(f"Done. Downloaded {downloaded} HDR files to {args.output_dir}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
