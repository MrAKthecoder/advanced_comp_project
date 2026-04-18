from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project src directory for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdr_project.data import SingleShotHDRDataset, count_samples
from hdr_project.losses import HDRReconstructionLoss
from hdr_project.model import SmallUNetHDR
from hdr_project.utils import ensure_dir, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        ldr = batch["ldr"].to(device, non_blocking=True)
        target = batch["hdr_log"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(ldr)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="val", leave=False):
            ldr = batch["ldr"].to(device, non_blocking=True)
            target = batch["hdr_log"].to(device, non_blocking=True)
            pred = model(ldr)
            loss = criterion(pred, target)
            total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-shot LDR-to-HDR model.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_default.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    requested_device = str(cfg["device"]).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Check NVIDIA driver / PyTorch install.")

    device = torch.device("cuda" if requested_device == "cuda" else "cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        if "4070" not in gpu_name:
            print("Warning: GPU does not look like RTX 4070; continuing anyway.")

    processed_root = str(cfg["processed_root"])
    split_counts = count_samples(processed_root)
    print(f"Dataset samples: {split_counts}")

    train_ds = SingleShotHDRDataset(processed_root, "train")
    val_ds = SingleShotHDRDataset(processed_root, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = SmallUNetHDR(base=int(cfg["base_channels"])).to(device)
    criterion = HDRReconstructionLoss(grad_weight=float(cfg["grad_weight"]))
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    output_dir = Path(cfg["output_dir"])
    ensure_dir(output_dir)
    history_path = output_dir / "history.csv"

    best_val = float("inf")

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, int(cfg["epochs"]) + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = eval_one_epoch(model, val_loader, criterion, device)
            writer.writerow([epoch, train_loss, val_loss])
            f.flush()

            print(f"Epoch {epoch:03d}: train={train_loss:.6f}, val={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), output_dir / "best_model.pt")

            if epoch % int(cfg["save_every"]) == 0:
                torch.save(model.state_dict(), output_dir / f"epoch_{epoch:03d}.pt")

    print(f"Training complete. Best val loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
