# HDR Project Benchmark Summary

## Setup
- Dataset source: Public HDR images downloaded from Poly Haven.
- Prepared split sizes: Train 22, Val 4, Test 6.
- GPU used for training/inference: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB).

## Methods
1. Debevec HDR merge + tone mapping (bracketing)
2. Mertens exposure fusion (bracketing)
3. Single-shot U-Net HDR reconstruction

## Average Metrics (from reports/benchmark.csv)

| Method | Avg Latency (ms) | Avg PSNR | Avg SSIM | Peak VRAM (MB) |
|---|---:|---:|---:|---:|
| Debevec | 2656.99 | 19.07 | 0.7716 | 0.00 |
| Mertens | 74.19 | 23.17 | 0.8874 | 0.00 |
| Single-shot | 317.78 | 17.01 | 0.6225 | 471.91 |

## Key Observations
- Mertens gave the best quality on this dataset and was also the fastest among evaluated methods.
- Debevec is much slower due to response calibration and radiance map construction.
- Single-shot model is efficient enough for practical use but currently underperforms the bracketing baselines in quality; this can improve with larger and more diverse training data.

## Repro Commands

```powershell
# Prepare dataset
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --raw-hdr-dir data\raw_hdr --processed-root data\processed --bracketed-root data\bracketed --resize 512

# Train single-shot (uses CUDA when available)
.\.venv\Scripts\python.exe scripts\train_single_shot.py --config configs\train_default.yaml

# Run bracketing methods
.\.venv\Scripts\python.exe scripts\run_bracketing.py --bracketed-root data\bracketed --output outputs\bracketing

# Benchmark all methods
.\.venv\Scripts\python.exe scripts\benchmark_methods.py --bracketed-root data\bracketed --single-shot-weights outputs\train_runs\default\best_model.pt --output-csv reports\benchmark.csv --device cuda
```
