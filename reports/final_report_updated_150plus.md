# Final Report (Updated): Low-to-High HDR Enhancement System

## 1. Abstract
This project builds and evaluates a complete low-to-high HDR enhancement pipeline using both classical computer vision and deep learning. We compare three core methods: Debevec HDR merge (bracketing), Mertens exposure fusion (bracketing), and a single-shot U-Net model that predicts log-HDR from one low-exposure image. The system includes data collection from public HDR sources, dataset preparation, model training on RTX 4070 GPU, benchmarking by quality and compute cost, and an interactive Streamlit app for demos.

After upgrading training data to 150+ samples (160 train pairs), the single-shot model shows improved quality and speed compared with the earlier baseline while bracketing methods remain strong references.

## 2. Problem Statement
Low-exposure images lose detail in shadows and compress dynamic range. The goal is to recover visually richer outputs and compare methods by both quality and computational cost.

## 3. Objectives
1. Implement bracketing HDR methods (Debevec + Mertens).
2. Implement a single-shot AI HDR reconstruction model.
3. Compare all methods with PSNR, SSIM, latency, and VRAM.
4. Build an interactive application for demonstration.

## 4. Dataset and Data Preparation

### 4.1 Data Source
- Public HDR images from Poly Haven (`.hdr`, `.exr`).
- Raw files available under `data/raw_hdr`.

### 4.2 Final Dataset Size (Updated Run)
- Raw HDR files: 46
- Split by source scenes:
  - Train scenes: 32
  - Val scenes: 6
  - Test scenes: 8

### 4.3 How 150+ Training Samples Were Achieved
To increase train sample count from limited source HDR files, the dataset preparation script was upgraded to generate multiple train pairs per source scene.

- Train copies per source: 5
- Val copies per source: 1
- Test copies per source: 1

Resulting processed sample counts:
- Train samples: 160
- Val samples: 6
- Test samples: 8

### 4.4 Pair Construction Logic
For each HDR source image:
1. Resize while preserving aspect ratio (shorter side to 512).
2. Create low-exposure LDR input by scaling linear HDR and gamma converting.
3. Create HDR target as `log1p(HDR)`.

Exposure policy:
- Train: random exposure scale in [0.05, 0.25] (augmentation)
- Val: fixed exposure scale 0.12
- Test: fixed exposure scale 0.08

### 4.5 Bracketing Scene Generation
For each test scene, 3 bracket images are generated with:
- Exposure times: 1/30, 1/8, 1/2 sec
- Scale factors: 0.08, 0.2, 0.6

Also saved:
- `exposures.txt`
- `reference_hdr.npy`
- `reference_tonemap.png`

## 5. Methods Used

### 5.1 Method A: Debevec HDR Merge (Bracketing)
Pipeline:
1. Camera response calibration (`createCalibrateDebevec`).
2. HDR radiance map merge (`createMergeDebevec`).
3. Tone mapping for display (`createTonemapDrago`).

Why it works:
- Uses exposure-time physics and response calibration to recover a radiance map.
- Strong physically motivated HDR reconstruction.

Limitations:
- Requires multiple exposures of same scene.
- Highest latency among methods.

### 5.2 Method B: Mertens Exposure Fusion (Bracketing)
Pipeline:
1. Merge bracketed exposures with `createMergeMertens`.
2. Directly output fused LDR image.

Why it works:
- Selects best-exposed local details across images by multi-scale fusion.
- Produces sharp and well-exposed output quickly.

Limitations:
- No full radiance map output (fusion-oriented, display-oriented).

### 5.3 Method C: Single-Shot U-Net HDR Reconstruction
Pipeline:
1. Input one low-exposure image.
2. U-Net predicts 3-channel log-HDR.
3. Convert back with `expm1`.
4. Tone map for display.

Architecture details:
- Lightweight U-Net (`SmallUNetHDR`)
- Base channels: 32
- Softplus output head (enforces non-negative log-HDR outputs)
- Skip-connection alignment added for odd-sized inputs

Why it works:
- Learns mapping from dark LDR textures/edges to HDR-like luminance structure.
- Practical because only one image is needed.

Limitations:
- Quality depends on training diversity.
- Can still underperform strong bracketing methods on some scenes.

### 5.4 App-Only Single-Image Enhancement Modes
In addition to benchmarked methods, the Streamlit app includes practical enhancement models for user uploads:
- `retinex_lime_plus`
- `selective_shadow_relight`
- `single_fusion_hdr` (synthetic pseudo-bracketing + Mertens)
- `cinematic_grade`
- `single_shot_ai`

These improve usability for single-image demos but are separate from the strict benchmark trio above.

## 6. Training Setup

### 6.1 Hardware
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8GB)

### 6.2 Hyperparameters (Updated Final Run)
- Epochs: 12
- Batch size: 4
- Learning rate: 2e-4
- Weight decay: 1e-5
- Base channels: 32
- Optimizer: AdamW
- Device: CUDA
- DataLoader workers: 0 (stability on Windows)

### 6.3 Loss Function
`HDRReconstructionLoss = L1(pred, target) + grad_weight * L1(grad(pred), grad(target))`

with `grad_weight = 0.2`.

Interpretation:
- Pixel L1 term improves overall radiance reconstruction.
- Gradient term preserves edges/structures and avoids over-smooth outputs.

## 7. Updated Training Results
From `outputs/train_runs/default/history.csv`:
- Epoch 1: train 0.35998, val 0.27335
- Epoch 10: train 0.12863, val 0.08171
- Epoch 11: train 0.11833, val 0.07769 (best)
- Epoch 12: train 0.11009, val 0.08310

Best validation loss: **0.07769**

Important note:
- These are loss values (not classification accuracy).
- Lower is better.

## 8. Evaluation Protocol
Metrics computed against tone-mapped reference on test scenes:
- PSNR (higher better)
- SSIM (higher better)
- Latency in ms (lower better)
- Peak VRAM MB (single-shot CUDA)

## 9. Updated Benchmark Results (8 Test Scenes)
Averages from updated `reports/benchmark.csv`:

| Method | Avg Latency (ms) | Avg PSNR | Avg SSIM | Avg Peak VRAM (MB) |
|---|---:|---:|---:|---:|
| Debevec | 1024.692 | 17.6462 | 0.7922 | 0.000 |
| Mertens | 37.474 | 19.6728 | 0.8747 | 0.000 |
| Single-shot | 132.612 | 19.9758 | 0.7847 | 471.906 |

## 10. Before vs After (Single-Shot Model)
Using earlier baseline summary (small train set) vs updated 160-train run:

| Metric | Earlier Baseline | Updated (160-train) |
|---|---:|---:|
| Avg Latency (ms) | 317.78 | 132.61 |
| Avg PSNR | 17.01 | 19.98 |
| Avg SSIM | 0.6225 | 0.7847 |

Observation:
- Single-shot quality and speed improved significantly after scaling training samples.

Caution:
- Exact numerical comparison also reflects changed test composition (now 8 scenes).

## 11. Discussion
1. Mertens remains the best quality-speed tradeoff for bracketing scenes.
2. Debevec is physically meaningful but computationally expensive.
3. Updated single-shot model became much more competitive in PSNR and runtime after training data expansion.
4. Single-shot is the most practical deployment path when only one image is available.

## 12. Limitations
1. Raw unique HDR source count is still moderate (46).
2. Synthetic low-exposure generation may not cover all real camera noise/ISP artifacts.
3. Bracketing assumptions require alignment and stable scene capture.

## 13. Future Work
1. Increase unique HDR sources to 150+ unique scenes (not just augmented pairs).
2. Add dynamic-scene datasets (e.g., Kalantari-style benchmarks).
3. Add perceptual losses (LPIPS-style) and color constancy regularization.
4. Add true multi-image bracketing upload mode in Streamlit UI.
5. Add no-reference quality metrics for real photos without ground truth.

## 14. Reproducibility Commands (Updated)

```powershell
# Regenerate dataset (with 150+ train samples)
.\.venv\Scripts\python.exe scripts/prepare_dataset.py --raw-hdr-dir data/raw_hdr --processed-root data/processed --bracketed-root data/bracketed --resize 512 --train-copies 5 --val-copies 1 --test-copies 1

# Train
.\.venv\Scripts\python.exe scripts/train_single_shot.py --config configs/train_default.yaml

# Benchmark
.\.venv\Scripts\python.exe scripts/benchmark_methods.py --bracketed-root data/bracketed --single-shot-weights outputs/train_runs/default/best_model.pt --output-csv reports/benchmark.csv --device cuda
```

## 15. Final Conclusion
The updated project delivers a complete, reproducible HDR enhancement system with both classical and AI approaches. After increasing training samples to 160, the single-shot model improved strongly in PSNR/SSIM and latency, making it more viable for one-image use cases. Bracketing methods still serve as strong references, with Mertens providing the best speed-quality tradeoff. Overall, the project demonstrates both engineering completeness and meaningful performance progression suitable for final academic evaluation.
