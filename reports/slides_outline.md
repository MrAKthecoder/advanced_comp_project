# Presentation Outline: Low HDR to High HDR

## Slide 1: Title
- Low HDR to High HDR Enhancement
- Bracketing vs Single-Shot HDR Comparison

## Slide 2: Problem Statement
- Single low-exposure images lose detail in shadows/highlights.
- Goal: recover perceptual dynamic range and compare methods on quality and compute.

## Slide 3: Methods Compared
- Method A: Debevec HDR merge + tone mapping (bracketing)
- Method B: Mertens exposure fusion (bracketing)
- Method C: Single-shot U-Net HDR reconstruction

## Slide 4: Dataset and Preprocessing
- Public HDR files from Poly Haven
- Prepared to paired low-LDR/log-HDR dataset
- Split: Train 22, Val 4, Test 6

## Slide 5: Training Setup
- GPU: RTX 4070 Laptop GPU, 8 GB VRAM
- Model: Lightweight U-Net
- Loss: L1 reconstruction + gradient loss
- Epochs: 30

## Slide 6: Quantitative Results
- Insert table from reports/benchmark_summary.md
- Mention averages for latency, PSNR, SSIM, VRAM

## Slide 7: Visual Results
- Show per-scene comparison:
  - Reference tone map
  - Debevec output
  - Mertens output
  - Single-shot output

## Slide 8: Analysis
- Mertens best on quality and speed on this test set
- Debevec slower but physically motivated HDR merge
- Single-shot promising, needs more data for top quality

## Slide 9: Limitations
- Small dataset size
- Environment-map style HDR images, limited dynamic scene variety
- More training data and augmentation needed

## Slide 10: Future Work
- Train with larger benchmark datasets (Kalantari dynamic scenes)
- Add perceptual losses (LPIPS style)
- Explore larger model with mixed precision and scheduler

## Slide 11: Demo
- Run scripts:
  - prepare_dataset.py
  - train_single_shot.py
  - run_bracketing.py
  - benchmark_methods.py

## Slide 12: Conclusion
- Delivered full working pipeline
- Compared bracketing and single-shot methods on quality and compute
- Established reproducible baseline for final college submission
