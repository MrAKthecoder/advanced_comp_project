# Viva Ready Q&A (Updated Run)

## 1) What was your project goal?
To convert low dynamic range/low exposure images into high dynamic range enhanced outputs, and compare classical bracketing methods with a single-shot deep learning approach on quality and compute.

## 2) Which methods did you implement?
1. Debevec HDR merge + tone mapping (bracketing)
2. Mertens exposure fusion (bracketing)
3. Single-shot U-Net log-HDR reconstruction (AI)

## 3) Why compare these three?
- Debevec: physically motivated HDR merge baseline
- Mertens: fast, strong classical exposure fusion baseline
- Single-shot: practical real-world method requiring only one image

## 4) What dataset did you use?
- Public HDR data from Poly Haven (.hdr/.exr)
- Raw files: 46
- Source split: train 32, val 6, test 8
- Processed train samples: 160 (via train copies)

## 5) Why train samples are 160 while raw files are 46?
We used controlled data augmentation in preprocessing by creating 5 train pairs per train source HDR image. This increases learning diversity in low-exposure conditions.

## 6) What exactly is the model predicting?
It predicts 3-channel log-HDR values from one low-exposure LDR image.
Then we recover HDR by `expm1` and tone-map for display.

## 7) Explain your loss function.
`Loss = L1(pred, target) + 0.2 * L1(gradient(pred), gradient(target))`
- L1 term improves pixel-level reconstruction
- Gradient term preserves edges and structure

## 8) What do train and val numbers (e.g., 0.12, 0.08) mean?
These are loss values, not accuracy.
Lower is better.
This is a regression/reconstruction task, so primary metrics are loss, PSNR, SSIM.

## 9) What were final training results?
Best validation loss = 0.07769 (epoch 11 in 12-epoch final run).

## 10) Which evaluation metrics did you use and why?
1. PSNR: measures pixel fidelity
2. SSIM: measures structural similarity
3. Latency (ms): runtime efficiency
4. Peak VRAM: memory cost for GPU inference

## 11) What are your updated benchmark averages?
- Debevec: 1024.69 ms, PSNR 17.6462, SSIM 0.7922
- Mertens: 37.47 ms, PSNR 19.6728, SSIM 0.8747
- Single-shot: 132.61 ms, PSNR 19.9758, SSIM 0.7847, VRAM 471.91 MB

## 12) Which method is best?
Depends on constraint:
- Best speed + strong structure: Mertens
- Physically grounded HDR merge: Debevec
- Best practicality with one image: single-shot model

## 13) Did increasing data help?
Yes. Compared with earlier baseline, single-shot improved significantly in PSNR, SSIM, and latency.

## 14) Why did you use 12 epochs in final run?
On this Windows environment, long runs intermittently stalled with loader/runtime stability issues. The 12-epoch run completed cleanly and reached strong validation loss; best checkpoint was used for benchmarking.

## 15) What is the main limitation?
Unique raw source scenes are still limited (46), so model generalization can be improved by adding more truly distinct HDR scenes.

## 16) What would you do next if given more time?
1. Increase unique HDR scenes to 150+
2. Add perceptual losses and better augmentation for real camera artifacts
3. Add real multi-image bracketing input in Streamlit UI
4. Evaluate on additional dynamic-scene HDR benchmarks

## 17) What is your key contribution?
A complete reproducible pipeline: data prep, training, classical baselines, quantitative benchmarking, and interactive demo app, with measurable improvement after data scaling.
