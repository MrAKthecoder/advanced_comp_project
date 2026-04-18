# Advanced Computer Vision Project: Low HDR to High HDR

This project converts low dynamic range inputs to higher dynamic range outputs using:

1. Bracketing method A: Debevec HDR merge + tone mapping.
2. Bracketing method B: Mertens exposure fusion.
3. Single-shot method C: Lightweight U-Net model that predicts log-HDR from one low-LDR image.

The codebase is structured for training, inference, and direct method comparison on quality and compute.

## 1) Environment Setup

```powershell
cd e:\advanced_comp_project
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2) Dataset Requirement

Add public HDR images (`.hdr` or `.exr`) to:

- `data/raw_hdr`

Recommended public sources:

1. Kalantari HDR dataset (dynamic scenes).
2. Any public HDR image collection containing `.hdr`/`.exr` files.

Quick auto-download option (public Poly Haven HDR set):

```powershell
.\.venv\Scripts\python.exe scripts\download_polyhaven_hdr.py --output-dir data\raw_hdr --limit 40
```

## 3) Prepare Dataset (Pairs + Bracketed Scenes)

```powershell
.\.venv\Scripts\python.exe scripts\prepare_dataset.py --raw-hdr-dir data\raw_hdr --processed-root data\processed --bracketed-root data\bracketed --resize 512
```

Outputs:

1. `data/processed/train|val|test` for single-shot training.
2. `data/bracketed/<scene>` for bracketing methods + reference HDR.

## 4) Train Single-Shot HDR (RTX 4070 8GB)

First ensure CUDA PyTorch is installed in the same `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

Verify GPU before training:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

```powershell
.\.venv\Scripts\python.exe scripts\train_single_shot.py --config configs\train_default.yaml
```

The trainer uses CUDA when available and prints the detected GPU name.

## 5) Run Bracketing Methods

```powershell
.\.venv\Scripts\python.exe scripts\run_bracketing.py --bracketed-root data\bracketed --output outputs\bracketing
```

## 6) Run Single-Shot Inference

```powershell
.\.venv\Scripts\python.exe scripts\infer_single_shot.py --weights outputs\train_runs\default\best_model.pt --input data\processed\test\ldr --output outputs\single_shot --device cuda
```

## 7) Benchmark and Compare Methods

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_methods.py --bracketed-root data\bracketed --single-shot-weights outputs\train_runs\default\best_model.pt --output-csv reports\benchmark.csv --device cuda
```

CSV columns include runtime, PSNR, SSIM, and peak VRAM (single-shot).

## 8) Suggested Report Tables

1. Quality table: method vs average PSNR/SSIM.
2. Compute table: method vs latency and memory/VRAM.
3. Visual figure: reference, Debevec, Mertens, single-shot outputs on same scene.

## 9) Interactive Frontend (Streamlit)

Run this to open a web UI where you can choose the method and upload images:

```powershell
.\.venv\Scripts\python.exe -m pip install streamlit
.\.venv\Scripts\python.exe -m streamlit run scripts\streamlit_app.py
```

In the app you can choose:

1. `single_shot` (one low image -> HDR + tone-mapped preview)
2. `debevec_bracketing` (multiple exposure images)
3. `mertens_bracketing` (multiple exposure images)
4. `bracketing_both` (Debevec + Mertens together)
5. `clahe_fast` (fast classical enhancement)
6. `gamma_boost` (extra classical baseline)

## Project Structure

```text
advanced_comp_project/
  configs/
  data/
    raw_hdr/
    processed/
    bracketed/
  scripts/
  src/hdr_project/
  outputs/
  reports/
```
