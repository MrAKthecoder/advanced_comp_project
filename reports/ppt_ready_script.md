# PPT Ready Script: Low-Light LDR to HDR Enhancement

## Slide 1: Title
### On Slide
- Project Title: Low-Light LDR to HDR Enhancement Using Multiple Methods
- Student Name
- Department / Class
- Guide Name
- Date

### What I Will Say
This project improves dark low-dynamic-range images using multiple methods.
I used one AI method and multiple classical enhancement methods, and built a working app for demo.

---

## Slide 2: Problem Statement
### On Slide
- Dark images lose details in shadows
- Bright parts can clip easily
- Goal: make images more visible and balanced
- Need practical method that works on normal photos

### What I Will Say
In low-light photos, details are hidden and image quality looks poor.
My goal is to recover visibility and better dynamic range feeling using multiple approaches.

---

## Slide 3: Project Objective
### On Slide
- Build complete image enhancement pipeline
- Compare multiple methods
- Create user-friendly app
- Give downloadable outputs for comparison

### What I Will Say
This is not only model training.
I built end-to-end flow: data, methods, testing, and an interactive app for real use.

---

## Slide 4: One-Page Full Flow
### On Slide
- Collect HDR source images
- Prepare processed dataset
- Train single-shot AI model
- Implement classical methods
- Build Streamlit app
- User uploads image
- User selects method
- App generates enhanced output
- User compares and downloads result

### What I Will Say
This is the full pipeline from raw data to final app output.
First we prepare and train, then user uses app with one click inference.

---

## Slide 5: Data Preparation Flow
### On Slide
- Input: public HDR images
- Create low-exposure LDR inputs
- Create HDR targets in log space
- Split into train / val / test
- Also generate bracket scenes for classical HDR tests

### What I Will Say
I generated paired training data from HDR sources.
For testing classical HDR, I also created exposure stacks for bracketed methods.

---

## Slide 6: Methods Used (High Level)
### On Slide
- single_shot_ai (trained AI)
- retinex_lime_plus (classical enhancement)
- msrcp (strong Retinex color-preserving enhancement)
- single_fusion_hdr (synthetic fusion style)
- cinematic_grade (color + tone enhancement)
- Classical bracketing methods also available separately

### What I Will Say
In app there are 5 usable modes.
Only single_shot_ai is trained AI. Others are strong enhancement baselines.

---

## Slide 7: Single Shot AI Method
### On Slide
- Input: one dark image
- Model predicts HDR-like output
- Uses trained checkpoint
- Extra controls added for quality:
  - exposure compensation
  - gamma / black / white points
  - denoise / detail / color balance / warmth

### What I Will Say
This is the main learned method.
I improved inference controls to reduce washout and blue tint and make output more usable.

---

## Slide 8: Retinex LIME Plus
### On Slide
- Estimates illumination and reflectance
- Lifts shadows intelligently
- Adds denoise and detail recovery
- Good for very dark scenes

### What I Will Say
This method brightens dark regions with structure-aware behavior instead of only global brightness.

---

## Slide 9: Selective Shadow Relight
### On Slide
- Works mainly on shadow areas
- Protects already bright regions
- Reduces noise in low-light areas
- Good for portraits and low-key images

### What I Will Say
This method is useful when we only want shadow improvement and natural look.

---

## Slide 10: Single Fusion HDR
### On Slide
- Creates synthetic exposure stack from one image
- Merges using fusion logic
- Recovers more local details
- Good for balanced visibility

### What I Will Say
It mimics bracketing behavior from one input image, so it is practical when only one photo is available.

---

## Slide 11: Cinematic Grade
### On Slide
- Denoise
- Local contrast enhancement
- Color balancing
- Mild saturation and sharpening
- Cinematic visual output

### What I Will Say
This is a style plus visibility method.
It gives pleasing output for presentation and visual quality demos.

---

## Slide 12: Bracketing Methods (Core HDR Baseline)
### On Slide
- Debevec HDR merge
- Mertens exposure fusion
- Needs multiple images of same scene at different exposures
- Used for benchmark and strong reference quality

### What I Will Say
True multi-image HDR is available in project scripts.
In app we use single-image flow for usability, but bracketing pipeline is implemented and tested.

---

## Slide 13: App User Journey
### On Slide
- Open app
- Upload image (jpg/png)
- Choose method
- If single_shot_ai: tune quality sliders
- Click Generate
- See Original vs Enhanced
- Download output PNG (and HDR file for AI mode)

### What I Will Say
App is made for simple demo: upload, choose, generate, compare, download.

---

## Slide 14: What to Use for Which Image
### On Slide
- Very dark image: single_shot_ai or retinex_lime_plus
- Very dark image needing stronger natural visibility: msrcp
- Need balanced detail: single_fusion_hdr
- Need stylish final look: cinematic_grade

### What I Will Say
This slide helps quickly choose the best mode depending on input type.

---

## Slide 15: Improvements Added Recently
### On Slide
- Exposure-aware preprocessing
- Better tone mapping with black/white percentiles
- Denoise before enhancement
- Gray-world color neutralization
- Warm color shift to reduce blue cast
- Live sliders for fine control

### What I Will Say
These practical changes improved output quality significantly for difficult dark images.

---

## Slide 16: Limitations
### On Slide
- Single-shot AI may still fail on extreme cases
- App currently takes one image at a time
- True bracketing in app UI is not yet integrated
- Some outputs depend on tuning sliders

### What I Will Say
I am clear about limitations and where future work is needed.

---

## Slide 17: Future Work
### On Slide
- Improve model training with more diverse data
- Add real multi-image bracketing mode directly in app
- Better evaluation metrics for HDR realism
- Deploy as web tool with presets

### What I Will Say
Next step is to make it stronger and easier for non-technical users.

---

## Slide 18: Conclusion
### On Slide
- Built complete end-to-end pipeline
- Implemented multiple usable methods
- Delivered working demo app
- Achieved practical low-light enhancement with HDR-style outputs

### What I Will Say
Project is complete, functional, and explainable.
It combines AI and classical approaches and can be demonstrated live.

---

## Backup Slide: Fast Viva Q and A
### On Slide
- Q: Which method is trained?
  - A: single_shot_ai
- Q: Are others useless?
  - A: No, they are strong classical baselines and practical for many images
- Q: Where is true bracketing?
  - A: Implemented separately, needs multiple exposures
- Q: Why multiple methods?
  - A: Different image types need different enhancement behavior

### What I Will Say
I chose multiple methods so the system is practical, not dependent on one model only.
