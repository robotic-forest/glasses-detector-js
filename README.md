# Real-time Glasses Detection

<p align="center">
    <img src="./img/example_1.gif">
</p>

**Recent Updates:**

* 15 Aug 2019: Add ROI_2 to the measurement regions to improve robustness

## Introduction
This is a light-weight glasses detector written in Python for real-time videos. The algorithm is based on the method presented by the paper in [*Reference*](#Reference) with some modifications. Note that the goal is to determine the presence but not the precise location of glasses.

## Requirements
* python 3.6
- numpy 1.14
* opencv-python 3.4.0
- dlib 19.7.0

## Method
To determine the presence of glasses, the edgeness value (y-direction) of two important regions on the aligned face are computed. Then a indicator is constructed based on these values to do the classification.

The two measurement regions are shown on the schematic below.

<p align="center">
    <img src="./img/schematic.PNG" width="500">
</p>

## What's Next
A threshold is manually chosen in this specific version, which is based on experiment results. The next goal is to develop an algorithm that can choose the threshold automatically in order to enchance robustness.

Welcome to star, fork and try it on your own! :blush:

## Reference
Jiang, X., Binkert, M., Achermann, B. et al. Pattern Analysis & Applications (2000) 3: 9. https://doi.org/10.1007/s100440050002

## JavaScript (Browser) Version

A browser-based implementation has been added using MediaPipe FaceMesh for landmarks and OpenCV.js for image processing. It runs in real time from your webcam and replicates the Python pipeline:

- Landmark extraction (approximate eye corners)
- Linear regression to estimate the eye line and eye centers
- Face alignment via affine warp
- Edgeness with Sobel Y, Otsu thresholding, ROI measurements, and a fixed threshold for glasses detection

### Files
- `index.html`: App shell and CDN scripts for OpenCV.js and MediaPipe
- `styles.css`: Minimal UI styles
- `src/main.js`: The detection pipeline and UI overlays

### How to run
Run a simple static server from this directory (required for OpenCV.js/MediaPipe to load correctly):

```bash
# Using Python
python3 -m http.server 5173

# Or with Node (if installed)
npx serve -l 5173 .
```

Then open in your browser:

```bash
http://localhost:5173/index.html
```

Grant camera access when prompted. The result badge shows "With Glasses" or "No Glasses" and the canvases display the aligned face, Sobel Y, and ROI overlays.

### Notes
- The decision threshold is currently 0.15 to mirror the Python version; adjust in `src/main.js` (`updateResultBadge`) if needed for your environment.
- The landmark indices are approximations from MediaPipe FaceMesh and may differ slightly from dlib’s 5-point model, but the overall logic is maintained.

### Optional: Python live-reload server
You can run a local dev server that auto-reloads the browser when you change files:

```bash
python3 -m pip install livereload
python3 serve.py
```

Open:

```bash
http://localhost:5173/index.html
```

Now edits to `index.html`, `batch.html`, `styles.css`, or files under `src/` will refresh automatically.

## Batch testing on a folder of images

You can evaluate a dataset of images (e.g., `H:/Faces/faces-spring-2020/faces-spring-2020`) and export a CSV of results.

1. Start the static server as above and open:
   - Default algorithm (Sobel/Otsu): `http://localhost:5173/batch.html`
   - Edge-based nasal-bridge algorithm: `http://localhost:5173/edge-batch/`
2. Click the file picker and select the root folder of your images. On Windows/WSL, you can navigate to `H:` via the picker if your browser is running on Windows. If you run the server in WSL, ensure the folder is accessible from the browser by copying a subset to your Linux filesystem or mounting.
3. Click "Start" to process all images. You can cancel at any time.
4. When finished, click "Download CSV" to save results: `path,measure,withGlasses`.

CSV columns:
- `path`: relative path inside the chosen folder
- `measure`: combined edgeness measure
- `withGlasses`: 1 if measure > 0.15, else 0

Tips:
- If your dataset is labeled into subfolders (e.g., `with_glasses/` and `without_glasses/`), the `path` column will let you compute accuracy by joining with your labels.
- For consistent results, prefer frontal, reasonably sized faces; FaceMesh may skip very small or occluded faces.
