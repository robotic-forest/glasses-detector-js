/*
  Real-time Glasses Detection in JavaScript
  - Face landmarks: MediaPipe FaceMesh
  - Image processing: OpenCV.js
*/

// DOM elements
const videoEl = document.getElementById('video');
const outputCanvas = document.getElementById('output');
const outputCtx = outputCanvas.getContext('2d');
const alignedCanvas = document.getElementById('aligned');
const alignedCtx = alignedCanvas.getContext('2d');
const sobelCanvas = document.getElementById('sobel');
const sobelCtx = sobelCanvas.getContext('2d');
const roisCanvas = document.getElementById('rois');
const roisCtx = roisCanvas.getContext('2d');
const resultEl = document.getElementById('result');
const toggleBtn = document.getElementById('toggleBtn');
const resetBtn = document.getElementById('resetBtn');
const debugToggle = document.getElementById('debugToggle');
const debugLogEl = document.getElementById('debugLog');
const thresholdInput = document.getElementById('thresholdInput');
const thresholdLabel = document.getElementById('thresholdLabel');

// Threshold UI (bind immediately so it updates even before init completes)
if (thresholdInput && thresholdLabel) {
  thresholdLabel.textContent = String(parseFloat(thresholdInput.value).toFixed(2));
  thresholdInput.addEventListener('input', () => {
    thresholdLabel.textContent = String(parseFloat(thresholdInput.value).toFixed(2));
  });
}

function dlog(...args) {
  if (!debugToggle || !debugToggle.checked) return;
  const msg = args.map(a => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ');
  if (debugLogEl) {
    debugLogEl.textContent += msg + '\n';
    debugLogEl.scrollTop = debugLogEl.scrollHeight;
  }
  console.log('[debug]', ...args);
}

// Offscreen processing canvas (unmirrored) for correct alignment math
const processCanvas = document.createElement('canvas');
const processCtx = processCanvas.getContext('2d');

// Utility to wait for OpenCV.js to be ready
function waitForOpenCV() {
  return new Promise(resolve => {
    if (typeof cv !== 'undefined' && cv.Mat) return resolve();
    const timer = setInterval(() => {
      if (typeof cv !== 'undefined' && cv.Mat) {
        clearInterval(timer);
        resolve();
      }
    }, 50);
  });
}

// Get 5-point landmarks analog using FaceMesh indices:
// We'll approximate the four eye corners and compute centers via regression on those points
// FaceMesh landmark indices (rough approximations):
// Right eye outer (33), right eye inner (133), left eye inner (362), left eye outer (263)
function getEyeCornerPoints(landmarks, width, height) {
  const idx = { rOuter: 33, rInner: 133, lInner: 362, lOuter: 263 };
  const scalePoint = p => ([p.x * width, p.y * height]);
  const rOuter = scalePoint(landmarks[idx.rOuter]);
  const rInner = scalePoint(landmarks[idx.rInner]);
  const lInner = scalePoint(landmarks[idx.lInner]);
  const lOuter = scalePoint(landmarks[idx.lOuter]);
  return { rOuter, rInner, lInner, lOuter };
}

function leastSquaresKAndB(points) {
  // points: array of [x,y]
  const n = points.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (const [x, y] of points) {
    sumX += x; sumY += y; sumXY += x * y; sumXX += x * x;
  }
  const k = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX + 1e-6);
  const b = (sumY - k * sumX) / n;
  return { k, b };
}

function getCentersFromCorners(imgW, imgH, corners, ctx) {
  const { rOuter, rInner, lInner, lOuter } = corners;
  const p = [rOuter, rInner, lInner, lOuter];
  const { k, b } = leastSquaresKAndB(p);
  const xLeft = (lOuter[0] + lInner[0]) / 2;
  const xRight = (rOuter[0] + rInner[0]) / 2;
  const leftCenter = [Math.round(xLeft), Math.round(xLeft * k + b)];
  const rightCenter = [Math.round(xRight), Math.round(xRight * k + b)];

  // draw regression line and centers
  ctx.save();
  ctx.strokeStyle = '#00A2FF';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, b);
  ctx.lineTo(imgW, imgW * k + b);
  ctx.stroke();
  ctx.fillStyle = '#FF3B3B';
  ctx.beginPath(); ctx.arc(leftCenter[0], leftCenter[1], 4, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath(); ctx.arc(rightCenter[0], rightCenter[1], 4, 0, Math.PI * 2); ctx.fill();
  ctx.restore();

  return { leftCenter, rightCenter };
}

function getAlignedFaceFromCenters(srcCanvas, leftCenter, rightCenter, desiredW = 256, desiredH = 256) {
  const dx = rightCenter[0] - leftCenter[0];
  const dy = rightCenter[1] - leftCenter[1];
  const dist = Math.hypot(dx, dy);
  const desiredDist = desiredW * 0.5;
  const scale = desiredDist / (dist || 1);
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;

  const eyesCenter = [(leftCenter[0] + rightCenter[0]) * 0.5, (leftCenter[1] + rightCenter[1]) * 0.5];

  const M = cv.getRotationMatrix2D(new cv.Point(eyesCenter[0], eyesCenter[1]), angle, scale);
  M.doublePtr(0, 2)[0] += (desiredW * 0.5 - eyesCenter[0]);
  M.doublePtr(1, 2)[0] += (desiredH * 0.5 - eyesCenter[1]);

  const src = cv.imread(srcCanvas);
  const dst = new cv.Mat();
  const dsize = new cv.Size(desiredW, desiredH);
  cv.warpAffine(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
  src.delete(); M.delete();
  return dst;
}

function computeSobelYAligned(aligned) {
  const gray = new cv.Mat();
  try {
    const ch = aligned.channels();
    if (ch === 4) {
      cv.cvtColor(aligned, gray, cv.COLOR_RGBA2GRAY);
    } else if (ch === 3) {
      cv.cvtColor(aligned, gray, cv.COLOR_RGB2GRAY);
    } else {
      aligned.copyTo(gray);
    }
    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(11, 11), 0, 0, cv.BORDER_DEFAULT);
    const sobel = new cv.Mat();
    cv.Sobel(blurred, sobel, cv.CV_64F, 0, 1, 3);
    const sobelAbs = new cv.Mat();
    cv.convertScaleAbs(sobel, sobelAbs);
    blurred.delete(); sobel.delete();
    return sobelAbs;
  } finally {
    gray.delete();
  }
}

function otsuAndMeasure(sobelAbs) {
  // sobelAbs should already be 1-channel; ensure it
  const gray = new cv.Mat();
  if (sobelAbs.channels() !== 1) {
    const tmp = new cv.Mat();
    try {
      const ch = sobelAbs.channels();
      if (ch === 4) cv.cvtColor(sobelAbs, tmp, cv.COLOR_RGBA2GRAY);
      else if (ch === 3) cv.cvtColor(sobelAbs, tmp, cv.COLOR_RGB2GRAY);
      else sobelAbs.copyTo(tmp);
      tmp.copyTo(gray);
    } finally {
      tmp.delete();
    }
  } else {
    sobelAbs.copyTo(gray);
  }
  const thresh = new cv.Mat();
  cv.threshold(gray, thresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

  const H = thresh.rows;
  const W = thresh.cols;
  const d = Math.floor(H * 0.5);
  let x = Math.floor(d * 6 / 7);
  let y = Math.floor(d * 3 / 4);
  let w = Math.floor(d * 2 / 7);
  let h = Math.floor(d * 2 / 4);

  let x21 = Math.floor(d * 1 / 4);
  let x22 = Math.floor(d * 5 / 4);
  let w2 = Math.floor(d * 1 / 2);
  let y2 = Math.floor(d * 8 / 7);
  let h2 = Math.floor(d * 1 / 2);

  // Clamp ROIs to image bounds
  function clampRect(x0, y0, w0, h0) {
    x0 = Math.max(0, Math.min(x0, W - 1));
    y0 = Math.max(0, Math.min(y0, H - 1));
    w0 = Math.max(1, Math.min(w0, W - x0));
    h0 = Math.max(1, Math.min(h0, H - y0));
    return new cv.Rect(x0, y0, w0, h0);
  }

  const roi1 = thresh.roi(clampRect(x, y, w, h));
  const roi21 = thresh.roi(clampRect(x21, y2, w2, h2));
  const roi22 = thresh.roi(clampRect(x22, y2, w2, h2));

  const roiVec = new cv.MatVector();
  roiVec.push_back(roi21);
  roiVec.push_back(roi22);
  const roi2 = new cv.Mat();
  cv.hconcat(roiVec, roi2);
  roiVec.delete();

  const measure1 = cv.countNonZero(roi1) / (roi1.rows * roi1.cols || 1);
  const measure2 = cv.countNonZero(roi2) / (roi2.rows * roi2.cols || 1);
  const measure = measure1 * 0.3 + measure2 * 0.7;

  gray.delete(); thresh.delete(); roi1.delete(); roi21.delete(); roi22.delete(); roi2.delete();
  return { measure };
}

function drawRoisPreview(sobelAbs, measure) {
  // Left panel: Sobel
  cv.imshow(sobelCanvas, sobelAbs);
  const d = Math.floor(sobelCanvas.height * 0.5);
  const x = Math.floor(d * 6 / 7);
  const y = Math.floor(d * 3 / 4);
  const w = Math.floor(d * 2 / 7);
  const h = Math.floor(d * 2 / 4);
  const x21 = Math.floor(d * 1 / 4);
  const x22 = Math.floor(d * 5 / 4);
  const w2 = Math.floor(d * 1 / 2);
  const y2 = Math.floor(d * 8 / 7);
  const h2 = Math.floor(d * 1 / 2);

  // Draw boxes on Sobel
  {
    const ctx = sobelCtx;
    ctx.save();
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.strokeRect(x21, y2, w2, h2);
    ctx.strokeRect(x22, y2, w2, h2);
    ctx.fillStyle = '#fff';
    ctx.font = '12px system-ui';
    ctx.fillText(`measure: ${measure.toFixed(3)}`, 8, 16);
    ctx.restore();
  }

  // Right panel: Otsu threshold mask with same ROIs
  const gray = new cv.Mat();
  const thresh = new cv.Mat();
  try {
    if (sobelAbs.channels() !== 1) {
      const tmp = new cv.Mat();
      try {
        const ch = sobelAbs.channels();
        if (ch === 4) cv.cvtColor(sobelAbs, tmp, cv.COLOR_RGBA2GRAY);
        else if (ch === 3) cv.cvtColor(sobelAbs, tmp, cv.COLOR_RGB2GRAY);
        else sobelAbs.copyTo(tmp);
        tmp.copyTo(gray);
      } finally { tmp.delete(); }
    } else {
      sobelAbs.copyTo(gray);
    }
    cv.threshold(gray, thresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    cv.imshow(roisCanvas, thresh);
    const rctx = roisCtx;
    rctx.save();
    rctx.strokeStyle = '#22d3ee';
    rctx.lineWidth = 2;
    rctx.strokeRect(x, y, w, h);
    rctx.strokeRect(x21, y2, w2, h2);
    rctx.strokeRect(x22, y2, w2, h2);
    rctx.fillStyle = '#fff';
    rctx.font = '12px system-ui';
    rctx.fillText('Otsu mask', 8, 16);
    rctx.restore();
  } finally {
    gray.delete();
    thresh.delete();
  }
}

function updateResultBadge(measure) {
  const threshold = thresholdInput && !isNaN(parseFloat(thresholdInput.value)) ? parseFloat(thresholdInput.value) : 0.13;
  const isGlasses = measure > threshold;
  const text = `${isGlasses ? 'With Glasses' : 'No Glasses'} (${measure.toFixed(3)})`;
  resultEl.textContent = text;
  resultEl.classList.toggle('ok', !isGlasses);
  resultEl.classList.toggle('warn', isGlasses);
}

async function init() {
  await waitForOpenCV();

  // Setup camera stream
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 720, height: 540 } });
  videoEl.srcObject = stream;
  await videoEl.play();

  // Size canvases
  outputCanvas.width = videoEl.videoWidth; outputCanvas.height = videoEl.videoHeight;
  alignedCanvas.width = 256; alignedCanvas.height = 256;
  sobelCanvas.width = 256; sobelCanvas.height = 256;
  roisCanvas.width = 256; roisCanvas.height = 256;
  processCanvas.width = outputCanvas.width; processCanvas.height = outputCanvas.height;

  // Threshold UI handled above at module load time

  // Setup MediaPipe FaceMesh
  const { FaceMesh } = window;
  const faceMesh = new FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}` });
  faceMesh.setOptions({
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  // State for pause/reset and smoothing
  let paused = false;
  let recentMeasures = [];
  const maxWindow = 7; // temporal smoothing window
  const minEyeDistPx = 40; // gate small/false faces

  toggleBtn.addEventListener('click', () => {
    paused = !paused;
    toggleBtn.textContent = paused ? 'Resume' : 'Pause';
  });

  function clearOutputs() {
    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    alignedCtx.clearRect(0, 0, alignedCanvas.width, alignedCanvas.height);
    sobelCtx.clearRect(0, 0, sobelCanvas.width, sobelCanvas.height);
    roisCtx.clearRect(0, 0, roisCanvas.width, roisCanvas.height);
    resultEl.textContent = '—';
    resultEl.classList.remove('ok', 'warn');
  }

  resetBtn.addEventListener('click', () => {
    recentMeasures = [];
    clearOutputs();
    if (debugLogEl) debugLogEl.textContent = '';
  });

  faceMesh.onResults((results) => {
    let aligned = null;
    let sobelAbs = null;
    try {
      dlog('frame:start', Date.now());
      if (paused) return;
      // Draw raw frame to processing canvas (no mirror)
      processCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);
      // Draw mirrored frame to output for UI
      outputCtx.save();
      outputCtx.scale(-1, 1);
      outputCtx.drawImage(results.image, -outputCanvas.width, 0, outputCanvas.width, outputCanvas.height);
      outputCtx.restore();
      dlog('frame:drew');

      if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
        recentMeasures = [];
        clearOutputs();
        dlog('no-face');
        return;
      }

      const landmarks = results.multiFaceLandmarks[0];
      const corners = getEyeCornerPoints(landmarks, outputCanvas.width, outputCanvas.height);
      dlog('corners', corners);
      // Draw overlays in mirrored coordinate system but compute centers from unmirrored coords
      let leftCenter, rightCenter;
      outputCtx.save();
      outputCtx.translate(outputCanvas.width, 0);
      outputCtx.scale(-1, 1);
      ({ leftCenter, rightCenter } = getCentersFromCorners(outputCanvas.width, outputCanvas.height, corners, outputCtx));
      outputCtx.restore();
      dlog('centers', leftCenter, rightCenter);

      // Gate by minimum eye distance (reject tiny/false faces)
      const eyeDx = rightCenter[0] - leftCenter[0];
      const eyeDy = rightCenter[1] - leftCenter[1];
      const eyeDist = Math.hypot(eyeDx, eyeDy);
      if (!isFinite(eyeDist) || eyeDist < minEyeDistPx) {
        recentMeasures = [];
        clearOutputs();
        dlog('gate:eyeDist', eyeDist);
        return;
      }

      // Aligned face (flip vertically to match measurement orientation)
      aligned = getAlignedFaceFromCenters(processCanvas, leftCenter, rightCenter, 256, 256);
      const alignedFlipped = new cv.Mat();
      cv.flip(aligned, alignedFlipped, 0);
      aligned.delete();
      aligned = alignedFlipped;
      cv.imshow(alignedCanvas, aligned);
      dlog('aligned:ok');

      // Sobel Y and measurement
      sobelAbs = computeSobelYAligned(aligned);
      const { measure } = otsuAndMeasure(sobelAbs);
      if (!isFinite(measure)) return;
      dlog('measure', measure);
      // temporal smoothing (simple moving average)
      recentMeasures.push(measure);
      if (recentMeasures.length > maxWindow) recentMeasures.shift();
      const smoothed = recentMeasures.reduce((a, b) => a + b, 0) / recentMeasures.length;
      drawRoisPreview(sobelAbs, smoothed);
      updateResultBadge(smoothed);

      // ROIs panel just mirrors sobel for now (debug panel)
      cv.imshow(roisCanvas, sobelAbs);
    } catch (err) {
      console.error('onResults error:', err);
      dlog('error', String(err && err.message ? err.message : err));
      recentMeasures = [];
      clearOutputs();
    } finally {
      try { if (aligned) aligned.delete(); } catch (_) {}
      try { if (sobelAbs) sobelAbs.delete(); } catch (_) {}
      dlog('frame:end');
    }
  });

  // Use MediaPipe Camera Utils for frames
  const { Camera } = window;
  let frameCounter = 0;
  const camera = new Camera(videoEl, {
    onFrame: async () => {
      try {
        frameCounter = (frameCounter + 1) % 2; // process every 2nd frame
        if (frameCounter !== 0) return;
        await faceMesh.send({ image: videoEl });
      } catch (e) {
        console.error('send error:', e);
      }
    },
    width: outputCanvas.width,
    height: outputCanvas.height,
  });
  camera.start();
}

init().catch(err => {
  console.error(err);
  resultEl.textContent = 'Error: ' + err.message;
  resultEl.classList.add('warn');
});


