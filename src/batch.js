// Utilities shared with main.js (duplicated here to avoid module graph changes)
function waitForOpenCV() {
  return new Promise(resolve => {
    if (typeof cv !== 'undefined' && cv.Mat) return resolve();
    const timer = setInterval(() => {
      if (typeof cv !== 'undefined' && cv.Mat) { clearInterval(timer); resolve(); }
    }, 50);
  });
}

function leastSquaresKAndB(points) {
  const n = points.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (const [x, y] of points) { sumX += x; sumY += y; sumXY += x * y; sumXX += x * x; }
  const k = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX + 1e-6);
  const b = (sumY - k * sumX) / n;
  return { k, b };
}

function getEyeCornerPoints(landmarks, width, height) {
  const idx = { rOuter: 33, rInner: 133, lInner: 362, lOuter: 263 };
  const scalePoint = p => ([p.x * width, p.y * height]);
  const rOuter = scalePoint(landmarks[idx.rOuter]);
  const rInner = scalePoint(landmarks[idx.rInner]);
  const lInner = scalePoint(landmarks[idx.lInner]);
  const lOuter = scalePoint(landmarks[idx.lOuter]);
  return { rOuter, rInner, lInner, lOuter };
}

function getCentersFromCorners(imgW, imgH, corners) {
  const { rOuter, rInner, lInner, lOuter } = corners;
  const p = [rOuter, rInner, lInner, lOuter];
  const { k, b } = leastSquaresKAndB(p);
  const xLeft = (lOuter[0] + lInner[0]) / 2;
  const xRight = (rOuter[0] + rInner[0]) / 2;
  const leftCenter = [Math.round(xLeft), Math.round(xLeft * k + b)];
  const rightCenter = [Math.round(xRight), Math.round(xRight * k + b)];
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
    if (ch === 4) { cv.cvtColor(aligned, gray, cv.COLOR_RGBA2GRAY); }
    else if (ch === 3) { cv.cvtColor(aligned, gray, cv.COLOR_RGB2GRAY); }
    else { aligned.copyTo(gray); }
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
  const H = thresh.rows; const W = thresh.cols;
  const d = Math.floor(H * 0.5);
  function clampRect(x0, y0, w0, h0) {
    x0 = Math.max(0, Math.min(x0, W - 1));
    y0 = Math.max(0, Math.min(y0, H - 1));
    w0 = Math.max(1, Math.min(w0, W - x0));
    h0 = Math.max(1, Math.min(h0, H - y0));
    return new cv.Rect(x0, y0, w0, h0);
  }
  let x = Math.floor(d * 6 / 7);
  let y = Math.floor(d * 3 / 4);
  let w = Math.floor(d * 2 / 7);
  let h = Math.floor(d * 2 / 4);
  let x21 = Math.floor(d * 1 / 4);
  let x22 = Math.floor(d * 5 / 4);
  let w2 = Math.floor(d * 1 / 2);
  let y2 = Math.floor(d * 8 / 7);
  let h2 = Math.floor(d * 1 / 2);
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

function drawPreview(canvases, img, sobelAbs, measure) {
  const { preview, aligned, sobel } = canvases;
  const pctx = preview.getContext('2d');
  preview.width = img.width; preview.height = img.height;
  pctx.drawImage(img, 0, 0);
  // Ensure sobel canvas shows correct (non-flipped) orientation
  cv.imshow(sobel, sobelAbs);
  const actx = aligned.getContext('2d');
  actx.font = '12px system-ui';
  actx.fillStyle = '#fff';
  actx.fillText(`measure: ${measure.toFixed(3)}`, 6, 16);

  // Draw ROI overlays on Sobel canvas (like realtime)
  try {
    const sctx = sobel.getContext('2d');
    const d = Math.floor(sobel.height * 0.5);
    const x = Math.floor(d * 6 / 7);
    const y = Math.floor(d * 3 / 4);
    const w = Math.floor(d * 2 / 7);
    const h = Math.floor(d * 2 / 4);
    const x21 = Math.floor(d * 1 / 4);
    const x22 = Math.floor(d * 5 / 4);
    const w2 = Math.floor(d * 1 / 2);
    const y2 = Math.floor(d * 8 / 7);
    const h2 = Math.floor(d * 1 / 2);
    sctx.save();
    sctx.strokeStyle = '#22d3ee';
    sctx.lineWidth = 2;
    sctx.strokeRect(x, y, w, h);
    sctx.strokeRect(x21, y2, w2, h2);
    sctx.strokeRect(x22, y2, w2, h2);
    sctx.fillStyle = '#fff';
    sctx.font = '12px system-ui';
    sctx.fillText(`measure: ${measure.toFixed(3)}`, 8, 16);
    sctx.restore();
  } catch (_) {}
}

function createCsv(rows) {
  const header = ['path', 'measure', 'withGlasses'];
  const lines = [header.join(',')].concat(rows.map(r => [r.path, r.measure.toFixed(6), r.withGlasses ? 1 : 0].join(',')));
  return new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
}

async function runBatch(files, groundTruthMap = null) {
  const statusEl = document.getElementById('status');
  const progressEl = document.getElementById('progress');
  const summaryEl = document.getElementById('summary');
  const downloadEl = document.getElementById('downloadCsv');
  const measureBadge = document.getElementById('measureBadge');
  const gridEl = document.getElementById('thumbGrid');
  const thresholdInput = document.getElementById('thresholdInput');
  const thresholdLabel = document.getElementById('thresholdLabel');

  const preview = document.getElementById('preview');
  const alignedCanvas = document.getElementById('aligned');
  const sobelCanvas = document.getElementById('sobel');
  alignedCanvas.width = 256; alignedCanvas.height = 256;
  sobelCanvas.width = 256; sobelCanvas.height = 256;
  if (gridEl) gridEl.innerHTML = '';
  if (thresholdInput && thresholdLabel) {
    thresholdLabel.textContent = String(parseFloat(thresholdInput.value).toFixed(2));
    thresholdInput.addEventListener('input', () => {
      thresholdLabel.textContent = String(parseFloat(thresholdInput.value).toFixed(2));
      // Reclassify existing grid items and regenerate CSV when threshold changes
      if (gridEl) reclassifyGridAndCsv();
    });
  }

  await waitForOpenCV();
  const { FaceMesh } = window;
  const faceMesh = new FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}` });
  faceMesh.setOptions({
    staticImageMode: true,
    maxNumFaces: 1,
    refineLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  const resultsQueue = [];
  faceMesh.onResults((res) => {
    const resolver = resultsQueue.shift();
    if (resolver) resolver(res);
  });
  const baseMinEyeDistPx = 40;
  function getThreshold() {
    return thresholdInput && !isNaN(parseFloat(thresholdInput.value)) ? parseFloat(thresholdInput.value) : 0.13;
  }

  const totalFiles = files.length;
  let processedCount = 0;

  function reclassifyGridAndCsv() {
    if (!gridEl) return;
    const threshold = getThreshold();
    const thumbs = Array.from(gridEl.children || []);
    const rowsNew = [];
    let gtTotal = 0;
    let gtCorrect = 0;
    let gtFp = 0; // saw non-existent glasses
    let gtFn = 0; // failed to see glasses
    for (const t of thumbs) {
      const measureStr = t.dataset && t.dataset.measure ? t.dataset.measure : 'NaN';
      const measureVal = parseFloat(measureStr);
      const path = (t.dataset && t.dataset.path) || '';
      const predicted = isFinite(measureVal) ? (measureVal > threshold) : false;
      const truthStr = (t.dataset && typeof t.dataset.truth !== 'undefined') ? t.dataset.truth : '';
      const hasTruth = truthStr !== '' && truthStr !== null;
      const truth = hasTruth ? (truthStr === '1') : null;
      // Reset any previous classes
      t.classList.remove('ok', 'warn', 'blue', 'yellow');
      if (hasTruth) {
        gtTotal++;
        if (predicted === truth) {
          gtCorrect++;
        } else {
          if (predicted && !truth) gtFp++;
          if (!predicted && truth) gtFn++;
        }
        if (predicted === truth) {
          t.classList.add(predicted ? 'blue' : 'ok');
        } else {
          t.classList.add(predicted ? 'warn' : 'yellow');
        }
      } else {
        // Fallback to prediction-only coloring if no ground truth
        t.classList.add(predicted ? 'warn' : 'ok');
      }
      t.title = isFinite(measureVal)
        ? `${predicted ? 'With' : 'No'} Glasses (${measureVal.toFixed(3)})${hasTruth ? ` • GT: ${truth ? 'With' : 'No'} Glasses` : ''}`
        : 'No face detected';
      rowsNew.push({ path, measure: measureVal, withGlasses: predicted });
    }
    // Update badge for selected item, if any
    const selected = gridEl.querySelector('.thumb.selected');
    if (selected) {
      const m = parseFloat(selected.dataset.measure || 'NaN');
      const wgSel = isFinite(m) ? (m > threshold) : false;
      measureBadge.textContent = isFinite(m) ? `${wgSel ? 'With' : 'No'} Glasses (${m.toFixed(3)})` : 'No face';
      measureBadge.classList.toggle('warn', wgSel);
      measureBadge.classList.toggle('ok', !wgSel);
    }
    // Regenerate CSV download based on current threshold
    const blob = createCsv(rowsNew);
    const url = URL.createObjectURL(blob);
    downloadEl.href = url;
    downloadEl.style.display = 'inline-block';
    const isDone = processedCount >= totalFiles;
    const prefix = isDone ? `Done: ${thumbs.length} images.` : `Processed: ${processedCount}/${totalFiles}.`;
    if (gtTotal > 0) {
      const pct = ((gtCorrect / gtTotal) * 100).toFixed(1);
      const wrong = gtTotal - gtCorrect;
      const wrongPct = ((wrong / gtTotal) * 100).toFixed(1);
      summaryEl.textContent = `${prefix} Correct: ${gtCorrect}/${gtTotal} (${pct}%). Wrong: ${wrong} (${wrongPct}%). FP: ${gtFp}, FN: ${gtFn}. Click Download CSV.`;
    } else {
      summaryEl.textContent = `${prefix} Click Download CSV.`;
    }
  }

  function createScaledCanvasFromImage(imageEl, maxSide = 720) {
    const maxDim = Math.max(imageEl.width, imageEl.height);
    const scale = maxDim > maxSide ? (maxSide / maxDim) : 1;
    const w = Math.round(imageEl.width * scale);
    const h = Math.round(imageEl.height * scale);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');
    ctx.drawImage(imageEl, 0, 0, w, h);
    return c;
  }

  // Helper to (re)analyze and display outputs for a given HTMLImageElement
  async function showResultsForImage(imageEl) {
    const tmp = createScaledCanvasFromImage(imageEl, 720);
    const results = await new Promise((resolve) => {
      resultsQueue.push(resolve);
      faceMesh.send({ image: tmp });
    });
    let measure = NaN; let withGlasses = false;
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      const landmarks = results.multiFaceLandmarks[0];
      const corners = getEyeCornerPoints(landmarks, tmp.width, tmp.height);
      const { leftCenter, rightCenter } = getCentersFromCorners(tmp.width, tmp.height, corners);
      const eyeDx = rightCenter[0] - leftCenter[0];
      const eyeDy = rightCenter[1] - leftCenter[1];
      const eyeDist = Math.hypot(eyeDx, eyeDy);
      const dynamicMinEyeDist = Math.max(baseMinEyeDistPx, Math.round(0.055 * Math.min(tmp.width, tmp.height)));
      if (!isFinite(eyeDist) || eyeDist < dynamicMinEyeDist) {
        // too small/invalid face region
        const pctx = preview.getContext('2d');
        pctx.clearRect(0, 0, preview.width, preview.height);
        const actx = alignedCanvas.getContext('2d');
        actx.clearRect(0, 0, alignedCanvas.width, alignedCanvas.height);
        const sctx = sobelCanvas.getContext('2d');
        sctx.clearRect(0, 0, sobelCanvas.width, sobelCanvas.height);
        measureBadge.textContent = 'No face';
        measureBadge.classList.remove('ok', 'warn');
        return;
      }
      let aligned = getAlignedFaceFromCenters(tmp, leftCenter, rightCenter, 256, 256);
      // Ensure orientation matches realtime measurement expectations (flip vertically)
      const alignedFlipped = new cv.Mat();
      cv.flip(aligned, alignedFlipped, 0);
      aligned.delete();
      aligned = alignedFlipped;
      cv.imshow(alignedCanvas, aligned);
      const sobelAbs = computeSobelYAligned(aligned);
      ({ measure } = otsuAndMeasure(sobelAbs));
      withGlasses = measure > getThreshold();
      drawPreview({ preview, aligned: alignedCanvas, sobel: sobelCanvas }, imageEl, sobelAbs, measure);
      aligned.delete(); sobelAbs.delete();
    } else {
      // Clear canvases when no face
      const pctx = preview.getContext('2d');
      pctx.clearRect(0, 0, preview.width, preview.height);
      const actx = alignedCanvas.getContext('2d');
      actx.clearRect(0, 0, alignedCanvas.width, alignedCanvas.height);
      const sctx = sobelCanvas.getContext('2d');
      sctx.clearRect(0, 0, sobelCanvas.width, sobelCanvas.height);
    }
    measureBadge.textContent = isFinite(measure) ? `${withGlasses ? 'With' : 'No'} Glasses (${measure.toFixed(3)})` : 'No face';
    measureBadge.classList.toggle('warn', withGlasses);
    measureBadge.classList.toggle('ok', !withGlasses);
  }

  const rows = [];
  let cancelled = false;
  document.getElementById('cancelBtn').onclick = () => { cancelled = true; };

  for (let i = 0; i < files.length; i++) {
    if (cancelled) break;
    const f = files[i];
    statusEl.textContent = `Processing ${i + 1}/${files.length}: ${f.webkitRelativePath || f.name}`;
    progressEl.value = Math.round(((i + 1) / files.length) * 100);

    const img = await new Promise((resolve, reject) => {
      const url = URL.createObjectURL(f);
      const im = new Image();
      im.onload = () => resolve(im);
      im.onerror = reject;
      im.src = url;
    });

    // Draw to a scaled temp canvas for pipeline input (match realtime scale)
    const tmp = createScaledCanvasFromImage(img, 720);

    // Run facemesh
    const results = await new Promise((resolve) => {
      resultsQueue.push(resolve);
      faceMesh.send({ image: tmp });
    });

    let measure = NaN; let withGlasses = false;
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
      const landmarks = results.multiFaceLandmarks[0];
      const corners = getEyeCornerPoints(landmarks, tmp.width, tmp.height);
      const { leftCenter, rightCenter } = getCentersFromCorners(tmp.width, tmp.height, corners);
      const eyeDx = rightCenter[0] - leftCenter[0];
      const eyeDy = rightCenter[1] - leftCenter[1];
      const eyeDist = Math.hypot(eyeDx, eyeDy);
      const dynamicMinEyeDist = Math.max(baseMinEyeDistPx, Math.round(0.055 * Math.min(tmp.width, tmp.height)));
      if (!isFinite(eyeDist) || eyeDist < dynamicMinEyeDist) {
        const pctx = preview.getContext('2d');
        pctx.clearRect(0, 0, preview.width, preview.height);
        const actx = alignedCanvas.getContext('2d');
        actx.clearRect(0, 0, alignedCanvas.width, alignedCanvas.height);
        const sctx = sobelCanvas.getContext('2d');
        sctx.clearRect(0, 0, sobelCanvas.width, sobelCanvas.height);
        measureBadge.textContent = 'No face';
        measureBadge.classList.remove('ok', 'warn');
        return;
      }
      let aligned = getAlignedFaceFromCenters(tmp, leftCenter, rightCenter, 256, 256);
      const alignedFlipped = new cv.Mat();
      cv.flip(aligned, alignedFlipped, 0);
      aligned.delete();
      aligned = alignedFlipped;
      cv.imshow(alignedCanvas, aligned);
      const sobelAbs = computeSobelYAligned(aligned);
      ({ measure } = otsuAndMeasure(sobelAbs));
      withGlasses = measure > getThreshold();
      drawPreview({ preview, aligned: alignedCanvas, sobel: sobelCanvas }, img, sobelAbs, measure);
      aligned.delete(); sobelAbs.delete();
    }

    const filePath = f.webkitRelativePath || f.name;
    rows.push({ path: filePath, measure, withGlasses });
    measureBadge.textContent = isFinite(measure) ? `${withGlasses ? 'With' : 'No'} Glasses (${measure.toFixed(3)})` : 'No face';
    measureBadge.classList.toggle('warn', withGlasses);
    measureBadge.classList.toggle('ok', !withGlasses);
    if (gridEl) {
      const thumb = document.createElement('div');
      // Determine ground truth for this file (if provided)
      let truth = null;
      if (groundTruthMap) {
        const base = filePath.split('/').slice(-1)[0];
        const baseNoExt = base.replace(/\.[^.]+$/, '');
        const m = baseNoExt.match(/^face-(.+)$/);
        if (m) {
          const idKey = String(m[1]);
          if (Object.prototype.hasOwnProperty.call(groundTruthMap, idKey)) {
            truth = groundTruthMap[idKey];
          }
        }
      }
      // Assign border class based on prediction vs truth (if available)
      const cls = (() => {
        if (truth === null) return withGlasses ? 'warn' : 'ok';
        if (withGlasses === truth) return withGlasses ? 'blue' : 'ok';
        return withGlasses ? 'warn' : 'yellow';
      })();
      thumb.className = `thumb ${cls}`;
      thumb.title = isFinite(measure)
        ? `${withGlasses ? 'With' : 'No'} Glasses (${measure.toFixed(3)})${truth === null ? '' : ` • GT: ${truth ? 'With' : 'No'} Glasses`}`
        : 'No face detected';
      const thumbImg = document.createElement('img');
      thumbImg.src = img.src;
      thumbImg.alt = filePath;
      thumb.appendChild(thumbImg);
      const cap = document.createElement('div');
      cap.className = 'cap';
      cap.textContent = filePath.split('/').slice(-1)[0];
      thumb.appendChild(cap);
      thumb.dataset.measure = String(measure);
      thumb.dataset.path = filePath;
      if (truth !== null) thumb.dataset.truth = truth ? '1' : '0';
      gridEl.appendChild(thumb);
      thumb.addEventListener('click', () => {
        // mark selection
        for (const el of gridEl.querySelectorAll('.thumb.selected')) el.classList.remove('selected');
        thumb.classList.add('selected');
        showResultsForImage(img);
        console.log('[batch] clicked', { filePath, measure, withGlasses });
      });
      // Update running progress and stats in the UI
      processedCount = Math.min(totalFiles, (i + 1));
      reclassifyGridAndCsv();
    }
    await new Promise(r => setTimeout(r)); // allow UI to paint
  }

  // Finalize by classifying with current threshold and updating summary + CSV
  reclassifyGridAndCsv();
}

document.getElementById('startBtn').addEventListener('click', async () => {
  const input = document.getElementById('dirInput');
  const allFiles = Array.from(input.files || []);
  const files = allFiles.filter(f => /\.(jpg|jpeg|png|bmp|webp)$/i.test(f.name));
  if (files.length === 0) {
    alert('Please choose a folder with images.');
    return;
  }
  // Attempt to locate and parse train.csv for ground truth
  async function buildGroundTruthMapFromFiles(list) {
    const csvFile = list.find(f => f.name && f.name.toLowerCase() === 'train.csv');
    if (!csvFile) return null;
    try {
      const text = await csvFile.text();
      const lines = text.split(/\r?\n/).filter(l => l.trim().length > 0);
      if (lines.length === 0) return null;
      const header = lines.shift();
      const cols = header.split(',').map(s => s.trim().replace(/^"|"$/g, ''));
      const idIdx = cols.findIndex(c => /^id$/i.test(c));
      const gIdx = cols.findIndex(c => /^glasses$/i.test(c));
      if (idIdx < 0 || gIdx < 0) return null;
      const map = {};
      for (const line of lines) {
        const parts = line.split(',');
        const rawId = (parts[idIdx] || '').trim().replace(/^"|"$/g, '');
        const gRaw = (parts[gIdx] || '').trim().replace(/^"|"$/g, '');
        if (!rawId) continue;
        const gNum = parseFloat(gRaw);
        const gBool = (gRaw === '1') || (!Number.isNaN(gNum) ? gNum > 0.5 : /^true$/i.test(gRaw));
        map[String(rawId)] = !!gBool;
      }
      return map;
    } catch (_) {
      return null;
    }
  }
  const groundTruthMap = await buildGroundTruthMapFromFiles(allFiles);
  document.getElementById('downloadCsv').style.display = 'none';
  document.getElementById('progress').value = 0;
  document.getElementById('status').textContent = 'Starting...';
  runBatch(files, groundTruthMap);
});



// Initialize threshold label immediately on page load (not only after Start)
(function initThresholdLabel() {
  const thrInput = document.getElementById('thresholdInput');
  const thrLabel = document.getElementById('thresholdLabel');
  if (thrInput && thrLabel) {
    thrLabel.textContent = String(parseFloat(thrInput.value).toFixed(2));
    thrInput.addEventListener('input', () => {
      thrLabel.textContent = String(parseFloat(thrInput.value).toFixed(2));
    });
  }
})();