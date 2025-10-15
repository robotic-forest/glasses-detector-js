// Edge-based batch detector using notebook algorithm (Canny on nasal bridge ROI)
// Exposes the same runBatch(files, groundTruthMap) interface as src/batch.js

// Utilities shared with batch.js (duplicated to avoid module graph changes)
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

function drawPreview(canvases, img, edges, measure) {
  const { preview, aligned, sobel } = canvases;
  const pctx = preview.getContext('2d');
  preview.width = img.width; preview.height = img.height;
  pctx.drawImage(img, 0, 0);
  // Display edges into sobel canvas for consistency with batch UI
  cv.imshow(sobel, edges);
  const actx = aligned.getContext('2d');
  actx.font = '12px system-ui';
  actx.fillStyle = '#fff';
  actx.fillText(`measure: ${isFinite(measure) ? measure.toFixed(3) : 'NaN'}`, 6, 16);
}

function createCsv(rows) {
  const header = ['path', 'measure', 'withGlasses'];
  const lines = [header.join(',')].concat(rows.map(r => [r.path, isFinite(r.measure) ? r.measure.toFixed(6) : 'NaN', r.withGlasses ? 1 : 0].join(',')));
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
  // Ensure consistent 256x256 canvas size for display
  alignedCanvas.style.width = '256px'; alignedCanvas.style.height = '256px';
  sobelCanvas.style.width = '256px'; sobelCanvas.style.height = '256px';
  alignedCanvas.width = 256; alignedCanvas.height = 256;
  sobelCanvas.width = 256; sobelCanvas.height = 256;
  if (gridEl) gridEl.innerHTML = '';
  if (thresholdInput && thresholdLabel) {
    // For edge-based nasal bridge detection, threshold is binary (0)
    thresholdInput.value = '0';
    thresholdInput.disabled = true;
    thresholdLabel.textContent = '0.00';
  }

  await waitForOpenCV();
  const { FaceMesh } = window;
  const faceMesh = new FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}` });
  faceMesh.setOptions({ staticImageMode: true, maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  const resultsQueue = [];
  faceMesh.onResults((res) => { const resolver = resultsQueue.shift(); if (resolver) resolver(res); });

  const baseMinEyeDistPx = 40;
  function getThreshold() { return 0; }

  const totalFiles = files.length;
  let processedCount = 0;
  const rows = [];

  function reclassifyGridAndCsv() {
    if (!gridEl) return;
    const threshold = getThreshold();
    const thumbs = Array.from(gridEl.children || []);
    let gtTotal = 0; let gtCorrect = 0; let gtFp = 0; let gtFn = 0;
    for (const t of thumbs) {
      const measureStr = t.dataset && t.dataset.measure ? t.dataset.measure : 'NaN';
      const measureVal = parseFloat(measureStr);
      const path = (t.dataset && t.dataset.path) || '';
      const predicted = isFinite(measureVal) ? (measureVal > threshold) : false;
      const truthStr = (t.dataset && typeof t.dataset.truth !== 'undefined') ? t.dataset.truth : '';
      const hasTruth = truthStr !== '' && truthStr !== null;
      const truth = hasTruth ? (truthStr === '1') : null;
      t.classList.remove('ok', 'warn', 'blue', 'yellow');
      if (hasTruth) {
        gtTotal++;
        if (predicted === truth) { gtCorrect++; } else { if (predicted && !truth) gtFp++; if (!predicted && truth) gtFn++; }
        if (predicted === truth) { t.classList.add(predicted ? 'blue' : 'ok'); } else { t.classList.add(predicted ? 'warn' : 'yellow'); }
      } else {
        t.classList.add(predicted ? 'warn' : 'ok');
      }
      t.title = isFinite(measureVal)
        ? `${predicted ? 'With' : 'No'} Glasses (${measureVal.toFixed(3)})${hasTruth ? ` • GT: ${truth ? 'With' : 'No'} Glasses` : ''}`
        : 'No face detected';
      // UI only; CSV rows come from ordered rows array
    }
    const selected = gridEl.querySelector('.thumb.selected');
    if (selected) {
      const m = parseFloat(selected.dataset.measure || 'NaN');
      const wgSel = isFinite(m) ? (m > threshold) : false;
      measureBadge.textContent = isFinite(m) ? `${wgSel ? 'With' : 'No'} Glasses (${m.toFixed(3)})` : 'No face';
      measureBadge.classList.toggle('warn', wgSel);
      measureBadge.classList.toggle('ok', !wgSel);
    }
    const rowsForCsv = rows.map(r => ({ path: r.path, measure: r.measure, withGlasses: isFinite(r.measure) ? (r.measure > threshold) : false }));
    const blob = createCsv(rowsForCsv);
    const url = URL.createObjectURL(blob);
    downloadEl.href = url;
    downloadEl.style.display = 'inline-block';
    const isDone = processedCount >= totalFiles;
    const prefix = isDone ? `Done: ${thumbs.length} images.` : `Processed: ${processedCount}/${totalFiles}.`;
    if (gtTotal > 0) {
      const pct = ((gtCorrect / gtTotal) * 100).toFixed(1);
      const wrong = gtTotal - gtCorrect; const wrongPct = ((wrong / gtTotal) * 100).toFixed(1);
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

  // Compute nasal-bridge ROI from FaceMesh indices, following the notebook intent
  function computeNasalRoiRect(landmarks, imgW, imgH) {
    // Use indices approximating dlib 68 points used in notebook
    // Bridge vertical: 28,29,30,31,33,34,35 (approx equivalents in FaceMesh)
    const noseIdx = [6, 197, 195, 5, 4, 1, 275];
    const browIdx = 105; // approximate top eyebrow
    const lowerIdx = 2;  // bottom bound near tip/nostrils
    const xs = [];
    for (const i of noseIdx) { const p = landmarks[i]; xs.push(p.x * imgW); }
    const xMin = Math.max(0, Math.min(...xs));
    const xMax = Math.min(imgW - 1, Math.max(...xs));
    const yMin = Math.max(0, Math.min(imgH - 1, landmarks[browIdx].y * imgH));
    const yMax = Math.max(0, Math.min(imgH - 1, landmarks[lowerIdx].y * imgH));
    const rect = new cv.Rect(
      Math.round(xMin),
      Math.round(Math.min(yMin, yMax)),
      Math.max(1, Math.round(Math.abs(xMax - xMin))),
      Math.max(1, Math.round(Math.abs(yMax - yMin)))
    );
    return rect;
  }

  async function showResultsForImage(imageEl) {
    const tmp = createScaledCanvasFromImage(imageEl, 720);
    const results = await new Promise((resolve) => { resultsQueue.push(resolve); faceMesh.send({ image: tmp }); });
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
        const pctx = preview.getContext('2d'); pctx.clearRect(0, 0, preview.width, preview.height);
        const actx = alignedCanvas.getContext('2d'); actx.clearRect(0, 0, alignedCanvas.width, alignedCanvas.height);
        const sctx = sobelCanvas.getContext('2d'); sctx.clearRect(0, 0, sobelCanvas.width, sobelCanvas.height);
        measureBadge.textContent = 'No face'; measureBadge.classList.remove('ok', 'warn');
        return;
      }
      // Align and flip to match measurement orientation
      let aligned = getAlignedFaceFromCenters(tmp, leftCenter, rightCenter, 256, 256);
      const alignedFlipped = new cv.Mat(); cv.flip(aligned, alignedFlipped, 0); aligned.delete(); aligned = alignedFlipped;
      cv.imshow(alignedCanvas, aligned);

      // From notebook: crop nasal bridge ROI, blur, canny, check center column for edges
      const gray = new cv.Mat();
      try {
        const ch = aligned.channels();
        if (ch === 4) cv.cvtColor(aligned, gray, cv.COLOR_RGBA2GRAY);
        else if (ch === 3) cv.cvtColor(aligned, gray, cv.COLOR_RGB2GRAY);
        else aligned.copyTo(gray);

        // Estimate nasal ROI using landmarks in aligned space by mapping FaceMesh landmarks into aligned coords.
        // Simpler: derive ROI heuristically from aligned face center: a narrow vertical strip below eyes.
        const H = aligned.rows; const W = aligned.cols;
        const stripW = Math.max(4, Math.round(W * 0.08));
        const x0 = Math.round(W * 0.5 - stripW * 0.5);
        const y0 = Math.round(H * 0.30);
        const h0 = Math.max(8, Math.round(H * 0.35));
        const roiRect = new cv.Rect(x0, y0, Math.min(stripW, W - x0), Math.min(h0, H - y0));
        const roi = gray.roi(roiRect);

        const blurred = new cv.Mat();
        cv.GaussianBlur(roi, blurred, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT);
        const edges = new cv.Mat();
        cv.Canny(blurred, edges, 100, 200);

        // Measure: presence of a bright edge pixel along center column indicates glasses bridge
        const centerX = Math.floor(edges.cols / 2);
        const col = edges.col(centerX);
        const nz = cv.countNonZero(col);
        measure = nz / (col.rows || 1); // ratio of edge pixels along center
        withGlasses = nz > 0; // mimic notebook's binary check

        // For preview, embed ROI edges into a full-size (256x256) image to match other panels
        const edgesFull = cv.Mat.zeros(aligned.rows, aligned.cols, cv.CV_8UC1);
        const dstRoi = edgesFull.roi(roiRect);
        edges.copyTo(dstRoi);
        dstRoi.delete();
        drawPreview({ preview, aligned: alignedCanvas, sobel: sobelCanvas }, imageEl, edgesFull, measure);

        roi.delete(); blurred.delete(); edges.delete(); col.delete(); edgesFull.delete();
      } finally { gray.delete(); }

      aligned.delete();
    } else {
      const pctx = preview.getContext('2d'); pctx.clearRect(0, 0, preview.width, preview.height);
      const actx = alignedCanvas.getContext('2d'); actx.clearRect(0, 0, alignedCanvas.width, alignedCanvas.height);
      const sctx = sobelCanvas.getContext('2d'); sctx.clearRect(0, 0, sobelCanvas.width, sobelCanvas.height);
    }
    measureBadge.textContent = isFinite(measure) ? `${withGlasses ? 'With' : 'No'} Glasses (${measure.toFixed(3)})` : 'No face';
    measureBadge.classList.toggle('warn', withGlasses);
    measureBadge.classList.toggle('ok', !withGlasses);
  }

  // rows declared above to preserve file order
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

    const tmp = createScaledCanvasFromImage(img, 720);
    const results = await new Promise((resolve) => { resultsQueue.push(resolve); faceMesh.send({ image: tmp }); });

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
        const pctx = preview.getContext('2d'); pctx.clearRect(0, 0, preview.width, preview.height);
        const actx = alignedCanvas.getContext('2d'); actx.clearRect(0, 0, alignedCanvas.width, alignedCanvas.height);
        const sctx = sobelCanvas.getContext('2d'); sctx.clearRect(0, 0, sobelCanvas.width, sobelCanvas.height);
        measureBadge.textContent = 'No face'; measureBadge.classList.remove('ok', 'warn');
      } else {
        let aligned = getAlignedFaceFromCenters(tmp, leftCenter, rightCenter, 256, 256);
        const alignedFlipped = new cv.Mat(); cv.flip(aligned, alignedFlipped, 0); aligned.delete(); aligned = alignedFlipped;
        cv.imshow(alignedCanvas, aligned);

        const gray = new cv.Mat();
        try {
          const ch = aligned.channels();
          if (ch === 4) cv.cvtColor(aligned, gray, cv.COLOR_RGBA2GRAY);
          else if (ch === 3) cv.cvtColor(aligned, gray, cv.COLOR_RGB2GRAY);
          else aligned.copyTo(gray);

          const H = aligned.rows; const W = aligned.cols;
          const stripW = Math.max(4, Math.round(W * 0.08));
          const x0 = Math.round(W * 0.5 - stripW * 0.5);
          const y0 = Math.round(H * 0.30);
          const h0 = Math.max(8, Math.round(H * 0.35));
          const roiRect = new cv.Rect(x0, y0, Math.min(stripW, W - x0), Math.min(h0, H - y0));
          const roi = gray.roi(roiRect);

          const blurred = new cv.Mat();
          cv.GaussianBlur(roi, blurred, new cv.Size(3, 3), 0, 0, cv.BORDER_DEFAULT);
          const edges = new cv.Mat();
          cv.Canny(blurred, edges, 100, 200);
          const centerX = Math.floor(edges.cols / 2);
          const col = edges.col(centerX);
          const nz = cv.countNonZero(col);
          measure = nz / (col.rows || 1);
          withGlasses = nz > 0;
          const edgesFull = cv.Mat.zeros(aligned.rows, aligned.cols, cv.CV_8UC1);
          const dstRoi = edgesFull.roi(roiRect);
          edges.copyTo(dstRoi);
          dstRoi.delete();
          drawPreview({ preview, aligned: alignedCanvas, sobel: sobelCanvas }, img, edgesFull, measure);
          roi.delete(); blurred.delete(); edges.delete(); col.delete(); edgesFull.delete();
        } finally { gray.delete(); }

        aligned.delete();
      }
    }

    const filePath = f.webkitRelativePath || f.name;
    rows.push({ path: filePath, measure, withGlasses });
    measureBadge.textContent = isFinite(measure) ? `${withGlasses ? 'With' : 'No'} Glasses (${measure.toFixed(3)})` : 'No face';
    measureBadge.classList.toggle('warn', withGlasses);
    measureBadge.classList.toggle('ok', !withGlasses);
    if (gridEl) {
      const thumb = document.createElement('div');
      let truth = null;
      const pathForTruth = (f.webkitRelativePath || filePath || '').toLowerCase();
      const parts = pathForTruth.split('/');
      const parent = parts.length > 1 ? parts[parts.length - 2] : '';
      if (parent === 'glasses') truth = true;
      else if (parent === 'no_glasses') truth = false;
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
      thumbImg.src = img.src; thumbImg.alt = filePath; thumb.appendChild(thumbImg);
      const cap = document.createElement('div'); cap.className = 'cap'; cap.textContent = filePath.split('/').slice(-1)[0]; thumb.appendChild(cap);
      thumb.dataset.measure = String(measure);
      thumb.dataset.path = filePath;
      if (truth !== null) thumb.dataset.truth = truth ? '1' : '0';
      gridEl.appendChild(thumb);
      thumb.addEventListener('click', () => {
        for (const el of gridEl.querySelectorAll('.thumb.selected')) el.classList.remove('selected');
        thumb.classList.add('selected');
        showResultsForImage(img);
        console.log('[edge-batch] clicked', { filePath, measure, withGlasses });
      });
      processedCount = Math.min(totalFiles, (i + 1));
      reclassifyGridAndCsv();
    }
    await new Promise(r => setTimeout(r));
  }

  reclassifyGridAndCsv();
}

document.getElementById('startBtn').addEventListener('click', async () => {
  const input = document.getElementById('dirInput');
  const allFiles = Array.from(input.files || []);
  const imgFiles = allFiles.filter(f => /\.(jpg|jpeg|png|bmp|webp)$/i.test(f.name));
  if (imgFiles.length === 0) { alert('Please choose a folder with images.'); return; }
  function parentFolderName(file) {
    const p = (file.webkitRelativePath || file.name).split('/');
    return p.length > 1 ? p[p.length - 2].toLowerCase() : '';
  }
  function classifyFolder(name) {
    const s = String(name || '').toLowerCase();
    const t = s.replace(/[^a-z]/g, '');
    if (t === 'glasses' || t === 'withglasses') return 'glasses';
    if (t === 'noglasses' || t === 'withoutglasses' || t === 'noeyeglasses') return 'no_glasses';
    return '';
  }
  const glassesFiles = imgFiles.filter(f => classifyFolder(parentFolderName(f)) === 'glasses');
  const noGlassesFiles = imgFiles.filter(f => classifyFolder(parentFolderName(f)) === 'no_glasses');
  const interleaved = [];
  let i = 0, j = 0;
  while (i < glassesFiles.length || j < noGlassesFiles.length) {
    if (i < glassesFiles.length) interleaved.push(glassesFiles[i++]);
    if (j < noGlassesFiles.length) interleaved.push(noGlassesFiles[j++]);
  }
  const files = interleaved.length > 0 ? interleaved : imgFiles;
  document.getElementById('downloadCsv').style.display = 'none';
  document.getElementById('progress').value = 0;
  document.getElementById('status').textContent = 'Starting...';
  runBatch(files, null);
});

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