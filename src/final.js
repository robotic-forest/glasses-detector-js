// Wrapper UI to test the pure algorithm from algorithms/glasses-detector.js
// Uses MediaPipe FaceMesh or face-api.js as the landmark provider and calls detectGlasses

import detectGlasses from '../algorithms/glasses-detector.js';

function createCsv(rows) {
  const header = ['path', 'measure', 'withGlasses'];
  const lines = [header.join(',')].concat(rows.map(r => [r.path, isFinite(r.measure) ? r.measure.toFixed(6) : 'NaN', r.withGlasses ? 1 : 0].join(',')));
  return new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
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

async function main() {
  const statusEl = document.getElementById('status');
  const progressEl = document.getElementById('progress');
  const summaryEl = document.getElementById('summary');
  const downloadEl = document.getElementById('downloadCsv');
  const measureBadge = document.getElementById('measureBadge');
  const gridEl = document.getElementById('thumbGrid');
  const thresholdInput = document.getElementById('thresholdInput');
  const thresholdLabel = document.getElementById('thresholdLabel');
  const providerSelect = document.getElementById('providerSelect');

  const preview = document.getElementById('preview');
  const alignedCanvas = document.getElementById('aligned');
  const sobelCanvas = document.getElementById('sobel');
  alignedCanvas.style.width = '256px'; alignedCanvas.style.height = '256px';
  sobelCanvas.style.width = '256px'; sobelCanvas.style.height = '256px';
  alignedCanvas.width = 256; alignedCanvas.height = 256;
  sobelCanvas.width = 256; sobelCanvas.height = 256;

  gridEl.innerHTML = '';
  thresholdInput.value = '0.07';
  thresholdInput.disabled = true;
  thresholdLabel.textContent = '0.07';

  // Maintain state for reclassification on threshold changes
  let rows = [];
  let processedCount = 0;
  let totalFiles = 0;

  // MediaPipe FaceMesh setup
  const { FaceMesh } = window;
  const faceMesh = new FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4/${file}` });
  faceMesh.setOptions({ staticImageMode: true, maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
  const resultsQueue = [];
  faceMesh.onResults((res) => { const r = resultsQueue.shift(); if (r) r(res); });

  // face-api.js setup (loaded via script tag in final.html)
  async function waitForFaceApi() {
    return new Promise(resolve => {
      if (window.faceapi) return resolve();
      const timer = setInterval(() => { if (window.faceapi) { clearInterval(timer); resolve(); } }, 50);
    });
  }
  let faceApiModelsLoaded = false;
  let faceApiDetectorType = null; // 'tiny' | 'ssd'
  async function ensureFaceApiModels(modelBase = './models') {
    await waitForFaceApi();
    const fa = window.faceapi;
    if (faceApiModelsLoaded) return { fa, detectorType: faceApiDetectorType };
    // Try TinyFaceDetector first; if unavailable, fall back to SSD Mobilenet
    try {
      await fa.nets.tinyFaceDetector.loadFromUri(modelBase);
      faceApiDetectorType = 'tiny';
    } catch (e) {
      console.warn('[final] tinyFaceDetector not found, trying ssdMobilenetv1', e);
      await fa.nets.ssdMobilenetv1.loadFromUri(modelBase);
      faceApiDetectorType = 'ssd';
    }
    await fa.nets.faceLandmark68Net.loadFromUri(modelBase);
    faceApiModelsLoaded = true;
    return { fa, detectorType: faceApiDetectorType };
  }

  function reclassifyGridAndCsv(rows, processedCount, totalFiles) {
    const thumbs = Array.from(gridEl.children || []);
    let gtTotal = 0; let gtCorrect = 0; let gtFp = 0; let gtFn = 0;
    for (const t of thumbs) {
      const measureVal = parseFloat(t.dataset.measure || 'NaN');
      const predicted = (t.dataset && t.dataset.wg === '1');
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
        ? `${predicted ? 'With' : 'No'} Glasses (${isFinite(measureVal) ? measureVal.toFixed(3) : 'NaN'})${hasTruth ? ` • GT: ${truth ? 'With' : 'No'} Glasses` : ''}`
        : 'No face detected';
    }
    const selected = gridEl.querySelector('.thumb.selected');
    if (selected) {
      const m = parseFloat(selected.dataset.measure || 'NaN');
      const wgSel = (selected.dataset && selected.dataset.wg === '1');
      measureBadge.textContent = isFinite(m) ? `${wgSel ? 'With' : 'No'} Glasses (${m.toFixed(3)})` : 'No face';
      measureBadge.classList.toggle('warn', wgSel);
      measureBadge.classList.toggle('ok', !wgSel);
    }
    const rowsForCsv = rows.map(r => ({ path: r.path, measure: r.measure, withGlasses: r.withGlasses }));
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

  let cancelled = false;
  document.getElementById('cancelBtn').onclick = () => { cancelled = true; };
  document.getElementById('startBtn').addEventListener('click', async () => {
    cancelled = false;
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
    const interleaved = []; let i = 0, j = 0;
    while (i < glassesFiles.length || j < noGlassesFiles.length) {
      if (i < glassesFiles.length) interleaved.push(glassesFiles[i++]);
      if (j < noGlassesFiles.length) interleaved.push(noGlassesFiles[j++]);
    }
    const files = interleaved.length > 0 ? interleaved : imgFiles;

    downloadEl.style.display = 'none';
    progressEl.value = 0; statusEl.textContent = 'Starting...'; gridEl.innerHTML = '';

    rows = []; processedCount = 0; totalFiles = files.length;
    const provider = (providerSelect && providerSelect.value) || 'mediapipe';

    // Prepare provider-specific dependencies
    let faceApi = null;
    let faceApiDetector = null;
    if (provider === 'faceapi') {
      try {
        const { fa, detectorType } = await ensureFaceApiModels('./models');
        faceApi = fa;
        faceApiDetector = detectorType;
      } catch (e) {
        console.error('[final] Failed to load face-api models', e);
        alert('Failed to load face-api models. Ensure ./models contains the model files.');
        return;
      }
    }

    for (let idx = 0; idx < files.length; idx++) {
      if (cancelled) break;
      const f = files[idx];
      statusEl.textContent = `Processing ${idx + 1}/${files.length}: ${f.webkitRelativePath || f.name}`;
      progressEl.value = Math.round(((idx + 1) / files.length) * 100);

      const img = await new Promise((resolve, reject) => {
        const url = URL.createObjectURL(f);
        const im = new Image();
        im.onload = () => resolve(im);
        im.onerror = reject;
        im.src = url;
      });

      const tmp = createScaledCanvasFromImage(img, 720);

      let measure = NaN; let withGlasses = false;
      if (provider === 'mediapipe') {
        const results = await new Promise((resolve) => { resultsQueue.push(resolve); faceMesh.send({ image: tmp }); });
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const landmarks = results.multiFaceLandmarks[0];
          const res = detectGlasses('mediapipe', landmarks, tmp);
          measure = res.measure; withGlasses = res.withGlasses;
        }
      } else if (provider === 'faceapi' && faceApi) {
        try {
          const options = faceApiDetector === 'ssd'
            ? new faceApi.SsdMobilenetv1Options({ minConfidence: 0.5 })
            : new faceApi.TinyFaceDetectorOptions();
          const det = await faceApi.detectSingleFace(tmp, options).withFaceLandmarks();
          if (det && det.landmarks && det.landmarks.positions) {
            const landmarks = det.landmarks.positions; // array of {x,y}
            const res = detectGlasses('faceapi', landmarks, tmp);
            measure = res.measure; withGlasses = res.withGlasses;
          }
        } catch (e) {
          console.warn('[final] face-api detection failed', e);
        }
      }

      const filePath = f.webkitRelativePath || f.name;
      rows.push({ path: filePath, measure, withGlasses });
      const thumb = document.createElement('div');
      let truth = null;
      const parts = (filePath || '').toLowerCase().split('/');
      const parent = parts.length > 1 ? parts[parts.length - 2] : '';
      if (parent === 'glasses') truth = true; else if (parent === 'no_glasses') truth = false;
      const cls = (() => {
        if (truth === null) return withGlasses ? 'warn' : 'ok';
        if (withGlasses === truth) return withGlasses ? 'blue' : 'ok';
        return withGlasses ? 'warn' : 'yellow';
      })();
      thumb.className = `thumb ${cls}`;
      thumb.title = isFinite(measure)
        ? `${withGlasses ? 'With' : 'No'} Glasses (${isFinite(measure) ? measure.toFixed(3) : 'NaN'})${truth === null ? '' : ` • GT: ${truth ? 'With' : 'No'} Glasses`}`
        : 'No face detected';
      const thumbImg = document.createElement('img');
      thumbImg.src = img.src; thumbImg.alt = filePath; thumb.appendChild(thumbImg);
      const cap = document.createElement('div'); cap.className = 'cap'; cap.textContent = filePath.split('/').slice(-1)[0]; thumb.appendChild(cap);
      thumb.dataset.measure = String(measure);
      thumb.dataset.path = filePath;
      thumb.dataset.wg = withGlasses ? '1' : '0';
      if (truth !== null) thumb.dataset.truth = truth ? '1' : '0';
      gridEl.appendChild(thumb);

      // For preview panes
      document.getElementById('measureBadge').textContent = isFinite(measure) ? `${withGlasses ? 'With' : 'No'} Glasses (${measure.toFixed(3)})` : 'No face';
      document.getElementById('measureBadge').classList.toggle('warn', withGlasses);
      document.getElementById('measureBadge').classList.toggle('ok', !withGlasses);

      processedCount = Math.min(totalFiles, idx + 1);
      reclassifyGridAndCsv(rows, processedCount, totalFiles);
      await new Promise(r => setTimeout(r));
    }

    reclassifyGridAndCsv(rows, processedCount, totalFiles);
  });

  (function initThresholdLabel() {
    const thrInput = document.getElementById('thresholdInput');
    const thrLabel = document.getElementById('thresholdLabel');
    thrLabel.textContent = String(parseFloat(thrInput.value).toFixed(2));
    thrInput.addEventListener('input', () => {
      thrLabel.textContent = String(parseFloat(thrInput.value).toFixed(2));
      // Reclassify existing results live when the threshold changes
      reclassifyGridAndCsv(rows, processedCount, totalFiles);
    });
  })();
}

main();


