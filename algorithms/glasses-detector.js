// Edge-based glasses detector as a pure function (no DOM interactions)
// Pure Canvas implementation (no OpenCV) of the nasal-bridge edge measure
// Supports landmark providers: "mediapipe" (FaceMesh) and "faceapi" (68-point)

/**
 * Compute least-squares line y = kx + b through a set of 2D points.
 * @param {Array<[number, number]>} points
 * @returns {{k:number,b:number}}
 */
function leastSquaresKAndB(points) {
  const n = points.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (const [x, y] of points) { sumX += x; sumY += y; sumXY += x * y; sumXX += x * x; }
  const denom = (n * sumXX - sumX * sumX) + 1e-6;
  const k = (n * sumXY - sumX * sumY) / denom;
  const b = (sumY - k * sumX) / n;
  return { k, b };
}

/**
 * Map eye corner points depending on provider.
 * - mediapipe: expects normalized landmarks (x,y in [0,1]) from FaceMesh
 * - faceapi: expects 68-point landmark array with pixel coordinates
 * @param {"mediapipe"|"faceapi"} provider
 * @param {any} landmarks
 * @param {number} imgW
 * @param {number} imgH
 * @returns {{ rOuter:[number,number], rInner:[number,number], lInner:[number,number], lOuter:[number,number] }}
 */
function getEyeCornerPoints(provider, landmarks, imgW, imgH) {
  if (provider === 'mediapipe') {
    const idx = { rOuter: 33, rInner: 133, lInner: 362, lOuter: 263 };
    const scale = p => ([p.x * imgW, p.y * imgH]);
    const rOuter = scale(landmarks[idx.rOuter]);
    const rInner = scale(landmarks[idx.rInner]);
    const lInner = scale(landmarks[idx.lInner]);
    const lOuter = scale(landmarks[idx.lOuter]);
    return { rOuter, rInner, lInner, lOuter };
  }
  if (provider === 'faceapi') {
    // dlib 68 landmarks indexing per face-api.js FaceLandmarks68
    // Right eye: 36 (outer/temporal), 39 (inner)
    // Left eye: 42 (inner), 45 (outer/temporal)
    // Docs: https://justadudewhohacks.github.io/face-api.js/docs/index.html
    const safePt = idx => {
      const p = landmarks[idx];
      // face-api.js returns pixels already; fall back to object with x,y
      if (!p) throw new Error(`faceapi landmarks missing point ${idx}`);
      const x = typeof p.x === 'number' ? p.x : p[0];
      const y = typeof p.y === 'number' ? p.y : p[1];
      return [x, y];
    };
    const rOuter = safePt(36);
    const rInner = safePt(39);
    const lInner = safePt(42);
    const lOuter = safePt(45);
    return { rOuter, rInner, lInner, lOuter };
  }
  throw new Error(`Unsupported landmark_provider: ${provider}`);
}

/**
 * Compute left/right eye centers by projecting the midpoints to the eye-line fit.
 * @param {number} imgW
 * @param {number} imgH
 * @param {{ rOuter:[number,number], rInner:[number,number], lInner:[number,number], lOuter:[number,number] }} corners
 */
function getCentersFromCorners(imgW, imgH, corners) {
  const { rOuter, rInner, lInner, lOuter } = corners;
  const pts = [rOuter, rInner, lInner, lOuter];
  const { k, b } = leastSquaresKAndB(pts);
  const xLeft = (lOuter[0] + lInner[0]) / 2;
  const xRight = (rOuter[0] + rInner[0]) / 2;
  const leftCenter = [Math.round(xLeft), Math.round(xLeft * k + b)];
  const rightCenter = [Math.round(xRight), Math.round(xRight * k + b)];
  return { leftCenter, rightCenter };
}

/**
 * Create an aligned face by rotating/scaling about eye centers to desired size using Canvas.
 * Returns a Canvas (OffscreenCanvas if available) of size desiredW x desiredH, vertically flipped
 * to match the measurement orientation used previously.
 * @param {HTMLCanvasElement|HTMLImageElement|ImageData|{width:number,height:number,data?:Uint8ClampedArray}} src
 * @param {[number,number]} leftCenter
 * @param {[number,number]} rightCenter
 * @param {number} desiredW
 * @param {number} desiredH
 * @returns {HTMLCanvasElement|OffscreenCanvas}
 */
function getAlignedFaceCanvas(src, leftCenter, rightCenter, desiredW = 256, desiredH = 256) {
  function createCanvas(w, h) {
    if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
    const c = document.createElement('canvas'); c.width = w; c.height = h; return c;
  }
  function ensureSourceCanvas(imageLike) {
    // If it's an HTMLCanvasElement, return as-is
    if (typeof HTMLCanvasElement !== 'undefined' && imageLike instanceof HTMLCanvasElement) return imageLike;
    // If ImageData or generic {width,height,data}
    if ((typeof ImageData !== 'undefined' && imageLike instanceof ImageData) || (imageLike && imageLike.width && imageLike.height && imageLike.data)) {
      const c = createCanvas(imageLike.width, imageLike.height);
      const ctx = c.getContext('2d');
      const id = (typeof ImageData !== 'undefined' && imageLike instanceof ImageData)
        ? imageLike
        : new ImageData(new Uint8ClampedArray(imageLike.data), imageLike.width, imageLike.height);
      ctx.putImageData(id, 0, 0);
      return c;
    }
    // If HTMLImageElement
    if (typeof HTMLImageElement !== 'undefined' && imageLike instanceof HTMLImageElement) {
      const c = createCanvas(imageLike.naturalWidth || imageLike.width, imageLike.naturalHeight || imageLike.height);
      const ctx = c.getContext('2d');
      ctx.drawImage(imageLike, 0, 0);
      return c;
    }
    throw new Error('Unsupported src image type');
  }

  const srcCanvas = ensureSourceCanvas(src);
  const imgW = srcCanvas.width; const imgH = srcCanvas.height;

  const dx = rightCenter[0] - leftCenter[0];
  const dy = rightCenter[1] - leftCenter[1];
  const dist = Math.hypot(dx, dy);
  const desiredDist = desiredW * 0.5;
  const scale = desiredDist / (dist || 1);
  const angle = Math.atan2(dy, dx); // radians
  const eyesCenterX = (leftCenter[0] + rightCenter[0]) * 0.5;
  const eyesCenterY = (leftCenter[1] + rightCenter[1]) * 0.5;

  const out = createCanvas(desiredW, desiredH);
  const ctx = out.getContext('2d');
  // Flip vertically after drawing by applying scale(1,-1) and translating
  ctx.translate(desiredW / 2, desiredH / 2);
  ctx.scale(1, -1);
  ctx.rotate(angle);
  ctx.scale(scale, scale);
  ctx.translate(-eyesCenterX, -eyesCenterY);
  ctx.drawImage(srcCanvas, 0, 0, imgW, imgH);
  return out;
}

/**
 * Compute nasal-bridge edge measure on an aligned face image canvas (256x256 default).
 * - Convert to grayscale
 * - Crop a narrow vertical strip centered horizontally, below the eyes
 * - Apply a small box blur and compute vertical gradient
 * - Count pixels above a gradient threshold along the center column
 * @param {HTMLCanvasElement|OffscreenCanvas} alignedCanvas
 * @returns {{ measure:number, withGlasses:boolean }}
 */
function computeBridgeEdgeMeasureCanvas(alignedCanvas) {
  const ctx = alignedCanvas.getContext('2d');
  const W = alignedCanvas.width; const H = alignedCanvas.height;
  const stripW = Math.max(4, Math.round(W * 0.08));
  const x0 = Math.round(W * 0.5 - stripW * 0.5);
  const y0 = Math.round(H * 0.30);
  const h0 = Math.max(8, Math.round(H * 0.35));
  const { data, width, height } = ctx.getImageData(x0, y0, stripW, h0);

  // Convert to grayscale
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    gray[p] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  // Simple 3x3 box blur
  const blur = new Float32Array(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let s = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          s += gray[(y + ky) * width + (x + kx)];
        }
      }
      blur[y * width + x] = s / 9;
    }
  }

  // Vertical gradient along the center column
  const cx = Math.floor(width / 2);
  let nz = 0; const denom = Math.max(0, height - 2);
  // Adaptive threshold based on intensity scale
  // Use a fixed threshold tuned for 8-bit grayscale
  const threshold = 12; // tweak if needed
  for (let y = 1; y < height - 1; y++) {
    const gy = blur[(y - 1) * width + cx] - blur[(y + 1) * width + cx];
    if (Math.abs(gy) > threshold) nz++;
  }
  const measure = nz / (denom || 1);
  // Classification threshold tuned empirically; see UI notes.
  const withGlasses = measure > 0.07;
  return { measure, withGlasses };
}

/**
 * Detect eyeglasses using an edge-based nasal-bridge measure.
 * Returns { measure, withGlasses }. Does not interact with any DOM.
 *
 * @param {"mediapipe"|"faceapi"} landmark_provider
 * @param {any} landmarks - provider-specific landmarks
 *   - mediapipe: array of {x:[0..1], y:[0..1], z?}
 *   - faceapi: 68-point array with pixel coordinates ({ x:number, y:number } or [x,y])
 * @param {HTMLCanvasElement|HTMLImageElement|ImageData|{width:number,height:number,data?:Uint8ClampedArray}} src - source image
 * @returns {{ measure:number, withGlasses:boolean }}
 */
export function detectGlasses(landmark_provider, landmarks, src) {
  // Derive image dimensions from src
  let imgW = 0, imgH = 0;
  if (typeof HTMLCanvasElement !== 'undefined' && src instanceof HTMLCanvasElement) { imgW = src.width; imgH = src.height; }
  else if (typeof HTMLImageElement !== 'undefined' && src instanceof HTMLImageElement) { imgW = src.naturalWidth || src.width; imgH = src.naturalHeight || src.height; }
  else if (typeof ImageData !== 'undefined' && src instanceof ImageData) { imgW = src.width; imgH = src.height; }
  else if (src && src.width && src.height) { imgW = src.width; imgH = src.height; }
  else { throw new Error('Unsupported or missing src for size extraction'); }

  const corners = getEyeCornerPoints(landmark_provider, landmarks, imgW, imgH);
  const { leftCenter, rightCenter } = getCentersFromCorners(imgW, imgH, corners);
  const eyeDx = rightCenter[0] - leftCenter[0];
  const eyeDy = rightCenter[1] - leftCenter[1];
  const eyeDist = Math.hypot(eyeDx, eyeDy);
  const dynamicMinEyeDist = Math.max(40, Math.round(0.055 * Math.min(imgW, imgH)));
  if (!isFinite(eyeDist) || eyeDist < dynamicMinEyeDist) {
    return { measure: NaN, withGlasses: false };
  }

  const alignedCanvas = getAlignedFaceCanvas(src, leftCenter, rightCenter, 256, 256);
  return computeBridgeEdgeMeasureCanvas(alignedCanvas);
}

export default detectGlasses;


