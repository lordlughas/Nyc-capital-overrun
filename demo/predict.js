// LightGBM tree evaluator for the exported model (demo/model.json).
// x: array in model.features order; categorical values are integer codes, missing = null.
function walk(node, x) {
  while (node.v === undefined) {
    const val = x[node.f];
    let left;
    if (node.c) {
      // Categorical split: listed codes go left; missing or unseen go right
      left = val !== null && val >= 0 && node.c.includes(val);
    } else if (val === null || Number.isNaN(val)) {
      left = node.m === "None" ? 0 <= node.t : node.dl;
    } else if (node.m === "Zero" && val === 0) {
      left = node.dl;
    } else {
      left = val <= node.t;
    }
    node = left ? node.l : node.r;
  }
  return node.v;
}

function rawScore(model, x) {
  let s = 0;
  for (const t of model.trees) s += walk(t, x);
  return 1 / (1 + Math.exp(-s));
}

function calibrate(model, p) {
  const { x, y } = model.calibration;
  if (p <= x[0]) return y[0];
  if (p >= x[x.length - 1]) return y[y.length - 1];
  let i = 1;
  while (x[i] < p) i++;
  const f = (p - x[i - 1]) / (x[i] - x[i - 1]);
  return y[i - 1] + f * (y[i] - y[i - 1]);
}

if (typeof module !== "undefined") module.exports = { rawScore, calibrate };
