const API_URL = "https://model-financial-crime-detection.onrender.com/predict"; // this the deployed url of the project on render

let requestCount  = 0;
let lastProb      = null;   // store last fraud probability for re-evaluation
let modelThreshold = null;  // threshold returned by the model

const form       = document.getElementById("fraud-form");
const scanBtn    = document.getElementById("scan-btn");
const reqCount   = document.getElementById("req-count");

const idleState    = document.getElementById("idle-state");
const loadingState = document.getElementById("loading-state");
const resultState  = document.getElementById("result-state");

const verdictBlock    = document.getElementById("verdict-block");
const verdictLabel    = document.getElementById("verdict-label");
const verdictText     = document.getElementById("verdict-text");
const probValue       = document.getElementById("prob-value");
const probBar         = document.getElementById("prob-bar");
const riskLevel       = document.getElementById("risk-level");
const thresholdDisplay = document.getElementById("threshold-display");
const decisionVal     = document.getElementById("decision-val");

const thresholdSlider = document.getElementById("threshold-slider");
const thresholdInput  = document.getElementById("threshold-input");
const reevalBtn       = document.getElementById("reeval-btn");

// ── Threshold controls sync ───────────────────────────────────────────────

thresholdSlider.addEventListener("input", () => {
  thresholdInput.value = parseFloat(thresholdSlider.value).toFixed(2);
});

thresholdInput.addEventListener("input", () => {
  let v = parseFloat(thresholdInput.value);
  if (isNaN(v)) return;
  v = Math.min(99.99, Math.max(0.01, v));
  thresholdSlider.value = Math.round(v);
});

// ── Helpers ───────────────────────────────────────────────────────────────

function setState(state) {
  idleState.classList.add("hidden");
  loadingState.classList.add("hidden");
  resultState.classList.add("hidden");
  document.getElementById(`${state}-state`).classList.remove("hidden");
}

function getBarColor(prob) {
  if (prob < 0.3)  return "#00ff9d";
  if (prob < 0.6)  return "#ffb547";
  return "#ff3b5c";
}

function getUserThreshold() {
  const v = parseFloat(thresholdInput.value);
  if (!isNaN(v) && v > 0 && v < 100) return v / 100;
  return modelThreshold;
}

function applyVerdict(prob, threshold) {
  const isFraud = prob >= threshold;

  // risk level based on prob alone
  let risk;
  if (prob < 0.3)       risk = "LOW";
  else if (prob < 0.6)  risk = "MEDIUM";
  else                  risk = "HIGH";

  // verdict display
  verdictBlock.className = "verdict-block";
  if (isFraud) {
    verdictBlock.classList.add("fraud");
    verdictLabel.textContent = "FRAUDULENT";
    verdictText.textContent  = "Transaction flagged — high risk activity detected";
  } else if (risk === "MEDIUM") {
    verdictBlock.classList.add("warn");
    verdictLabel.textContent = "SUSPICIOUS";
    verdictText.textContent  = "Transaction within normal range but elevated risk";
  } else {
    verdictBlock.classList.add("safe");
    verdictLabel.textContent = "LEGITIMATE";
    verdictText.textContent  = "Transaction appears normal — no anomalies detected";
  }

  // meta
  riskLevel.textContent     = risk;
  riskLevel.style.color     = risk === "HIGH" ? "#ff3b5c" : risk === "MEDIUM" ? "#ffb547" : "#00ff9d";
  thresholdDisplay.textContent = (threshold * 100).toFixed(2) + "%";
  decisionVal.textContent   = isFraud ? "BLOCKED" : "APPROVED";
  decisionVal.style.color   = isFraud ? "#ff3b5c" : "#00ff9d";
}

function renderResult(data) {
  const prob      = data.fraud_probability;
  modelThreshold  = data.threshold_used;
  lastProb        = prob;

  // set slider to model threshold on first result
  const pct = Math.round(modelThreshold * 100);
  thresholdSlider.value = pct;
  thresholdInput.value  = (modelThreshold * 100).toFixed(2);

  // probability bar
  probValue.textContent = (prob * 100).toFixed(2) + "%";
  setTimeout(() => {
    probBar.style.width      = (prob * 100) + "%";
    probBar.style.background = getBarColor(prob);
  }, 100);

  applyVerdict(prob, modelThreshold);
  setState("result");
}

// ── Re-evaluate at custom threshold ──────────────────────────────────────

reevalBtn.addEventListener("click", () => {
  if (lastProb === null) return;
  const customThreshold = getUserThreshold();
  applyVerdict(lastProb, customThreshold);

  // animate bar color update
  probBar.style.background = getBarColor(lastProb);
});

// ── Form submit ───────────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setState("loading");
  scanBtn.disabled = true;

  const raw = {
    TransactionAmt: parseFloat(form.TransactionAmt.value) || null,
    card4:          form.card4.value          || null,
    card6:          form.card6.value          || null,
    P_emaildomain:  form.P_emaildomain.value  || null,
    ProductCD:      form.ProductCD.value      || null,
    DeviceType:     form.DeviceType.value     || null,
  };

  const payload = Object.fromEntries(
    Object.entries(raw).filter(([_, v]) => v !== null)
  );

  try {
    const res = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    requestCount++;
    reqCount.textContent = `${requestCount} REQUEST${requestCount !== 1 ? "S" : ""}`;
    renderResult(data);

  } catch (err) {
    setState("result");
    verdictBlock.className   = "verdict-block fraud";
    verdictLabel.textContent = "ERROR";
    verdictText.textContent  = err.message || "Could not reach the API. Is the backend running?";
    probValue.textContent    = "—";
    probBar.style.width      = "0%";
    riskLevel.textContent    = "—";
    thresholdDisplay.textContent = "—";
    decisionVal.textContent  = "—";
  } finally {
    scanBtn.disabled = false;
  }
});