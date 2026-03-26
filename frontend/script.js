const API_URL = "http://localhost:8000/predict"; // change to Render URL before deploying

let requestCount = 0;

const form       = document.getElementById("fraud-form");
const scanBtn    = document.getElementById("scan-btn");
const reqCount   = document.getElementById("req-count");

const idleState    = document.getElementById("idle-state");
const loadingState = document.getElementById("loading-state");
const resultState  = document.getElementById("result-state");

const verdictBlock = document.getElementById("verdict-block");
const verdictLabel = document.getElementById("verdict-label");
const verdictText  = document.getElementById("verdict-text");
const probValue    = document.getElementById("prob-value");
const probBar      = document.getElementById("prob-bar");
const riskLevel    = document.getElementById("risk-level");
const thresholdVal = document.getElementById("threshold-val");
const decisionVal  = document.getElementById("decision-val");

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

function renderResult(data) {
  const prob     = data.fraud_probability;
  const isFraud  = data.is_fraud;
  const risk     = data.risk_level;
  const thresh   = data.threshold_used;

  // verdict
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

  // probability bar
  probValue.textContent = (prob * 100).toFixed(2) + "%";
  setTimeout(() => {
    probBar.style.width      = (prob * 100) + "%";
    probBar.style.background = getBarColor(prob);
  }, 100);

  // meta
  riskLevel.textContent  = risk;
  riskLevel.style.color  = risk === "HIGH" ? "#ff3b5c" : risk === "MEDIUM" ? "#ffb547" : "#00ff9d";
  thresholdVal.textContent = (thresh * 100).toFixed(1) + "%";
  decisionVal.textContent  = isFraud ? "BLOCKED" : "APPROVED";
  decisionVal.style.color  = isFraud ? "#ff3b5c" : "#00ff9d";

  setState("result");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  setState("loading");
  scanBtn.disabled = true;

  // build payload — only send non-empty fields
  const raw = {
    TransactionAmt: parseFloat(form.TransactionAmt.value) || null,
    card4:          form.card4.value          || null,
    card6:          form.card6.value          || null,
    P_emaildomain:  form.P_emaildomain.value  || null,
    ProductCD:      form.ProductCD.value      || null,
    DeviceType:     form.DeviceType.value     || null,
  };

  // remove nulls
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
    thresholdVal.textContent = "—";
    decisionVal.textContent  = "—";
  } finally {
    scanBtn.disabled = false;
  }
});