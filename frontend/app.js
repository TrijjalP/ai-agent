// ===== Config =====
const API_BASE = "http://localhost:8000";
const POLL_INTERVAL_MS = 2500;

// ===== Element refs =====
const topicInput = document.getElementById("topicInput");
const globalToggle = document.getElementById("globalToggle");
const globalToggleLabel = document.getElementById("globalToggleLabel");
const yearFrom = document.getElementById("yearFrom");
const yearTo = document.getElementById("yearTo");
const runBtn = document.getElementById("runBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const fileInput = document.getElementById("fileInput");
const fileList = document.getElementById("fileList");
const fileCount = document.getElementById("fileCount");
const balancePill = document.getElementById("balancePill");

const idleState = document.getElementById("idleState");
const loadingState = document.getElementById("loadingState");
const resultsState = document.getElementById("resultsState");
const errorState = document.getElementById("errorState");
const loadingStage = document.getElementById("loadingStage");
const loadingSub = document.getElementById("loadingSub");
const errorTitle = document.getElementById("errorTitle");
const errorMessage = document.getElementById("errorMessage");

let selectedFiles = [];

// ===== Swap-papers state =====
// Only meaningful for topic-driven runs (uploads-only mode has no
// candidate_papers to swap in — you're comparing exactly what you uploaded).
let lastRunPayload = null;     // the topic/filters used for the last run, so a swap can re-run with the same settings
let lastUploadedPaths = [];    // [] unless the last run was uploads-only
let currentWorkingSet = [];    // the papers currently being compared (swappable)
let currentCandidates = [];    // the "other papers found" pool (swap-in sources)
let swapSourceId = null;       // paper_id in currentWorkingSet picked to be replaced, or null

// ===== Evidence priority sliders -> score_weights =====
const sliderIds = ["novelty", "clinical", "rigor", "specificity"];
const sliders = sliderIds.map((id) => document.getElementById(`slider-${id}`));
const sliderVals = sliderIds.map((id) => document.getElementById(`val-${id}`));

function normalizeSliders(changedIndex) {
  // Keep all 4 sliders summing to 100%, scaling the others proportionally
  // (same UX as the reference design's "Balanced" pill).
  const raw = sliders.map((s) => parseFloat(s.value));
  const total = raw.reduce((a, b) => a + b, 0);

  if (total === 0) {
    sliders.forEach((s) => (s.value = 25));
  } else {
    const scale = 100 / total;
    sliders.forEach((s, i) => {
      if (i !== changedIndex) {
        s.value = Math.max(0, Math.round(parseFloat(s.value) * scale * 100) / 100);
      }
    });
    // fix rounding drift on the last non-changed slider
    const newTotal = sliders.reduce((a, b) => a + parseFloat(b.value), 0);
    const drift = 100 - newTotal;
    const lastIdx = sliders.findIndex((_, i) => i !== changedIndex);
    if (lastIdx !== -1) {
      sliders[lastIdx].value = Math.max(0, parseFloat(sliders[lastIdx].value) + drift);
    }
  }

  sliders.forEach((s, i) => {
    sliderVals[i].textContent = `${Math.round(s.value)}%`;
  });
}

sliders.forEach((slider, i) => {
  slider.addEventListener("input", () => normalizeSliders(i));
});

function getScoreWeights() {
  const weights = {};
  sliders.forEach((s) => {
    const key = s.dataset.key;
    weights[key] = Math.round(parseFloat(s.value)) / 100;
  });
  return weights;
}

// ===== Year range =====
// Data covers 2020-2026, so keep both fields inside that range and in order.
const YEAR_MIN = 2020;
const YEAR_MAX = 2026;

function clampYearFields(changed) {
  let from = parseInt(yearFrom.value, 10);
  let to = parseInt(yearTo.value, 10);
  if (isNaN(from)) from = YEAR_MIN;
  if (isNaN(to)) to = YEAR_MAX;
  from = Math.min(Math.max(from, YEAR_MIN), YEAR_MAX);
  to = Math.min(Math.max(to, YEAR_MIN), YEAR_MAX);
  if (from > to) {
    if (changed === yearTo) from = to;
    else to = from;
  }
  yearFrom.value = from;
  yearTo.value = to;
}
yearFrom.addEventListener("change", () => clampYearFields(yearFrom));
yearTo.addEventListener("change", () => clampYearFields(yearTo));

// ===== Evidence scope label (Global / India based) =====
function updateGlobalToggleLabel() {
  globalToggleLabel.textContent = globalToggle.checked ? "Global" : "India based";
}
globalToggle.addEventListener("change", updateGlobalToggleLabel);

// ===== File upload (display only for now — see note in README) =====
fileInput.addEventListener("change", () => {
  const incoming = Array.from(fileInput.files).slice(0, 4 - selectedFiles.length);
  selectedFiles = selectedFiles.concat(incoming).slice(0, 4);
  renderFileList();
  fileInput.value = "";
});

function renderFileList() {
  fileList.innerHTML = "";
  selectedFiles.forEach((f, i) => {
    const chip = document.createElement("div");
    chip.className = "file-chip";
    chip.innerHTML = `<span>${f.name}</span>`;
    const btn = document.createElement("button");
    btn.textContent = "Remove";
    btn.addEventListener("click", () => {
      selectedFiles.splice(i, 1);
      renderFileList();
    });
    chip.appendChild(btn);
    fileList.appendChild(chip);
  });
  fileCount.textContent = `${selectedFiles.length} / 4`;
}

// ===== State machine for the results panel =====
function showState(name) {
  idleState.classList.add("hidden");
  loadingState.classList.add("hidden");
  resultsState.classList.add("hidden");
  errorState.classList.add("hidden");
  ({ idle: idleState, loading: loadingState, results: resultsState, error: errorState }[name]).classList.remove("hidden");

  // On phones the results panel sits below the form, so bring it into view.
  if (name !== "idle" && window.matchMedia("(max-width: 820px)").matches) {
    const panel = document.querySelector(".panel-results");
    if (panel) panel.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function setAgentStatus(label, busy) {
  statusText.textContent = label;
  statusDot.classList.toggle("busy", !!busy);
}

// ===== DNA helix animation (used in both idle and loading states) =====
function buildDna(containerId, rungCount = 14) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  for (let i = 0; i < rungCount; i++) {
    const rung = document.createElement("div");
    rung.className = "dna-rung";
    const delay = `${(i * -0.18).toFixed(2)}s`;

    const bar = document.createElement("div");
    bar.className = "dna-bar";
    bar.style.animationDelay = delay;

    const left = document.createElement("div");
    left.className = "dna-dot left";
    left.style.animationDelay = delay;

    const right = document.createElement("div");
    right.className = "dna-dot right";
    right.style.animationDelay = delay;

    rung.appendChild(bar);
    rung.appendChild(left);
    rung.appendChild(right);
    container.appendChild(rung);
  }
}

// ===== Upload attached files, return their server-side paths =====
async function uploadAttachedFiles() {
  if (selectedFiles.length === 0) return [];

  const formData = new FormData();
  selectedFiles.forEach((f) => formData.append("files", f));

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData
  });
  if (!res.ok) throw new Error(`File upload failed (server returned ${res.status})`);
  const data = await res.json();
  return data.uploaded_paths || [];
}

// ===== Run pipeline =====
// Topic is required UNLESS files are attached — if you've uploaded your
// own papers, you're comparing exactly those, so a topic is optional
// framing context rather than a search query.
runBtn.addEventListener("click", async () => {
  const topic = topicInput.value.trim();
  const hasFiles = selectedFiles.length > 0;

  if (!topic && !hasFiles) {
    topicInput.focus();
    topicInput.style.borderColor = "#b5652f";
    return;
  }
  topicInput.style.borderColor = "";

  runBtn.disabled = true;
  showState("loading");

  if (hasFiles) {
    loadingStage.textContent = "READING YOUR PAPERS";
    loadingSub.textContent = `Extracting text from ${selectedFiles.length} uploaded file(s) for direct comparison.`;
  } else {
    loadingStage.textContent = "RETRIEVING PAPERS";
    loadingSub.textContent = "Finding and selecting the most relevant papers for your topic.";
  }
  setAgentStatus("Agent running…", true);

  try {
    const uploadedPaths = await uploadAttachedFiles();

    const payload = {
      topic: topic || "Comparison of uploaded papers",
      india_only: !globalToggle.checked,
      year_from: yearFrom.value ? parseInt(yearFrom.value, 10) : null,
      year_to: yearTo.value ? parseInt(yearTo.value, 10) : null,
      gap_focus: "Compare these papers' methodology, conclusions, and differences",
      score_weights: getScoreWeights(),
      uploaded_file_paths: uploadedPaths,
      confirmed_paper_ids: null
    };

    // Remember this run's settings and reset any in-progress swap, so a
    // fresh "Run research pipeline" always starts a clean working set.
    lastRunPayload = payload;
    lastUploadedPaths = uploadedPaths;
    currentWorkingSet = [];
    currentCandidates = [];
    swapSourceId = null;

    const startRes = await fetch(`${API_BASE}/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!startRes.ok) throw new Error(`Server returned ${startRes.status}`);
    const { job_id } = await startRes.json();

    pollJob(job_id);
  } catch (err) {
    runBtn.disabled = false;
    showError("Could not reach the research agent", err.message);
  }
});

const loadingMessages = [
  ["RETRIEVING PAPERS", "Finding and selecting the most relevant papers for your topic."],
  ["COMPARING EVIDENCE", "Reading methodology, conclusions, and differences across papers."],
  ["GENERATING HYPOTHESES", "Drafting novel, evidence-grounded research directions."],
  ["CRITIQUING & RANKING", "Scoring each hypothesis for feasibility, novelty, and rigor."]
];
let loadingMsgIndex = 0;
let loadingMsgTimer = null;

function cycleLoadingMessages() {
  loadingMsgTimer = setInterval(() => {
    loadingMsgIndex = (loadingMsgIndex + 1) % loadingMessages.length;
    const [stage, sub] = loadingMessages[loadingMsgIndex];
    loadingStage.textContent = stage;
    loadingSub.textContent = sub;
  }, 4000);
}
cycleLoadingMessages();

async function pollJob(jobId) {
  try {
    const res = await fetch(`${API_BASE}/research/${jobId}`);
    const data = await res.json();

    if (data.status === "completed") {
      clearInterval(loadingMsgTimer);
      setAgentStatus("Agent ready", false);
      runBtn.disabled = false;
      renderResults(data.result);
      return;
    }
    if (data.status === "failed") {
      clearInterval(loadingMsgTimer);
      runBtn.disabled = false;
      showError("The pipeline hit an error", data.error || "Unknown error.");
      return;
    }
    // still running/queued — keep polling
    setTimeout(() => pollJob(jobId), POLL_INTERVAL_MS);
  } catch (err) {
    clearInterval(loadingMsgTimer);
    runBtn.disabled = false;
    showError("Lost connection to the research agent", err.message);
  }
}

function showError(title, message) {
  clearInterval(loadingMsgTimer);
  setAgentStatus("Agent ready", false);
  errorTitle.textContent = title;
  errorMessage.textContent = message;
  showState("error");
}

// ===== Render results =====
// NOTE: this is a first-pass layout using the same visual language as the
// input screen (cards, sage/rust accents, Playfair headings). Once you send
// the actual output-screen design, this function gets replaced to match it
// pixel-for-pixel — the data wiring (job polling, state shape) stays the same.
function renderResults(result) {
  if (!result.is_valid_topic) {
    showError("Not a biomedical research topic", result.validation_message || "This system only processes biomedical/medical/life sciences research topics.");
    return;
  }

  const selectedPapers = result.selected_papers || [];
  const candidatePapers = result.candidate_papers || [];
  const paperSummaries = result.paper_summaries || [];
  const recommended = result.recommended_paper || {};
  const rankedHypotheses = result.ranked_hypotheses || [];

  // Swapping only makes sense for topic-driven runs — uploads-only mode
  // compares exactly what was uploaded and has no candidate pool.
  const isUploadsMode = lastUploadedPaths.length > 0;
  currentWorkingSet = selectedPapers.map((p) => ({
    paper_id: p.paper_id,
    title: p.title || p.paper_title || "Untitled",
    source: p.source,
    published: p.published,
    is_india_affiliated: p.is_india_affiliated
  }));
  currentCandidates = isUploadsMode ? [] : candidatePapers.map((p) => ({
    paper_id: p.paper_id,
    title: p.title || p.paper_title || "Untitled",
    source: p.source,
    published: p.published,
    is_india_affiliated: p.is_india_affiliated
  }));
  swapSourceId = null;

  let html = `<h2>Research map</h2><p class="hero-sub" style="margin-bottom:24px;">${escapeHtml(topicInput.value.trim())}</p>`;

  html += `<div id="paperSwapSection">${renderPaperSwapSection()}</div>`;

  // Comparison summaries
  if (paperSummaries.length) {
    html += `<div class="result-section"><h3>Methodology &amp; conclusions</h3>`;
    paperSummaries.forEach((s) => {
      const title = s.title || s.paper_title || "Untitled";
      const methodology = s.methodology || s.methodology_summary || "";
      const conclusion = s.conclusion || s.conclusion_summary || "";
      html += `<div class="paper-row">
        <div class="paper-title">${escapeHtml(title)}</div>
        <div class="paper-meta"><strong>Methodology:</strong> ${escapeHtml(methodology)}</div>
        <div class="paper-meta"><strong>Conclusion:</strong> ${escapeHtml(conclusion)}</div>
      </div>`;
    });
    html += `</div>`;
  }

  // Recommended paper
  if (recommended && (recommended.paper_title || recommended.title)) {
    html += `<div class="result-section">
      <h3><span class="badge rust">Recommended</span> ${escapeHtml(recommended.paper_title || recommended.title)}</h3>
      <p class="paper-meta">${escapeHtml(recommended.rationale || "")}</p>
    </div>`;
  }

  // Ranked hypotheses
  if (rankedHypotheses.length) {
    html += `<div class="result-section"><h3>Ranked hypotheses</h3>`;
    rankedHypotheses.forEach((h) => {
      html += `<div class="hyp-card">
        <div class="hyp-rank">RANK #${h.rank} &middot; ${h.overall_soundness_score}/10</div>
        <div class="hyp-text">${escapeHtml(h.hypothesis || "")}</div>
        <div class="hyp-scores">
          <span>Feasibility ${h.feasibility_score}</span>
          <span>Novelty ${h.novelty_score}</span>
          <span>Grounding ${h.evidence_grounding_score}</span>
          <span>Specificity ${h.specificity_score}</span>
        </div>
      </div>`;
    });
    html += `</div>`;
  }

  resultsState.innerHTML = html;
  showState("results");
  wirePaperSwapEvents();
}

// ===== Swap papers: render =====
// Two linked lists inside one section: "Papers compared" (the working set)
// and "Other papers found" (the candidate pool). Clicking "Swap out" on a
// working paper arms it; clicking "Use this instead" on a candidate
// completes the trade in the browser only — nothing hits the backend until
// "Apply swap & re-run" is clicked, which re-runs Comparison/Hypothesis
// Generation/Critique against the new paper set via confirmed_paper_ids.
function renderPaperSwapSection() {
  const isUploadsMode = lastUploadedPaths.length > 0;

  let html = `<div class="result-section"><h3>Papers compared (${currentWorkingSet.length})</h3>`;

  if (isUploadsMode) {
    html += `<p class="paper-meta" style="margin-bottom:10px;">Comparing exactly the papers you uploaded — swap isn't available in uploads mode.</p>`;
  } else if (swapSourceId) {
    html += `<p class="paper-meta" style="margin-bottom:10px;">Pick a paper below from "Other papers found" to swap in.</p>`;
  }

  currentWorkingSet.forEach((p) => {
    const armed = p.paper_id === swapSourceId;
    html += `<div class="paper-row${armed ? " swap-armed" : ""}">
      <div class="paper-title">${escapeHtml(p.title)}</div>
      <div class="paper-meta">
        <span class="badge">${p.is_india_affiliated ? "India" : "Global"}</span>
        ${p.published ? `<span>${escapeHtml(String(p.published))}</span>` : ""}
        ${!isUploadsMode && currentCandidates.length ? `
          <button class="swap-btn" data-swap-source="${escapeHtml(p.paper_id)}" type="button">
            ${armed ? "Cancel swap" : "Swap out"}
          </button>` : ""}
      </div>
    </div>`;
  });
  html += `</div>`;

  if (!isUploadsMode && currentCandidates.length) {
    html += `<div class="result-section"><h3>Other papers found (${currentCandidates.length})</h3>`;
    html += `<p class="paper-meta" style="margin-bottom:10px;">Not currently in the comparison.</p>`;
    currentCandidates.slice(0, 8).forEach((p) => {
      html += `<div class="paper-row">
        <div class="paper-title" style="font-weight:500;">${escapeHtml(p.title)}</div>
        <div class="paper-meta">
          <span class="badge">${p.is_india_affiliated ? "India" : "Global"}</span>
          ${p.published ? `<span>${escapeHtml(String(p.published))}</span>` : ""}
          <button class="swap-btn" data-swap-in="${escapeHtml(p.paper_id)}" type="button" ${swapSourceId ? "" : "disabled"}>
            Use this instead
          </button>
        </div>
      </div>`;
    });
    html += `</div>`;
  }

  if (!isUploadsMode) {
    html += `<div class="swap-bar hidden" id="swapBar">
      <span>Working set changed — re-run comparison, hypothesis generation, and critique against the new papers.</span>
      <button class="run-btn run-btn-sm" id="applySwapBtn" type="button">Apply swap &amp; re-run &#8599;</button>
    </div>`;
  }

  return html;
}

function wirePaperSwapEvents() {
  const section = document.getElementById("paperSwapSection");
  if (!section) return;

  section.addEventListener("click", (e) => {
    const outBtn = e.target.closest("[data-swap-source]");
    if (outBtn) {
      const id = outBtn.dataset.swapSource;
      swapSourceId = swapSourceId === id ? null : id;
      refreshPaperSwapSection();
      return;
    }
    const inBtn = e.target.closest("[data-swap-in]");
    if (inBtn && !inBtn.disabled) {
      completeSwap(inBtn.dataset.swapIn);
      return;
    }
    if (e.target.id === "applySwapBtn") {
      applySwapAndRerun();
    }
  });
}

function completeSwap(candidateId) {
  if (!swapSourceId) return;

  const outIdx = currentWorkingSet.findIndex((p) => p.paper_id === swapSourceId);
  const inIdx = currentCandidates.findIndex((p) => p.paper_id === candidateId);
  if (outIdx === -1 || inIdx === -1) return;

  const [outgoing] = currentWorkingSet.splice(outIdx, 1, currentCandidates[inIdx]);
  currentCandidates.splice(inIdx, 1, outgoing);

  swapSourceId = null;
  refreshPaperSwapSection(/* dirty */ true);
}

function refreshPaperSwapSection(markDirty) {
  const section = document.getElementById("paperSwapSection");
  if (!section) return;
  section.innerHTML = renderPaperSwapSection();

  const swapBar = document.getElementById("swapBar");
  if (swapBar && markDirty) {
    swapBar.classList.remove("hidden");
  }
}

async function applySwapAndRerun() {
  if (!lastRunPayload) return;

  const applyBtn = document.getElementById("applySwapBtn");
  if (applyBtn) {
    applyBtn.disabled = true;
    applyBtn.textContent = "Re-running…";
  }

  showState("loading");
  loadingStage.textContent = "RE-RUNNING WITH SWAPPED PAPERS";
  loadingSub.textContent = "Comparing the updated paper set and regenerating hypotheses.";
  setAgentStatus("Agent running…", true);

  try {
    const payload = {
      ...lastRunPayload,
      confirmed_paper_ids: currentWorkingSet.map((p) => p.paper_id),
      uploaded_file_paths: [] // swap is topic-driven only; no re-upload needed
    };

    const startRes = await fetch(`${API_BASE}/research`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!startRes.ok) throw new Error(`Server returned ${startRes.status}`);
    const { job_id } = await startRes.json();

    pollJob(job_id);
  } catch (err) {
    showError("Could not re-run with the swapped papers", err.message);
  }
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str == null ? "" : String(str);
  return div.innerHTML;
}

// ===== Splash screen =====
// Shown once on first load; fades + scales out on its own after a beat,
// then is removed from the DOM entirely so it can't block clicks.
function initSplash() {
  const splash = document.getElementById("splashScreen");
  if (!splash) {
    showTrendingPopup();
    return;
  }
  setTimeout(() => {
    splash.classList.add("splash-hide");
    setTimeout(() => {
      splash.remove();
      showTrendingPopup();
    }, 800);
  }, 1900);
}

// ===== Trending-topics popup =====
// Fetched in the background while the splash screen plays. The topics come
// from papers already in the knowledge base (GET /trending), each shown with
// a real paper title and its publication date. Picking one fills the search
// box; "Maybe next time" just closes it. If the request fails or returns
// nothing, the popup is skipped silently.
let trendingPromise = null;

function fetchTrending() {
  if (!trendingPromise) {
    trendingPromise = fetch(`${API_BASE}/trending`)
      .then((r) => (r.ok ? r.json() : { topics: [] }))
      .catch(() => ({ topics: [] }));
  }
  return trendingPromise;
}

// Capitalise the first letter for display, but leave names like "mRNA" or
// "CRISPR" alone (only an all-lowercase first word gets capitalised).
function capitalizeFirst(s) {
  s = String(s || "");
  const firstWord = s.split(" ")[0];
  if (firstWord !== firstWord.toLowerCase()) return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

async function showTrendingPopup() {
  const data = await fetchTrending();
  const topics = (data && data.topics) || [];
  if (!topics.length) return;

  const overlay = document.createElement("div");
  overlay.className = "trend-overlay";
  overlay.innerHTML = `
    <div class="trend-modal" role="dialog" aria-modal="true" aria-labelledby="trendTitle">
      <button class="trend-close" type="button" aria-label="Close">&times;</button>
      <div class="eyebrow">TRENDING IN ${escapeHtml(data.year || "2026")}</div>
      <h2 class="trend-title" id="trendTitle">Start with what&rsquo;s new</h2>
      <p class="trend-sub">Top topics from the latest papers in your knowledge base. Pick one to fill the search box, then choose your years.</p>
      <div class="trend-list">
        ${topics.map((t, i) => `
          <button class="trend-item" type="button" data-topic="${escapeHtml(t.topic)}">
            <span class="trend-num">0${i + 1}</span>
            <span class="trend-body">
              <span class="trend-topic">${escapeHtml(capitalizeFirst(t.topic))}</span>
              <span class="trend-paper">${escapeHtml(t.paper_title)}</span>
              <span class="trend-meta">Published ${escapeHtml(t.published)} &middot; ${t.paper_count} paper${t.paper_count === 1 ? "" : "s"} this year</span>
            </span>
          </button>`).join("")}
      </div>
      <button class="trend-later" type="button">Maybe next time</button>
    </div>`;
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add("show"));

  function close() {
    document.removeEventListener("keydown", onKey);
    overlay.classList.remove("show");
    setTimeout(() => overlay.remove(), 250);
  }
  function onKey(e) {
    if (e.key === "Escape") close();
  }
  document.addEventListener("keydown", onKey);

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay || e.target.closest(".trend-close") || e.target.closest(".trend-later")) {
      close();
      return;
    }
    const item = e.target.closest(".trend-item");
    if (item) {
      topicInput.value = item.dataset.topic;
      close();
      topicInput.focus();
    }
  });
}

// ===== Init =====
normalizeSliders(-1);
renderFileList();
updateGlobalToggleLabel();
buildDna("dnaIdle");
buildDna("dnaLoading");
showState("idle");
fetchTrending(); // start loading trending topics while the splash plays
initSplash();