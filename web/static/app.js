// CarLaneI showcase — wires the UI to the FastAPI backend. All displayed values
// come from the real pipeline run; nothing is hardcoded except the held-out
// model metrics (which are measured, documented in RESULTS.md).

const API = ""; // same origin (served by FastAPI)
const $ = (id) => document.getElementById(id);

let selected = null;         // {type:'sample'|'file', value}
let telemetry = null;        // active {video, summary, frames}
let pollTimer = null;
let hudTimer = null;
let curMode = "egoseg";      // active lane method being shown
const resultsByMode = {};    // cache {egoseg:telemetry, corridor:telemetry}
let lastAlertState = { ldw: "", light: "" };  // for toast de-dup
let viewMode = "final";      // "final" (play rendered MP4) | "live" (MJPEG stream)
let currentJob = null;       // active job id (for stop/cancel)

// ---- health + samples -------------------------------------------------------
async function boot() {
  try {
    const h = await (await fetch(`${API}/api/health`)).json();
    $("status-dot").className = "w-2 h-2 rounded-full " + (h.ok ? "bg-acc-green" : "bg-acc-red");
    $("status-text").textContent = h.ok ? "BENCH ONLINE" : "BENCH DEGRADED";
    $("gpu-text").textContent = "GPU: " + (h.gpu ? "CUDA ACTIVE" : "CPU FALLBACK");
    const ready = Object.values(h.models).filter(Boolean).length;
    $("models-text").textContent = `MODELS: ${ready}/${Object.keys(h.models).length} LOADED`;
  } catch (e) {
    $("status-text").textContent = "BACKEND UNREACHABLE";
    $("status-dot").className = "w-2 h-2 rounded-full bg-acc-red";
  }
  try {
    const samples = await (await fetch(`${API}/api/samples`)).json();
    const box = $("samples");
    box.innerHTML = "";
    const fullBox = $("samples-full");
    if (fullBox) fullBox.innerHTML = "";
    samples.forEach((s) => {
      const b = document.createElement("button");
      b.className = "flex flex-col text-left p-2 border border-outline-variant bg-surface hover:border-primary disabled:opacity-40";
      b.disabled = !s.available;
      b.innerHTML = `<span class="font-semibold text-primary text-[12px] truncate">${s.name}</span>
        <span class="text-[10px] text-outline leading-tight">${s.available ? (s.hint || "ready") : "missing"}</span>`;
      b.onclick = () => selectSample(s.id, b, s.full);
      // full-length clips go into their own "Live (long)" group
      (s.full && fullBox ? fullBox : box).appendChild(b);
    });
  } catch (e) {}
}

function clearSelection() {
  document.querySelectorAll("#samples button, #samples-full button").forEach((x) =>
    x.className = x.className.replace(" border-primary border-2", "").replace("border-primary", "border-outline-variant"));
}

function selectSample(id, btn, isFull) {
  selected = { type: "sample", value: id, full: !!isFull };
  clearSelection();
  if (btn) btn.className = "flex flex-col text-left p-2 border-2 border-primary bg-surface-container-low";
  $("file-input").value = "";
  if (isFull) {
    setViewMode("live");          // full clips are live-only (no saved file)
    $("run-label").textContent = `Live (long) "${id.replace(/_full$/, "").replace(/_/g, " ")}"`;
  } else {
    $("run-label").textContent = `Run "${id.replace(/_/g, " ")}"`;
  }
  $("run-btn").disabled = false;
}

// Accept a video by MIME type OR extension — drag-drop often reports an empty
// f.type, and some containers (mkv/avi) have non-standard MIME strings.
const VIDEO_EXT = /\.(mp4|mov|m4v|mkv|avi|webm|mpeg|mpg|3gp|ogv|ts)$/i;
function isVideoFile(f) {
  return (f.type && f.type.startsWith("video")) || VIDEO_EXT.test(f.name || "");
}
function acceptFile(f) {
  selected = { type: "file", value: f };
  clearSelection();
  $("file-input").value === undefined; // no-op guard
  $("run-label").textContent = `Run "${f.name}"`;
  $("run-btn").disabled = false;
  // clear any prior error state
  if ($("run-badge")) $("run-badge").innerHTML =
    `<span class="px-2.5 py-1 bg-surface-container border border-outline-variant text-on-surface font-semibold">READY</span>`;
}

$("file-input").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  if (!isVideoFile(f)) {
    fail(`"${f.name}" is not a recognized video. Use MP4, MOV, MKV, AVI, or WEBM.`);
    return;
  }
  acceptFile(f);
});

$("sec-range").addEventListener("input", (e) => {
  $("sec-val").textContent = e.target.value + "s";
  $("cap-seconds").textContent = e.target.value;
});

// Clip-length limit toggle. OFF (default) = process the FULL uploaded video;
// ON = cap to the slider value. Enables/disables the slider row accordingly.
function limitEnabled() { return $("limit-toggle").checked; }
$("limit-toggle").addEventListener("change", () => {
  const row = $("clip-len-row");
  if (limitEnabled()) {
    row.classList.remove("opacity-40", "pointer-events-none");
    $("cap-seconds").textContent = $("sec-range").value;
  } else {
    row.classList.add("opacity-40", "pointer-events-none");
    $("cap-seconds").textContent = "full";
  }
});

// ---- view mode toggle (final render vs live preview) ------------------------
function setViewMode(m) {
  viewMode = m;
  const on = "px-3 py-1 font-semibold bg-primary text-white";
  const off = "px-3 py-1 font-semibold bg-surface text-on-surface-variant border border-outline-variant";
  $("view-final").className = m === "final" ? on : off;
  $("view-live").className = m === "live" ? on : off;
}
$("view-final").addEventListener("click", () => setViewMode("final"));
$("view-live").addEventListener("click", () => setViewMode("live"));

// Stop ANY running job (render, live preview, or long-live).
$("stop-btn").addEventListener("click", async () => {
  if (!currentJob) return;
  $("stop-btn").disabled = true;
  setProgress(99, "Stopping…");
  try { await fetch(`${API}/api/stop/${currentJob}`, { method: "POST" }); } catch (e) {}
});
// End any running job if the user leaves the page (frees the GPU).
window.addEventListener("beforeunload", () => {
  if (currentJob) navigator.sendBeacon(`${API}/api/stop/${currentJob}`);
});

// ---- run + poll -------------------------------------------------------------
$("run-btn").addEventListener("click", () => startRun("egoseg"));

async function startRun(mode) {
  if (!selected) return;
  // a fresh run invalidates any cached before/after results
  delete resultsByMode.egoseg; delete resultsByMode.corridor;
  curMode = mode || "egoseg";
  $("run-btn").disabled = true;
  $("progress-wrap").classList.remove("hidden");
  setProgress(0, "Queued…");
  $("run-badge").innerHTML = `<span class="px-2.5 py-1 bg-acc-amber text-white font-semibold">PROCESSING</span>`;

  const longLive = !!(selected && selected.full);

  const fd = new FormData();
  // seconds=0 => backend processes the FULL video (no trim). Only cap when the
  // "Limit clip length" toggle is on.
  fd.append("seconds", limitEnabled() ? $("sec-range").value : "0");
  fd.append("mode", curMode);
  fd.append("lane_model", $("lane-model").value);
  if (longLive) fd.append("live_long", "1");
  if (selected.type === "sample") fd.append("sample", selected.value);
  else fd.append("file", selected.value);

  let job;
  try {
    job = (await (await fetch(`${API}/api/process`, { method: "POST", body: fd })).json()).job_id;
  } catch (e) { return fail("Could not reach backend"); }
  currentJob = job;

  // Every run is stoppable now (render, live preview, or long-live).
  $("stop-btn").classList.remove("hidden");
  $("stop-btn").disabled = false;

  // Live preview: stream annotated frames into the <img> while rendering.
  if (viewMode === "live" || longLive) startLivePreview(job);
  if (longLive) {
    setProgress(0, "Live (long) — streaming full clip…");
  }

  pollTimer = setInterval(async () => {
    let s;
    try { s = await (await fetch(`${API}/api/job/${job}`)).json(); }
    catch (e) { return; }
    if (s.status === "processing" || s.status === "queued") {
      setProgress(longLive ? 50 : (s.progress || 0),
                  longLive ? "Live (long) — streaming…" : "Running perception stack…");
    } else if (s.status === "stopped") {
      // user pressed Stop mid-run — no result to show
      clearInterval(pollTimer);
      stopLivePreview();
      setProgress(100, "Stopped");
      $("run-badge").innerHTML = `<span class="px-2.5 py-1 bg-outline text-white font-semibold">STOPPED</span>`;
      resetRunControls();
      return;
    } else if (s.status === "done") {
      clearInterval(pollTimer);
      if (longLive) {
        // no saved file/telemetry — the stream just ended
        stopLivePreview();
        setProgress(100, "Live run finished");
        $("run-badge").innerHTML = `<span class="px-2.5 py-1 bg-acc-green text-white font-semibold">DONE</span>`;
        resetRunControls();
        return;
      }
      setProgress(100, "Done");
      await loadResult(job);
    } else if (s.status === "error") {
      clearInterval(pollTimer);
      fail(s.error || "Processing failed");
    }
  }, 1200);
}

function startLivePreview(job) {
  const img = $("live-view");
  img.src = `${API}/api/stream/${job}?t=${Date.now()}`;
  img.classList.remove("hidden");
  $("live-badge").classList.remove("hidden");
  $("video-placeholder").classList.add("hidden");
  $("result-video").classList.add("hidden");
}
function stopLivePreview() {
  const img = $("live-view");
  img.src = "";                       // close the MJPEG connection
  img.classList.add("hidden");
  $("live-badge").classList.add("hidden");
  $("result-video").classList.remove("hidden");
}

function setProgress(p, label) {
  $("progress-bar").style.width = p + "%";
  $("progress-pct").textContent = p + "%";
  if (label) $("progress-label").textContent = label;
}
// Re-arm the run controls after any run ends (done / stopped / error).
function resetRunControls() {
  $("run-btn").disabled = false;
  $("stop-btn").classList.add("hidden");
  $("stop-btn").disabled = false;
  $("progress-wrap").classList.add("hidden");
  currentJob = null;
}

function fail(msg) {
  if (pollTimer) clearInterval(pollTimer);
  stopLivePreview();
  $("run-badge").innerHTML = `<span class="px-2.5 py-1 bg-acc-red text-white font-semibold">ERROR</span>`;
  $("progress-label").textContent = msg;
  resetRunControls();
}

// ---- results ----------------------------------------------------------------
async function loadResult(job) {
  const data = await (await fetch(`${API}/api/result/${job}`)).json();
  const method = (data.summary && data.summary.lane_method) || curMode;
  resultsByMode[method] = data;
  curMode = method;
  showResult(data);
  $("run-badge").innerHTML = `<span class="px-2.5 py-1 bg-acc-green text-white font-semibold">COMPLETE</span>`;
  // reveal before/after controls now that we have at least one result
  $("ba-wrap").classList.remove("hidden");
  updateModeButtons();
  // re-enable so the user can run again + hide the stop button
  resetRunControls();
}

// Render a given telemetry result into the viewport + panels.
function showResult(data) {
  telemetry = data;
  stopLivePreview();                 // swap the live stream for the rendered MP4

  // --- 1) Get the VIDEO playable FIRST, before any heavy telemetry work.
  // A full-length clip returns thousands of frames; building the timeline/charts
  // over all of them is heavy, so it must never block the video from appearing.
  const v = $("result-video");
  v.src = telemetry.video;
  $("video-placeholder").classList.add("hidden");
  $("hud").classList.remove("hidden");
  lastAlertState = { ldw: "", light: "" };
  v.load();
  v.muted = true;
  v.play().catch(() => { /* autoplay blocked — user can press play */ });

  const dl = $("dl-link");
  dl.href = telemetry.video;
  dl.setAttribute("download", "carlane_annotated.mp4");
  dl.classList.remove("hidden"); dl.classList.add("flex");

  // --- 2) Build the panels on the next tick, each guarded, so a slow/failed
  // panel on a huge clip can't stop the video (or the other panels).
  setTimeout(() => {
    try { fillSummary(telemetry.summary); } catch (e) { console.warn("summary", e); }
    try { fillTables(telemetry.summary); } catch (e) { console.warn("tables", e); }
    try { buildTimeline(telemetry); } catch (e) { console.warn("timeline", e); }
    try { buildCharts(telemetry); } catch (e) { console.warn("charts", e); }
    try { bindHud(v); } catch (e) { console.warn("hud", e); }
  }, 0);
  return;
}

function fillSummary(s) {
  $("stat-fps").textContent = s.fps_source;
  $("stat-proc").textContent = s.avg_processing_fps;
  $("stat-res").textContent = s.resolution.join("×");
  $("m-lane").textContent = s.lane_present_pct + "%";
  $("m-offset").textContent = s.mean_abs_lane_offset == null ? "n/a" : s.mean_abs_lane_offset;
  $("m-veh").textContent = s.vehicles_per_frame;
  $("m-fps").textContent = s.avg_processing_fps;
}

// mean confidence per class/state, computed from the real per-frame detections
function _confByKey(frames, listKey, keyName) {
  const acc = {};
  frames.forEach((f) => f[listKey].forEach((d) => {
    const k = d[keyName];
    (acc[k] = acc[k] || []).push(d.conf);
  }));
  const out = {};
  for (const k in acc) out[k] = acc[k].reduce((a, b) => a + b, 0) / acc[k].length;
  return out;
}
function _confBadge(c) {
  const col = c >= 0.7 ? "#2B593F" : c >= 0.5 ? "#B45309" : "#C2410C";
  return `<span class="inline-block px-1.5 py-0.5 text-[10px] font-bold text-white" style="background:${col}">${c.toFixed(2)}</span>`;
}

function fillTables(s) {
  const frames = (telemetry && telemetry.frames) || [];
  const signColor = { prohibitory: "text-acc-red", mandatory: "text-acc-blue",
    danger: "text-acc-amber", other: "text-on-surface" };
  const signConf = _confByKey(frames, "signs", "cls");
  const signTotal = Object.values(s.sign_class_counts).reduce((a, b) => a + b, 0);
  const st = $("sign-tbody");
  if (signTotal === 0) {
    st.innerHTML = `<tr><td class="p-3 text-outline" colspan="4">No signs in this clip (expected for US footage — detector is GTSDB/European).</td></tr>`;
  } else {
    st.innerHTML = Object.entries(s.sign_class_counts).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
      `<tr class="hover:bg-surface-container-low"><td class="p-3 font-semibold ${signColor[k] || ""}">${k}</td>
       <td class="p-3">${v}</td><td class="p-3">${_confBadge(signConf[k] || 0)}</td>
       <td class="p-3 text-right">${(v / signTotal * 100).toFixed(0)}%</td></tr>`).join("");
  }

  const lc = { RED: "text-acc-red", YELLOW: "text-acc-amber", GREEN: "text-acc-green" };
  const dot = { RED: "#C2410C", YELLOW: "#B45309", GREEN: "#2B593F" };
  const lightConf = _confByKey(frames, "lights", "state");
  const lightTotal = Object.values(s.light_state_counts).reduce((a, b) => a + b, 0);
  const lt = $("light-tbody");
  if (lightTotal === 0) {
    lt.innerHTML = `<tr><td class="p-3 text-outline" colspan="4">No traffic lights detected in this clip.</td></tr>`;
  } else {
    lt.innerHTML = ["RED", "YELLOW", "GREEN"].filter((k) => s.light_state_counts[k]).map((k) => {
      const v = s.light_state_counts[k];
      return `<tr class="hover:bg-surface-container-low">
        <td class="p-3"><span class="inline-flex items-center gap-1.5 font-bold ${lc[k]}">
        <span class="w-2 h-2 rounded-full" style="background:${dot[k]}"></span>${k}</span></td>
        <td class="p-3">${v}</td><td class="p-3">${_confBadge(lightConf[k] || 0)}</td>
        <td class="p-3 text-right">${(v / lightTotal * 100).toFixed(0)}%</td></tr>`;
    }).join("");
  }
}

// ---- HUD synced to video playback ------------------------------------------
function bindHud(video) {
  if (hudTimer) cancelAnimationFrame(hudTimer);
  const frames = telemetry.frames;
  const fps = telemetry.summary.fps_source || 30;

  function tick() {
    if (!video.paused && !video.ended) {
      const idx = Math.min(frames.length - 1, Math.round(video.currentTime * fps));
      const f = frames[idx];
      if (f) {
        $("hud-fps").textContent = f.fps;
        $("hud-t").textContent = f.t.toFixed(1);
        $("hud-cars").textContent = f.vehicles.length;
        $("hud-lights").textContent = f.lights.length;
        $("hud-signs").textContent = f.signs.length;
        const lane = f.lane_offset == null ? "no fix"
          : (f.ldw ? f.ldw : `offset ${f.lane_offset >= 0 ? "+" : ""}${f.lane_offset}`);
        const laneEl = $("hud-lane");
        laneEl.textContent = lane;
        laneEl.className = "font-bold " + (f.ldw === "LANE DEPARTURE" ? "text-acc-red"
          : f.ldw === "drifting" ? "text-acc-amber" : "text-acc-green");
        // dominant signal chip
        const sig = $("hud-signal");
        if (f.dominant_light) {
          sig.classList.remove("hidden");
          const col = { RED: "#C2410C", YELLOW: "#B45309", GREEN: "#2B593F" }[f.dominant_light] || "#000";
          sig.textContent = f.dominant_light;
          sig.style.color = "#fff"; sig.style.background = col; sig.style.borderColor = col;
        } else { sig.classList.add("hidden"); }

        // move timeline cursor
        const frac = video.currentTime / (video.duration || 1);
        const cur = $("tl-cursor");
        if (cur) cur.style.left = (frac * 100) + "%";
        // move chart cursors + update readouts
        const co = document.getElementById("cur-offset");
        const cc = document.getElementById("cur-conf");
        if (co) { co.setAttribute("x1", frac * 300); co.setAttribute("x2", frac * 300); }
        if (cc) { cc.setAttribute("x1", frac * 300); cc.setAttribute("x2", frac * 300); }
        const ov = $("chart-offset-val");
        if (ov) ov.textContent = f.lane_offset == null ? "no fix"
          : `${f.lane_offset >= 0 ? "+" : ""}${f.lane_offset.toFixed(2)}`;
        const cv = $("chart-conf-val");
        if (cv) cv.textContent = _meanConf(f).toFixed(2);

        // fire alerts on state CHANGE only (de-duplicated)
        if (f.ldw && f.ldw !== lastAlertState.ldw) {
          toast(f.ldw === "LANE DEPARTURE" ? "red" : "amber",
                f.ldw === "LANE DEPARTURE" ? "warning" : "info",
                f.ldw === "LANE DEPARTURE" ? "LANE DEPARTURE" : "Drifting from lane centre",
                `t=${f.t.toFixed(1)}s`);
        }
        lastAlertState.ldw = f.ldw || "";
        if (f.dominant_light === "RED" && lastAlertState.light !== "RED") {
          toast("red", "traffic", "RED signal ahead", `t=${f.t.toFixed(1)}s`);
        }
        lastAlertState.light = f.dominant_light || "";
      }
    }
    hudTimer = requestAnimationFrame(tick);
  }
  hudTimer = requestAnimationFrame(tick);
}

// ---- timeline strip ---------------------------------------------------------
function buildTimeline(data) {
  const wrap = $("timeline-wrap");
  const tl = $("timeline");
  const ldwRow = $("timeline-ldw");
  const frames = data.frames;
  if (!frames || !frames.length) { wrap.classList.add("hidden"); return; }
  wrap.classList.remove("hidden");
  tl.innerHTML = ""; ldwRow.innerHTML = "";
  const n = frames.length;
  const col = { RED: "#C2410C", YELLOW: "#B45309", GREEN: "#2B593F" };

  // build contiguous runs of the same dominant state -> one segment each
  let i = 0;
  while (i < n) {
    const state = frames[i].dominant_light || null;
    let j = i;
    while (j + 1 < n && (frames[j + 1].dominant_light || null) === state) j++;
    const seg = document.createElement("div");
    seg.className = "tl-seg";
    seg.style.left = (i / n * 100) + "%";
    seg.style.width = ((j - i + 1) / n * 100) + "%";
    seg.style.background = state ? col[state] : "#c5c6ca";
    tl.appendChild(seg);
    i = j + 1;
  }
  // LDW markers
  frames.forEach((f, k) => {
    if (f.ldw) {
      const m = document.createElement("div");
      m.className = "tl-ldw";
      m.style.left = (k / n * 100) + "%";
      m.style.background = f.ldw === "LANE DEPARTURE" ? "#C2410C" : "#B45309";
      ldwRow.appendChild(m);
    }
  });
  // playback cursor
  const cur = document.createElement("div");
  cur.id = "tl-cursor"; cur.className = "tl-cursor"; cur.style.left = "0%";
  tl.appendChild(cur);

  // click to seek
  tl.onclick = (e) => {
    const v = $("result-video");
    const rect = tl.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    if (v.duration) v.currentTime = frac * v.duration;
  };
}

// ---- charts (SVG sparklines from telemetry) --------------------------------
function _poly(vals, w, h, y0, y1) {
  // map an array of values in [y0,y1] to an SVG polyline points string.
  // Downsample to ~600 points max — the chart is only ~300px wide, so a
  // full-length clip's thousands of frames would just be wasted DOM/CPU.
  const MAX_PTS = 600;
  let arr = vals;
  if (vals.length > MAX_PTS) {
    const step = vals.length / MAX_PTS;
    arr = [];
    for (let k = 0; k < MAX_PTS; k++) arr.push(vals[Math.floor(k * step)]);
  }
  const n = arr.length;
  return arr.map((v, i) => {
    const x = (i / Math.max(n - 1, 1)) * w;
    const t = (v - y0) / (y1 - y0 || 1);
    const y = h - Math.max(0, Math.min(1, t)) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function _meanConf(f) {
  const all = [...f.vehicles, ...f.signs, ...f.lights].map((d) => d.conf).filter((c) => c != null);
  return all.length ? all.reduce((a, b) => a + b, 0) / all.length : 0;
}

function buildCharts(data) {
  const wrap = $("charts-wrap");
  const fr = data.frames;
  if (!fr || !fr.length) { wrap.classList.add("hidden"); return; }
  wrap.classList.remove("hidden");
  const W = 300, H = 60;

  // lane offset: range [-1, 1], 0 = centred; dashed lines at ±0.8 (departure)
  const offs = fr.map((f) => (f.lane_offset == null ? 0 : f.lane_offset));
  const offSvg = $("chart-offset");
  const zeroY = H / 2;
  const thrTop = H - ((0.8 - (-1)) / 2) * H, thrBot = H - ((-0.8 - (-1)) / 2) * H;
  offSvg.innerHTML =
    `<line x1="0" y1="${zeroY}" x2="${W}" y2="${zeroY}" stroke="#c5c6ca" stroke-width="1"/>` +
    `<line x1="0" y1="${thrTop}" x2="${W}" y2="${thrTop}" stroke="#C2410C" stroke-width="0.7" stroke-dasharray="3 3"/>` +
    `<line x1="0" y1="${thrBot}" x2="${W}" y2="${thrBot}" stroke="#C2410C" stroke-width="0.7" stroke-dasharray="3 3"/>` +
    `<polyline fill="none" stroke="#1E3A5F" stroke-width="1.5" points="${_poly(offs, W, H, -1, 1)}"/>` +
    `<line id="cur-offset" x1="0" y1="0" x2="0" y2="${H}" stroke="#000" stroke-width="1"/>`;

  // mean confidence: range [0,1]
  const confs = fr.map(_meanConf);
  const confSvg = $("chart-conf");
  confSvg.innerHTML =
    `<line x1="0" y1="${H * 0.5}" x2="${W}" y2="${H * 0.5}" stroke="#c5c6ca" stroke-width="0.7" stroke-dasharray="2 3"/>` +
    `<polyline fill="none" stroke="#2B593F" stroke-width="1.5" points="${_poly(confs, W, H, 0, 1)}"/>` +
    `<line id="cur-conf" x1="0" y1="0" x2="0" y2="${H}" stroke="#000" stroke-width="1"/>`;
}

// ---- before/after toggle ----------------------------------------------------
async function switchMode(mode) {
  if (mode === curMode && resultsByMode[mode]) return;
  if (resultsByMode[mode]) {                    // cached -> instant swap
    curMode = mode;
    showResult(resultsByMode[mode]);
    updateModeButtons();
    return;
  }
  // not cached -> process this clip in the requested mode
  $("ba-status").textContent = `processing ${mode}…`;
  updateModeButtons(mode);
  await startRun(mode);
  $("ba-status").textContent = "";
}
$("ba-egoseg").addEventListener("click", () => switchMode("egoseg"));
$("ba-corridor").addEventListener("click", () => switchMode("corridor"));

function updateModeButtons(pending) {
  const active = pending || curMode;
  const on = "i-btn px-3 py-1 mono text-[11px] font-semibold bg-primary text-white";
  const off = "i-btn px-3 py-1 mono text-[11px] font-semibold bg-surface text-on-surface-variant";
  $("ba-egoseg").className = active === "egoseg" ? on : off;
  $("ba-corridor").className = active === "corridor" ? on : off;
  $("ba-note").textContent = active === "corridor"
    ? "Heuristic corridor (old) — IoU 0.06 on held-out BDD"
    : "Learned ego-seg — mask mAP50 0.959, IoU 0.59 vs 0.06";
}

// ---- toasts -----------------------------------------------------------------
const _iconFor = { warning: "warning", info: "info", traffic: "traffic" };
function toast(kind, icon, title, sub) {
  const box = $("toasts");
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.innerHTML = `<span class="material-symbols-outlined" style="color:${kind === "red" ? "#C2410C" : "#B45309"}">${_iconFor[icon] || "info"}</span>
    <span><b>${title}</b>${sub ? `<br><span style="color:#75777b">${sub}</span>` : ""}</span>`;
  box.appendChild(t);
  requestAnimationFrame(() => t.classList.add("show"));
  setTimeout(() => { t.classList.remove("show"); setTimeout(() => t.remove(), 350); }, 3200);
}

// ---- interactivity polish: apply hover/reveal effects without editing markup -
function enhance() {
  // Cards: every bordered surface box becomes a lift-on-hover card.
  document.querySelectorAll(
    "section .border.border-outline-variant, section .bg-surface-container-lowest.border"
  ).forEach((el) => {
    // skip the big viewport + table wrappers (handled separately)
    if (el.querySelector("video") || el.querySelector("table")) return;
    el.classList.add("i-card", "i-accent");
    el.classList.add("cursor-default");
  });

  // Metric big-numbers get the pulse class.
  document.querySelectorAll(".mono.text-\\[26px\\]").forEach((n) => n.classList.add("i-num"));

  // Icon tiles inside cards animate.
  document.querySelectorAll(".material-symbols-outlined").forEach((i) => {
    const tile = i.closest("div");
    if (tile && tile.className.includes("flex items-center justify-center")) tile.classList.add("i-icon");
  });

  // Colour the accent bar to match each card's dominant accent.
  document.querySelectorAll(".i-accent").forEach((el) => {
    if (el.querySelector(".text-acc-blue")) el.classList.add("blue");
    else if (el.querySelector(".text-acc-amber")) el.classList.add("amber");
    else if (el.querySelector(".text-acc-red")) el.classList.add("red");
  });

  // Buttons + sample tiles + nav.
  document.querySelectorAll("button, label.cursor-pointer").forEach((b) => b.classList.add("i-btn"));
  document.querySelectorAll("nav a").forEach((a) => a.classList.add("i-nav"));

  // Video viewport glow.
  const view = document.querySelector(".aspect-video");
  if (view) view.classList.add("i-view");

  // Dropzone drag styling.
  const drop = document.querySelector(".border-dashed");
  if (drop) {
    drop.classList.add("i-drop");
    ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => {
      ev.preventDefault(); drop.classList.add("drag");
    }));
    ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, () => drop.classList.remove("drag")));
    drop.addEventListener("drop", (ev) => {
      ev.preventDefault();
      const f = ev.dataTransfer.files[0];
      if (f && isVideoFile(f)) {
        acceptFile(f);
      } else if (f) {
        fail(`"${f.name}" is not a recognized video. Use MP4, MOV, MKV, AVI, or WEBM.`);
      }
    });
  }

  // Live/status dots pulse.
  document.querySelectorAll("#status-dot, .bg-acc-green.rounded-full").forEach((d) => d.classList.add("live-dot"));

  // Scroll-reveal for cards only (never whole sections, to avoid blanking).
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.05 });
  document.querySelectorAll(".i-card").forEach((el) => {
    el.classList.add("reveal"); io.observe(el);
  });
  // safety net: force-reveal after a moment regardless of observer
  setTimeout(() => document.querySelectorAll(".reveal").forEach((e) => e.classList.add("in")), 600);
}

boot();
enhance();
setViewMode("final");
