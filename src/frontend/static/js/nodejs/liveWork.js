// What the node is doing right now: GPU temperature, a job awaiting the owner's
// approval, and the running job's log as it is written.
//
// The running job appears as a dismissible overlay. Closing it does not stop
// anything -- the job keeps running and its row in Jobs reopens the same view.
//
// Built with createElement + textContent throughout: log lines come from the
// node and job names come from whoever submitted the work.

const POLL_MS = 2000;

let timer = null;
let lastLogCount = 0;
let overlayDismissedFor = null;   // task_id the owner closed; do not reopen it
let latest = { task: null, thermal: null };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

// --- thermal strip -------------------------------------------------------

function renderThermal(thermal) {
  const box = document.getElementById("thermalStatus");
  if (!box) return;

  const gpus = thermal?.gpus || [];
  if (!gpus.length) {
    box.className = "thermal";
    box.replaceChildren(el("span", "thermal-none", "No GPU reporting temperature."));
    return;
  }

  box.className = `thermal is-${thermal.state || "ok"}`;
  box.replaceChildren();

  gpus.forEach((gpu) => {
    const row = el("div", `thermal-gpu is-${gpu.state}`);
    row.appendChild(el("span", "thermal-name", gpu.name));

    // Scaled to this card's own stop point, not an arbitrary 100 C.
    const span = Math.max(1, gpu.stop - 30);
    const pct = (value) => Math.max(0, Math.min(100, ((value - 30) / span) * 100));

    const track = el("div", "thermal-track");
    const fill = el("div", "thermal-fill");
    fill.style.width = `${pct(gpu.temperature)}%`;
    track.appendChild(fill);

    const mark = el("div", "thermal-mark");
    mark.style.left = `${pct(gpu.warn)}%`;
    mark.title = `Warning from ${gpu.warn}°C`;
    track.appendChild(mark);

    row.appendChild(track);
    row.appendChild(el("span", "thermal-temp", `${gpu.temperature}°C`));
    row.appendChild(el("span", "thermal-limit", `stop ${gpu.stop}°C`));
    box.appendChild(row);
  });

  if (thermal.reason) box.appendChild(el("p", "thermal-reason", thermal.reason));
}

// --- approval prompt -----------------------------------------------------

async function respond(path, taskId) {
  try {
    const res = await fetch(`${path}/${encodeURIComponent(taskId)}`, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail?.detail || body.detail || `Server returned ${res.status}`);
    }
    overlayDismissedFor = null;      // a fresh job may open its own overlay
    pollLiveWork();
  } catch (err) {
    console.error("Approval action failed:", err);
    const box = document.getElementById("approvalPrompt");
    if (box) box.appendChild(el("p", "error-message", err.message));
  }
}

function renderApproval(waiting) {
  const box = document.getElementById("approvalPrompt");
  if (!box) return;

  if (!waiting) {
    box.replaceChildren();
    box.hidden = true;
    return;
  }

  box.hidden = false;
  box.replaceChildren();

  const head = el("div", "approval-head");
  head.appendChild(el("strong", null, "A job is waiting for you"));
  head.appendChild(el("span", "approval-timer",
    `${waiting.seconds_left}s left`));
  box.appendChild(head);

  const detail = el("p", "approval-detail");
  detail.textContent = `${waiting.model_name || waiting.task_type || "job"}`
    + (waiting.has_dataset ? " · dataset attached" : " · no dataset");
  box.appendChild(detail);

  // Says plainly what happens if nobody clicks.
  box.appendChild(el("p", "approval-note",
    "If you do nothing it goes back to the queue for another node."));

  const actions = el("div", "approval-actions");
  const accept = el("button", "btn-primary", "Accept and run");
  accept.addEventListener("click", () => respond("/approve-task", waiting.task_id));
  const decline = el("button", "btn-ghost", "Decline");
  decline.addEventListener("click", () => respond("/decline-task", waiting.task_id));
  actions.append(accept, decline);
  box.appendChild(actions);
}

// --- running job overlay -------------------------------------------------

function stat(label, value, className) {
  const box = el("div", `run-stat ${className || ""}`.trim());
  box.appendChild(el("span", "run-stat-label", label));
  box.appendChild(el("span", "run-stat-value", value));
  return box;
}

function formatElapsed(seconds) {
  if (seconds == null) return "—";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

// The log alone does not say how hot the card is or how far along the job is.
// Progress, heat and live GPU stats come first; the log is underneath.
function buildOverlayBody(task, thermal) {
  const body = el("div", "run-body");

  const head = el("div", "run-head");
  head.appendChild(el("span", `job-status job-status-${task.status || "running"}`,
    task.status === "running" ? "Running" : (task.status || "")));
  head.appendChild(el("span", "run-name", task.model_name || task.task_id || "job"));

  const close = el("button", "modal-close", "×");
  close.setAttribute("aria-label", "Close");
  close.addEventListener("click", () => {
    overlayDismissedFor = task.task_id;
    closeOverlay();
  });
  head.appendChild(close);
  body.appendChild(head);

  const progress = task.progress;
  if (progress && progress.steps) {
    const wrap = el("div", "run-progress");
    const line = el("div", "run-progress-head");
    line.appendChild(el("span", null, `Step ${progress.step} of ${progress.steps}`));
    line.appendChild(el("span", "run-progress-pct",
      `${Math.round((progress.step / progress.steps) * 100)}%`));
    wrap.appendChild(line);

    const track = el("div", "usage-bar");
    const fill = el("div", "usage-bar-fill");
    fill.style.width = `${Math.min(100, (progress.step / progress.steps) * 100)}%`;
    track.appendChild(fill);
    wrap.appendChild(track);
    body.appendChild(wrap);
  }

  const stats = el("div", "run-stats");
  stats.appendChild(stat("Elapsed", formatElapsed(task.elapsed_s)));
  if (progress && progress.loss != null) {
    stats.appendChild(stat("Loss", progress.loss));
    if (progress.initial_loss != null) {
      stats.appendChild(stat("Started at", progress.initial_loss));
    }
  }

  const gpus = thermal?.gpus || [];
  gpus.forEach((gpu) => {
    stats.appendChild(stat("GPU load",
      gpu.utilisation == null ? "—" : `${gpu.utilisation}%`));
    stats.appendChild(stat("Temperature", `${gpu.temperature}°C`, `is-${gpu.state}`));
    if (gpu.power_w != null) stats.appendChild(stat("Power", `${gpu.power_w} W`));
    if (gpu.fan_percent != null) stats.appendChild(stat("Fan", `${gpu.fan_percent}%`));
    if (gpu.memory_used_mb != null && gpu.memory_total_mb != null) {
      stats.appendChild(stat("VRAM",
        `${(gpu.memory_used_mb / 1024).toFixed(1)} / ${(gpu.memory_total_mb / 1024).toFixed(1)} GB`));
    }
  });
  body.appendChild(stats);

  // Heat headroom, scaled to this card's own stop point.
  gpus.forEach((gpu) => {
    const row = el("div", `run-thermal is-${gpu.state}`);
    row.appendChild(el("span", "run-thermal-name", gpu.name));

    const span = Math.max(1, gpu.stop - 30);
    const pct = (v) => Math.max(0, Math.min(100, ((v - 30) / span) * 100));

    const track = el("div", "thermal-track");
    const fill = el("div", "thermal-fill");
    fill.style.width = `${pct(gpu.temperature)}%`;
    track.appendChild(fill);
    const mark = el("div", "thermal-mark");
    mark.style.left = `${pct(gpu.warn)}%`;
    mark.title = `Warning from ${gpu.warn}°C`;
    track.appendChild(mark);
    row.appendChild(track);

    row.appendChild(el("span", "thermal-limit", `stops at ${gpu.stop}°C`));
    body.appendChild(row);
  });

  if (thermal?.reason) {
    body.appendChild(el("p", `run-warning is-${thermal.state}`, thermal.reason));
  }

  const logDetails = el("details", "run-log-details");
  logDetails.open = true;
  logDetails.appendChild(el("summary", null, "Log"));
  const pre = el("pre", "run-log", (task.logs || []).join("\n"));
  logDetails.appendChild(pre);
  body.appendChild(logDetails);

  body.appendChild(el("p", "run-hint",
    "Closing this does not stop the job. Reopen it from Jobs."));
  return { body, pre };
}

function openOverlay(task, thermal) {
  let overlay = document.getElementById("runOverlay");
  if (!overlay) {
    overlay = el("div", "modal-backdrop run-overlay");
    overlay.id = "runOverlay";
    overlay.addEventListener("mousedown", (event) => {
      if (event.target === overlay) {
        overlayDismissedFor = task.task_id;
        closeOverlay();
      }
    });
    document.body.appendChild(overlay);
  }

  const panel = el("div", "modal-container run-panel");
  const { body, pre } = buildOverlayBody(task, thermal);
  panel.appendChild(body);
  overlay.replaceChildren(panel);
  overlay.style.display = "flex";

  const logs = task.logs || [];
  if (logs.length !== lastLogCount) {
    lastLogCount = logs.length;
    pre.scrollTop = pre.scrollHeight;
  }
}

export function closeOverlay() {
  const overlay = document.getElementById("runOverlay");
  if (overlay) overlay.style.display = "none";
}

/** Reopen the live view for a job, from its row in Jobs. */
export function showRunningJob(taskId) {
  if (!latest.task || latest.task.task_id !== taskId) return false;
  overlayDismissedFor = null;
  openOverlay(latest.task, latest.thermal);
  return true;
}

// --- inline panel --------------------------------------------------------

function renderInline(task, accepting) {
  const panel = document.getElementById("liveTask");
  if (!panel) return;

  if (!task) {
    panel.replaceChildren(el("p", "live-idle",
      accepting ? "Idle — waiting for work." : "Not accepting work."));
    lastLogCount = 0;
    return;
  }

  const head = el("div", "live-head");
  head.appendChild(el("span", `job-status job-status-${task.status || "running"}`,
    task.status === "running" ? "Running" : (task.status || "")));
  head.appendChild(el("span", "live-name", task.model_name || task.task_id || "job"));

  if (task.status === "running") {
    const expand = el("button", "btn-ghost live-expand", "Open");
    expand.addEventListener("click", () => showRunningJob(task.task_id));
    head.appendChild(expand);
  }
  panel.replaceChildren(head);

  const pre = el("pre", "live-log", (task.logs || []).join("\n"));
  panel.appendChild(pre);
  pre.scrollTop = pre.scrollHeight;
}

// --- polling -------------------------------------------------------------

export async function pollLiveWork() {
  try {
    const res = await fetch("/current-task");
    if (!res.ok) throw new Error(`current-task returned ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    latest = { task: data.task, thermal: data.thermal };

    renderThermal(data.thermal);
    renderApproval(data.awaiting_approval);
    renderInline(data.task, data.accepting_work);

    const running = data.task && data.task.status === "running";
    if (running && overlayDismissedFor !== data.task.task_id) {
      openOverlay(data.task, data.thermal);   // a newly started job opens itself
    } else if (!running) {
      closeOverlay();
      lastLogCount = 0;
    }

    const toggle = document.getElementById("approvalModeToggle");
    if (toggle && document.activeElement !== toggle) {
      toggle.checked = data.approval_mode === "ask";
    }
  } catch (err) {
    console.error("Error polling live work:", err);
  }
}

export async function setApprovalMode(ask) {
  try {
    const res = await fetch("/approval-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: ask ? "ask" : "auto" }),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    pollLiveWork();
  } catch (err) {
    console.error("Could not change approval mode:", err);
  }
}

export function startLiveWorkPolling() {
  stopLiveWorkPolling();
  pollLiveWork();
  timer = setInterval(() => {
    if (!document.hidden) pollLiveWork();
  }, POLL_MS);

  const toggle = document.getElementById("approvalModeToggle");
  if (toggle) {
    toggle.addEventListener("change", (event) => setApprovalMode(event.target.checked));
  }
}

export function stopLiveWorkPolling() {
  if (timer) clearInterval(timer);
  timer = null;
}
