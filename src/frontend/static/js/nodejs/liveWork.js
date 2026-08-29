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

// What the overlay and the inline panel currently hold. Rebuilding either from
// scratch every two seconds made both flash on every poll, so the DOM is only
// replaced when its shape changes; otherwise the values are written in place.
let overlayShape = null;
let inlineShape = null;
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

  const shown = el("span", "run-stat-value", value);
  // Keyed so a later poll can rewrite the number without touching the node
  // around it.
  shown.dataset.f = label;
  box.appendChild(shown);
  return box;
}

/** Write text only when it differs, so unchanged nodes are never touched. */
function setText(root, key, value) {
  const node = root.querySelector(`[data-f="${CSS.escape(key)}"]`);
  if (node && node.textContent !== String(value)) node.textContent = String(value);
}

function setWidth(node, percent) {
  const next = `${percent}%`;
  if (node && node.style.width !== next) node.style.width = next;
}

/**
 * What the rendered structure depends on -- as opposed to the values inside it.
 * While this is unchanged the same DOM can be reused and simply rewritten.
 */
function shapeOf(task, thermal) {
  const p = task.progress || {};
  const gpus = (thermal?.gpus || []).map((g) => [
    g.name,
    g.utilisation != null,
    g.power_w != null,
    g.fan_percent != null,
    g.memory_used_mb != null && g.memory_total_mb != null,
    g.state,
  ].join(","));

  return [
    task.task_id,
    task.status,
    p.steps ? "progress" : "",
    p.label ? "labelled" : "",
    task.self_test ? "selftest" : "",
    p.loss != null ? "loss" : "",
    p.initial_loss != null ? "initial" : "",
    thermal?.reason ? "warning" : "",
    gpus.join("|"),
  ].join("~");
}

/** Rewrite the numbers in an overlay that is already on screen. */
function updateOverlayBody(body, task, thermal) {
  const progress = task.progress;

  if (progress && progress.steps) {
    const pct = Math.min(100, (progress.step / progress.steps) * 100);
    setText(body, "progress-step",
      progress.label || `Step ${progress.step} of ${progress.steps}`);
    setText(body, "progress-pct", `${Math.round(pct)}%`);
    setWidth(body.querySelector(".usage-bar-fill"), pct);
  }

  setText(body, "Elapsed", formatElapsed(task.elapsed_s));
  if (progress && progress.loss != null) setText(body, "Loss", progress.loss);

  (thermal?.gpus || []).forEach((gpu, index) => {
    setText(body, `GPU load`, gpu.utilisation == null ? "—" : `${gpu.utilisation}%`);
    setText(body, `Temperature`, `${gpu.temperature}°C`);
    if (gpu.power_w != null) setText(body, "Power", `${gpu.power_w} W`);
    if (gpu.fan_percent != null) setText(body, "Fan", `${gpu.fan_percent}%`);
    if (gpu.memory_used_mb != null && gpu.memory_total_mb != null) {
      setText(body, "VRAM",
        `${(gpu.memory_used_mb / 1024).toFixed(1)} / ${(gpu.memory_total_mb / 1024).toFixed(1)} GB`);
    }

    const row = body.querySelectorAll(".run-thermal")[index];
    if (row) {
      const span = Math.max(1, gpu.stop - 30);
      setWidth(row.querySelector(".thermal-fill"),
        Math.max(0, Math.min(100, ((gpu.temperature - 30) / span) * 100)));
    }
  });

  updateLog(body.querySelector(".run-log"), task.logs || []);
}

/**
 * Keep the log current without stealing the reader's place in it.
 *
 * This used to be recreated and scrolled to the bottom on every poll, which
 * jumped the panel every two seconds and yanked anyone who had scrolled up
 * back down again.
 */
function updateLog(pre, logs) {
  if (!pre) return;

  const text = logs.join("\n");
  if (pre.textContent === text) return;

  // Only follow the tail for somebody already reading the tail.
  const atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 24;
  pre.textContent = text;
  if (atBottom) pre.scrollTop = pre.scrollHeight;
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

  // Your own test is the one thing here you are entitled to end, so the
  // control belongs in front of you rather than behind the panel you are
  // looking at.
  if (task.self_test && task.status === "running") {
    const stop = el("button", "run-stop", "Stop test");
    stop.type = "button";
    stop.addEventListener("click", async () => {
      stop.disabled = true;
      stop.textContent = "Stopping…";
      try {
        const res = await fetch("/self-test/stop", { method: "POST" });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
      } catch (error) {
        console.error("Could not stop the test:", error);
        stop.textContent = "Stop test";
        stop.disabled = false;
      }
      // It stops at its next step; the poll that follows reports it.
    });
    head.appendChild(stop);
  }

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
    // A run bounded by time labels itself; step counts are for the rest.
    const stepText = el("span", null,
      progress.label || `Step ${progress.step} of ${progress.steps}`);
    stepText.dataset.f = "progress-step";
    line.appendChild(stepText);

    const pctText = el("span", "run-progress-pct",
      `${Math.round((progress.step / progress.steps) * 100)}%`);
    pctText.dataset.f = "progress-pct";
    line.appendChild(pctText);
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

  // A test never becomes a queued job, so it cannot be reopened from Jobs --
  // which is what this used to tell people to do.
  body.appendChild(el("p", "run-hint",
    task.self_test
      ? "Closing this does not stop the test. Reopen it from Test this machine."
      : "Closing this does not stop the job. Reopen it from Jobs."));
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

  const shape = shapeOf(task, thermal);
  const existing = overlay.querySelector(".run-body");

  if (existing && shape === overlayShape) {
    // Same structure: rewrite the numbers and leave the nodes alone. Replacing
    // them wholesale is what made the panel flash every two seconds.
    updateOverlayBody(existing, task, thermal);
    return;
  }

  const panel = el("div", "modal-container run-panel");
  const { body, pre } = buildOverlayBody(task, thermal);
  panel.appendChild(body);
  overlay.replaceChildren(panel);
  overlay.style.display = "flex";
  overlayShape = shape;

  pre.scrollTop = pre.scrollHeight;
  lastLogCount = (task.logs || []).length;
}

export function closeOverlay() {
  const overlay = document.getElementById("runOverlay");
  if (overlay) overlay.style.display = "none";
  overlayShape = null;
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
    const idle = accepting ? "Idle — waiting for work." : "Not accepting work.";
    if (panel.dataset.shape !== `idle:${idle}`) {
      panel.replaceChildren(el("p", "live-idle", idle));
      panel.dataset.shape = `idle:${idle}`;
      inlineShape = null;
    }
    lastLogCount = 0;
    return;
  }

  const shape = [task.task_id, task.status, accepting].join("~");
  if (shape === inlineShape) {
    // Only the log moves between polls; rebuilding the panel around it was
    // what made this jump up and down.
    updateLog(panel.querySelector(".live-log"), task.logs || []);
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

  inlineShape = shape;
  panel.dataset.shape = shape;
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
