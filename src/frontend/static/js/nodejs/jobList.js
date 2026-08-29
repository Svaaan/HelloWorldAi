// Job list for a node.
//
// Replaces the old "Pending Tasks" panel, which read the node's own in-memory
// queue. Nothing fills that queue any more: work is now held by the coordinator
// and pulled by the node, so the panel could only ever be empty.
//
// Everything here is built with createElement + textContent rather than
// innerHTML. Job fields come from whoever submitted the work, so treating them
// as markup would let a submitter run script in a contributor's dashboard.

import { showRunningJob } from "./liveWork.js";

const POLL_INTERVAL_MS = 8000;
const MAX_JOBS = 15;

let pollTimer = null;

// Rows are rebuilt from scratch on every poll. Without this, an expanded row
// slams shut a few seconds after someone opens it, and the row under their
// cursor shifts mid-click onto a different job. Remember what is open and
// restore it after each rebuild.
const openJobs = new Set();

const STATUS_LABEL = {
  pending: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  rejected: "Rejected",
};

const VERDICT_LABEL = {
  accepted: "Verified",
  suspicious: "Suspicious",
  rejected: "Failed verification",
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function relativeTime(value) {
  if (!value) return "";
  // The coordinator stamps times with datetime.utcnow(), which serialises
  // without a zone -- and a zoneless date-time is parsed as *local* time here.
  // On a machine two hours ahead of UTC that made a job sent a moment ago read
  // "2h ago". Say UTC when nothing else says otherwise.
  const stamped = /(Z|[+-]\d{2}:?\d{2})$/.test(value) ? value : `${value}Z`;
  const then = new Date(stamped).getTime();
  if (Number.isNaN(then)) return "";
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

function statusPill(status) {
  const known = STATUS_LABEL[status] || status || "Unknown";
  const pill = el("span", `job-status job-status-${status || "unknown"}`, known);
  return pill;
}

function metricLine(label, value) {
  const row = el("div", "job-metric");
  row.appendChild(el("span", "job-metric-label", label));
  row.appendChild(el("span", "job-metric-value", value));
  return row;
}

function emptyState(title, detail) {
  const box = el("div", "job-empty");
  box.appendChild(el("strong", null, title));
  box.appendChild(el("span", null, detail));
  return box;
}

// A full-width panel of stacked cards left content pinned to the far edges with
// dead space between. A compact disclosure row fits the shape far better: the
// collapsed line carries the whole summary, detail is one click away.
function buildJobRow(job) {
  const row = el("details", "job-row");
  row.open = openJobs.has(job.task_id);
  row.addEventListener("toggle", () => {
    if (row.open) openJobs.add(job.task_id);
    else openJobs.delete(job.task_id);
  });

  const summary = el("summary", "job-summary");

  summary.appendChild(statusPill(job.status));

  const title = el("div", "job-title");
  title.appendChild(el("span", "job-name",
    job.task_data?.model_name || job.task_data?.task_type || "job"));
  if (job.dataset_id) title.appendChild(el("span", "job-badge", "data"));
  summary.appendChild(title);

  // One headline figure, so a collapsed row still says something useful.
  const metrics = job.metrics || {};
  if (metrics.achieved_tflops) {
    summary.appendChild(el("span", "job-headline", `${metrics.achieved_tflops} TFLOPS`));
  } else if (metrics.final_loss !== undefined && metrics.final_loss !== null) {
    summary.appendChild(el("span", "job-headline", `loss ${metrics.final_loss}`));
  }

  const verdict = job.verification?.verdict;
  if (verdict) {
    summary.appendChild(el("span", `job-verdict job-verdict-${verdict}`,
      VERDICT_LABEL[verdict] || verdict));
  }

  summary.appendChild(el("span", "job-time", relativeTime(job.submitted_at)));
  row.appendChild(summary);

  // --- detail, revealed on expand ---
  const body = el("div", "job-body");
  body.appendChild(el("div", "job-id", job.task_id || "unknown"));

  // A running job has no metrics yet; offer the live view instead.
  if (job.status === "running") {
    const open = el("button", "btn-ghost", "Watch it run");
    open.addEventListener("click", () => {
      if (!showRunningJob(job.task_id)) {
        open.textContent = "That job is no longer running here";
        open.disabled = true;
      }
    });
    body.appendChild(open);
  }

  if (verdict) {
    const failed = (job.verification.checks || [])
      .filter((check) => !check.passed)
      .map((check) => check.name);
    if (failed.length) {
      body.appendChild(el("p", "job-verdict-detail", `Failed checks: ${failed.join(", ")}`));
    }
  }

  if (Object.keys(metrics).length) {
    const grid = el("div", "job-metrics");
    if (metrics.final_loss !== undefined && metrics.final_loss !== null) {
      grid.appendChild(metricLine("Loss", `${metrics.initial_loss} → ${metrics.final_loss}`));
    }
    if (metrics.achieved_tflops) grid.appendChild(metricLine("Achieved", `${metrics.achieved_tflops} TFLOPS`));
    if (metrics.device_count) grid.appendChild(metricLine("GPUs used", metrics.device_count));
    if (metrics.dataset_rows) grid.appendChild(metricLine("Rows", metrics.dataset_rows.toLocaleString()));
    else if (metrics.synthetic_data) grid.appendChild(metricLine("Data", "synthetic"));
    if (metrics.steps) grid.appendChild(metricLine("Steps", metrics.steps));
    body.appendChild(grid);
  }

  if (job.result) body.appendChild(el("p", "job-result", job.result));

  if (Array.isArray(job.logs) && job.logs.length) {
    const logs = el("details", "job-logs");
    logs.appendChild(el("summary", null, `Log (${job.logs.length} lines)`));
    logs.appendChild(el("pre", null, job.logs.join("\n")));
    body.appendChild(logs);
  }

  row.appendChild(body);
  return row;
}

function updateJobCount(jobs) {
  const label = document.getElementById("jobCount");
  if (!label) return;

  if (!jobs || jobs.length === 0) {
    label.textContent = "";
    return;
  }
  const active = jobs.filter(j => j.status === "running" || j.status === "pending").length;
  label.textContent = active
    ? `${jobs.length} total, ${active} active`
    : `${jobs.length} total`;
}

export async function loadJobs() {
  const container = document.getElementById("taskList");
  if (!container) return;

  const nodeId = localStorage.getItem("currentNodeId");
  if (!nodeId) {
    updateJobCount([]);
    container.replaceChildren(emptyState(
      "Not connected",
      "Connect this node to see the work it has run."
    ));
    return;
  }

  try {
    const res = await fetch(`/tasks?node_id=${encodeURIComponent(nodeId)}&limit=${MAX_JOBS}`);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const jobs = await res.json();

    if (!Array.isArray(jobs)) {
      throw new Error(jobs?.error || "Unexpected response from the coordinator.");
    }

    updateJobCount(jobs);

    // Drop remembered rows for jobs that have aged out of the list.
    const present = new Set(jobs.map((job) => job.task_id));
    [...openJobs].forEach((id) => { if (!present.has(id)) openJobs.delete(id); });

    if (jobs.length === 0) {
      container.replaceChildren(emptyState(
        "No jobs yet",
        "Work sent to this node will appear here as it runs."
      ));
      return;
    }

    const list = el("div", "job-list");
    jobs.forEach((job) => list.appendChild(buildJobRow(job)));
    container.replaceChildren(list);
  } catch (error) {
    console.error("Error loading jobs:", error);
    updateJobCount([]);
    container.replaceChildren(
      el("p", "error-message", `Could not load jobs. ${error.message}`)
    );
  }
}

export function startJobPolling() {
  if (pollTimer) clearInterval(pollTimer);
  loadJobs();
  pollTimer = setInterval(loadJobs, POLL_INTERVAL_MS);
}

export function stopJobPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}
