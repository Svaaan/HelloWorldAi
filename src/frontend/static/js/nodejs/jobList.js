// Job list for a node.
//
// Replaces the old "Pending Tasks" panel, which read the node's own in-memory
// queue. Nothing fills that queue any more: work is now held by the coordinator
// and pulled by the node, so the panel could only ever be empty.
//
// Everything here is built with createElement + textContent rather than
// innerHTML. Job fields come from whoever submitted the work, so treating them
// as markup would let a submitter run script in a contributor's dashboard.

const POLL_INTERVAL_MS = 8000;
const MAX_JOBS = 15;

let pollTimer = null;

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
  const then = new Date(value).getTime();
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

function buildJobCard(job) {
  const card = el("div", "job-item");

  const header = el("div", "job-header");
  header.appendChild(el("span", "job-id", job.task_id || "unknown"));
  header.appendChild(statusPill(job.status));
  card.appendChild(header);

  const meta = el("div", "job-meta");
  const model = job.task_data?.model_name || job.task_data?.task_type || "job";
  meta.appendChild(el("span", null, model));
  const submitted = relativeTime(job.submitted_at);
  if (submitted) meta.appendChild(el("span", "job-time", `submitted ${submitted}`));
  if (job.dataset_id) meta.appendChild(el("span", "job-badge", "dataset attached"));
  card.appendChild(meta);

  // Verification verdict — the reason the node can be trusted at all.
  const verdict = job.verification?.verdict;
  if (verdict) {
    const banner = el("div", `job-verdict job-verdict-${verdict}`);
    banner.appendChild(el("span", null, VERDICT_LABEL[verdict] || verdict));

    const failed = (job.verification.checks || [])
      .filter((check) => !check.passed)
      .map((check) => check.name);
    if (failed.length) {
      banner.appendChild(el("span", "job-verdict-detail", `failed: ${failed.join(", ")}`));
    }
    card.appendChild(banner);
  }

  const metrics = job.metrics || {};
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
    card.appendChild(grid);
  }

  if (job.result) {
    card.appendChild(el("p", "job-result", job.result));
  }

  if (Array.isArray(job.logs) && job.logs.length) {
    const details = el("details", "job-logs");
    details.appendChild(el("summary", null, `Log (${job.logs.length} lines)`));
    details.appendChild(el("pre", null, job.logs.join("\n")));
    card.appendChild(details);
  }

  return card;
}

export async function loadJobs() {
  const container = document.getElementById("taskList");
  if (!container) return;

  const nodeId = localStorage.getItem("currentNodeId");
  if (!nodeId) {
    container.replaceChildren(el("p", "job-empty", "Connect this node to see its jobs."));
    return;
  }

  try {
    const res = await fetch(`/tasks?node_id=${encodeURIComponent(nodeId)}&limit=${MAX_JOBS}`);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const jobs = await res.json();

    if (!Array.isArray(jobs)) {
      throw new Error(jobs?.error || "Unexpected response from the coordinator.");
    }

    if (jobs.length === 0) {
      container.replaceChildren(
        el("p", "job-empty", "No jobs yet. Work sent to this node will appear here.")
      );
      return;
    }

    const fragment = document.createDocumentFragment();
    jobs.forEach((job) => fragment.appendChild(buildJobCard(job)));
    container.replaceChildren(fragment);
  } catch (error) {
    console.error("Error loading jobs:", error);
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
