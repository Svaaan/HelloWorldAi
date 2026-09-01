// What is happening to your data right now, on somebody else's machine.
//
// The workspace could tell you a job was "Running" and nothing else. You hand
// your data to a stranger's graphics card and then watch a word sit there --
// for a few minutes on a small job, considerably longer on a real one, with no
// way to tell the difference between working and stuck.
//
// The contributor lending the card has had a live view of this the whole time
// on their node page: which card, how hot, how loaded. None of it was secret --
// /nodes serves it to anyone who asks, because that is how somebody picks a
// machine to send work to in the first place. It simply was not shown to the
// person whose data was on it.
//
// So this is the same picture from the other side, built out of what is already
// published: the machine your job went to, and how it is doing.
//
// What is deliberately not here is training progress -- epochs, loss. The node
// tracks that (current_task["progress"]) but only ever tells its own dashboard;
// it never reaches the coordinator, so the workspace cannot honestly show it.
// Inventing a progress bar out of elapsed time would be a guess presented as a
// measurement, and this panel exists because a word with no information behind
// it is not worth the space.

const RUNNING = "running";
const QUEUED = "pending";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

/** How long ago, in words. Jobs run for minutes to hours; seconds are noise. */
export function elapsedSince(startedAt, now = Date.now()) {
  if (!startedAt) return null;

  const started = new Date(startedAt).getTime();
  if (Number.isNaN(started)) return null;

  // A clock that disagrees with the server's should read as "just now" rather
  // than as a negative duration.
  const seconds = Math.max(0, Math.round((now - started) / 1000));

  if (seconds < 60) return "less than a minute";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"}`;

  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  if (!rest) return `${hours} hour${hours === 1 ? "" : "s"}`;
  return `${hours}h ${rest}m`;
}

/** The first GPU on a node record, whatever shape the capabilities took. */
export function firstGpu(node) {
  const gpus = node?.capabilities?.gpu;
  if (Array.isArray(gpus)) return gpus[0] || null;
  return gpus || null;
}

/** The readings worth showing, skipping any the node did not report. */
export function readings(gpu) {
  if (!gpu) return [];

  const out = [];
  if (typeof gpu.load_percentage === "number") {
    out.push(["Load", `${gpu.load_percentage}%`]);
  }
  if (typeof gpu.temperature === "number") {
    out.push(["Temperature", `${gpu.temperature} °C`]);
  }
  if (typeof gpu.free_memory === "number" && typeof gpu.total_memory === "number") {
    const used = Math.max(0, gpu.total_memory - gpu.free_memory);
    out.push(["Memory", `${(used / 1024).toFixed(1)} / ${(gpu.total_memory / 1024).toFixed(1)} GB`]);
  }
  return out;
}

async function fetchNodes() {
  try {
    const res = await fetch("/nodes");
    if (!res.ok) return [];
    const nodes = await res.json();
    return Array.isArray(nodes) ? nodes : [];
  } catch {
    return [];
  }
}

function renderIdle(host, queued) {
  host.replaceChildren();

  if (queued) {
    host.appendChild(el("p", "ws-empty",
      queued === 1
        ? "One job is waiting for a free machine."
        : `${queued} jobs are waiting for a free machine.`));
    return;
  }

  host.appendChild(el("p", "ws-empty", "Nothing running right now."));
}

function renderJob(host, job, node) {
  const gpu = firstGpu(node);

  const row = el("div", "ws-running-job");

  const name = el("div", "ws-running-gpu",
    gpu?.name || "A machine on the network");
  row.appendChild(name);

  const since = elapsedSince(job.started_at);
  if (since) {
    row.appendChild(el("div", "ws-running-since", `Running for ${since}`));
  }

  const stats = el("dl", "ws-running-stats");
  for (const [label, value] of readings(gpu)) {
    stats.appendChild(el("dt", null, label));
    stats.appendChild(el("dd", null, value));
  }
  if (stats.childElementCount) row.appendChild(stats);

  // The node has gone quiet, which the coordinator will eventually act on by
  // requeueing the job. Saying so beats a temperature reading from ten minutes
  // ago presented as current.
  if (node && node.isConnected === false) {
    row.appendChild(el("p", "ws-running-stale",
      "This machine has stopped reporting. If it does not come back, the job "
      + "goes back in the queue for another one."));
  }

  host.appendChild(row);
}

/**
 * Draw the panel from the jobs the workspace already polls.
 *
 * Shares that poll rather than adding one: the list refreshes every ten
 * seconds, and a second timer would double the traffic to say the same thing.
 */
export async function renderRunning(jobs) {
  const host = document.getElementById("workspaceRunning");
  if (!host) return;

  // Told nothing, rather than told there is nothing. Saying "Nothing running"
  // because the coordinator did not answer would be the panel making something
  // up, which is the opposite of what it is for.
  if (!Array.isArray(jobs)) {
    host.replaceChildren(el("p", "ws-empty",
      "Could not reach the coordinator just now."));
    return;
  }

  const all = jobs;
  const running = all.filter((job) => job.status === RUNNING);
  const queued = all.filter((job) => job.status === QUEUED).length;

  if (!running.length) {
    renderIdle(host, queued);
    return;
  }

  // Only ask about the machines once something is actually on one.
  const nodes = await fetchNodes();
  const byId = new Map(nodes.map((n) => [n.node_id, n]));

  host.replaceChildren();
  for (const job of running) {
    renderJob(host, job, byId.get(job.node_id));
  }

  if (queued) {
    host.appendChild(el("p", "ws-running-queued",
      queued === 1
        ? "One more job is waiting for a free machine."
        : `${queued} more jobs are waiting for a free machine.`));
  }
}
