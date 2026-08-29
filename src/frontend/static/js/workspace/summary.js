// What the builder has actually got out of the network so far.
//
// The GPU owner's page opens with what their hardware is doing. This is the
// mirror of that for the other side: not "what is my card doing" but "what has
// the network done for me" -- how much work landed, how much of it can be
// trusted, and how much compute it took.
//
// Derived from the job list rather than a new endpoint, so it cannot disagree
// with the rows shown underneath it.

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function stat(label, value, hint) {
  const box = el("div", "ws-stat");
  box.appendChild(el("span", "ws-stat-label", label));
  box.appendChild(el("span", "ws-stat-value", value));
  if (hint) box.appendChild(el("span", "ws-stat-hint", hint));
  return box;
}

export function summarise(jobs) {
  const done = jobs.filter((j) => j.status === "completed");
  const ready = done.filter((j) => j.weights_id);
  const verified = jobs.filter((j) => j.verification?.verdict === "accepted");
  const doubted = jobs.filter((j) => ["rejected", "suspicious"]
    .includes(j.verification?.verdict));
  const running = jobs.filter((j) => j.status === "running" || j.status === "pending");

  const rows = jobs.reduce((sum, j) => sum + (Number(j.metrics?.dataset_rows) || 0), 0);
  const steps = jobs.reduce((sum, j) => sum + (Number(j.metrics?.steps) || 0), 0);

  return { jobs, done, ready, verified, doubted, running, rows, steps };
}

export function renderSummary(jobs) {
  const container = document.getElementById("workspaceSummary");
  if (!container) return;

  const s = summarise(jobs);

  if (!jobs.length) {
    container.replaceChildren(el("p", "ws-empty",
      "Nothing sent yet. Pick a node and send it a training job — results show up here."));
    return;
  }

  const grid = el("div", "ws-stats");

  grid.appendChild(stat("Models ready", s.ready.length,
    s.ready.length ? "download below" : null));

  grid.appendChild(stat("Jobs sent", jobs.length,
    s.running.length ? `${s.running.length} still working` : null));

  // Verification is the reason to trust a result from someone else's machine,
  // so it gets its own figure rather than hiding inside each row.
  grid.appendChild(stat(
    "Verified",
    `${s.verified.length}/${s.done.length}`,
    s.doubted.length ? `${s.doubted.length} failed checks` : "scored on held-back data",
  ));

  grid.appendChild(stat("Training steps", s.steps.toLocaleString(),
    s.rows ? `${s.rows.toLocaleString()} rows` : null));

  container.replaceChildren(grid);

  if (s.doubted.length) {
    const warn = el("p", "ws-warning",
      `${s.doubted.length} result${s.doubted.length === 1 ? "" : "s"} did not pass verification. ` +
      "Those models scored no better on held-back data than an untrained one.");
    container.appendChild(warn);
  }
}
