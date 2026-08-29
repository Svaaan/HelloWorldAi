// The data owner's view of their own jobs, and the download that ends the trip.
//
// Until this existed the product stopped one step short of its own point:
// someone uploaded data, a contributor's GPU trained a model, verification
// passed -- and there was no way to get the model back. The weights sat in
// storage with nothing pointing at them.
//
// Built with createElement + textContent. Job fields pass through nodes and
// back, so treating them as markup would let a node run script here.

import { hasSubmitterKey, submitterHeaders } from "./submitter.js";

const POLL_INTERVAL_MS = 10000;
const MAX_JOBS = 25;

let pollTimer = null;

// The workspace renders a summary from the same jobs this list draws, so both
// come from one fetch and cannot show different numbers.
let onJobs = null;

// Rows are rebuilt on every poll. Without this an expanded row slams shut a
// few seconds after it is opened, taking the download button with it.
const openJobs = new Set();

const STATUS_LABEL = {
  pending: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  rejected: "Declined",
  cancelled: "Cancelled",
};

// Nothing more will happen to a job in one of these states, so it can be run
// again but not stopped.
const FINISHED = ["completed", "failed", "rejected", "cancelled"];

const VERDICT_LABEL = {
  accepted: "Verified",
  suspicious: "Suspicious",
  rejected: "Failed verification",
};

// "Verified" answers whether the model is genuine, and people read it as
// whether the model is good. A text model that produced word-shaped nonsense
// came back marked Verified, which was true and gave entirely the wrong
// impression. The grade is shown beside the badge so the two questions stay
// separate and both get answered.
const STRENGTH_LABEL = {
  weak: "barely learned",
  clear: "learned",
  strong: "learned well",
};

function strengthNote(verification) {
  const measured = verification?.measured || {};
  const captured = measured.learned_fraction;
  const accuracy = measured.holdout_accuracy;
  const floor = measured.floor_accuracy ?? measured.baseline_accuracy;

  if (captured == null || accuracy == null || floor == null) return null;

  const pct = (value) => `${(value * 100).toFixed(1)}%`;
  const advice = {
    weak: "It found a real pattern, but a small one. For text that usually "
      + "means more data; for a table it can mean the columns do not carry "
      + "the answer.",
    clear: "A solid result. More data or more steps would still help.",
    strong: "Close to as good as this data allows.",
  }[verification.strength] || "";

  return `On data the node never saw it got ${pct(accuracy)} right, against `
    + `${pct(floor)} for a model that learned nothing — closing `
    + `${pct(captured)} of the gap. ${advice}`;
}

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

function notice(title, detail) {
  const box = el("div", "empty-message");
  box.appendChild(el("strong", null, title));
  if (detail) box.appendChild(el("p", null, detail));
  return box;
}

// --- the download -----------------------------------------------------------

// The key travels in a header, so a plain <a href> cannot fetch this: the file
// is pulled with fetch and handed to the browser as a blob.
async function downloadModel(job, button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Preparing…";

  try {
    const res = await fetch(`/artifacts/${encodeURIComponent(job.weights_id)}`, {
      headers: submitterHeaders(),
    });

    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail || `Server returned ${res.status}`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    const name = job.task_data?.model_name || "model";
    link.download = `${name}-${job.task_id.slice(0, 12)}.npz`;
    document.body.appendChild(link);
    link.click();
    link.remove();

    // Revoking immediately can cancel the save in some browsers.
    setTimeout(() => URL.revokeObjectURL(url), 30000);

    button.textContent = "Downloaded";
    setTimeout(() => { button.textContent = original; button.disabled = false; }, 2000);
  } catch (error) {
    console.error("Could not download the model:", error);
    button.textContent = "Download failed";
    button.disabled = false;
    setTimeout(() => { button.textContent = original; }, 2500);
  }
}

async function act(job, button, { path, verb, confirm }) {
  if (confirm && !window.confirm(confirm)) return;

  const original = button.textContent;
  button.disabled = true;
  button.textContent = `${verb}…`;

  try {
    const res = await fetch(`${path}/${encodeURIComponent(job.task_id)}`, {
      method: "POST",
      headers: submitterHeaders(),
    });

    const data = await res.json().catch(() => null);
    if (!res.ok) {
      const detail = data?.detail?.detail || data?.detail;
      throw new Error(detail || `Server returned ${res.status}`);
    }

    // The list refreshes on its own; reload now so the new state is immediate.
    await loadMyJobs();
  } catch (error) {
    console.error(`Could not ${verb.toLowerCase()} the job:`, error);
    button.textContent = error.message;
    button.disabled = false;
    setTimeout(() => { button.textContent = original; }, 3000);
  }
}

// --- rendering --------------------------------------------------------------

function metric(label, value) {
  const row = el("div", "job-metric");
  row.appendChild(el("span", "job-metric-label", label));
  row.appendChild(el("span", "job-metric-value", value));
  return row;
}

// What to actually do with the file once it is on disk.
function buildUsage(job) {
  const name = job.task_data?.model_name || "model";
  const filename = `${name}-${job.task_id.slice(0, 12)}.npz`;

  const box = el("details", "job-usage");
  box.appendChild(el("summary", null, "How to use this model"));

  const body = el("div", "job-usage-body");
  body.appendChild(el("p", null,
    "The file describes itself — the loader rebuilds the network and loads the "
    + "weights without needing anything from this project."));

  const link = el("a", "job-usage-link", "Download load_model.py");
  link.href = "/static/scripts/load_model.py";
  link.setAttribute("download", "load_model.py");
  body.appendChild(link);

  // No example feature vector here on purpose. The submitter rarely states
  // input_dim -- it is inferred from their CSV on the node -- so any number of
  // values printed here would be a guess, and a command that fails is worse
  // than one that explains itself. The first line reports what the model
  // expects, read from the file.
  //
  // The second line has to match what was actually trained: --input is for a
  // classifier, and printing it under a language model told people to run a
  // command that model refuses.
  const architecture = job.task_data?.model_spec?.architecture || "mlp";
  const isClassifier = ["mlp", "feedforward"].includes(architecture);

  const code = el("pre", "job-usage-code",
    `python load_model.py ${filename}\n`
    + (isClassifier
        ? `python load_model.py ${filename} --input <one value per feature>`
        : `python load_model.py ${filename} --prompt "some text to continue"`));
  body.appendChild(code);

  body.appendChild(el("p", "job-usage-hint",
    "The first command prints the model's inputs, outputs and training summary."));

  box.appendChild(body);
  return box;
}

// Same data, different settings. Only the numbers worth changing between two
// runs -- the dataset is fixed here, and its shape already decided the rest.
function buildTuner(job, onClose) {
  const spec = job.task_data?.model_spec || {};
  const hyper = job.task_data?.hyperparameters || {};
  const isText = !["mlp", "feedforward"].includes(spec.architecture || "mlp");

  const box = el("div", "job-tuner");
  box.appendChild(el("p", "job-field-hint",
    "Runs on the same data, scored against the same held-back rows — so the "
    + "difference in the result is your change, not a different split."));

  const grid = el("div", "job-field-grid");
  const inputs = new Map();

  const fields = isText
    ? [["d_model", "Model width", spec.d_model], ["n_layer", "Layers", spec.n_layer]]
    : [["hidden_dim", "Hidden width", spec.hidden_dim], ["depth", "Hidden layers", spec.depth]];

  fields.concat([
    ["steps", "Training steps", hyper.steps],
    ["learning_rate", "Learning rate", hyper.learning_rate],
  ]).forEach(([name, label, value]) => {
    const wrap = el("div", "job-field");
    const tag = el("label", "job-field-label", label);
    tag.htmlFor = `tune-${job.task_id}-${name}`;
    wrap.appendChild(tag);

    const input = document.createElement("input");
    input.type = "number";
    input.id = `tune-${job.task_id}-${name}`;
    input.value = value ?? "";
    if (name === "learning_rate") input.step = "any";
    wrap.appendChild(input);

    inputs.set(name, input);
    grid.appendChild(wrap);
  });
  box.appendChild(grid);

  const status = el("div", "field-status");

  const send = el("button", "btn", "Run with these settings");
  send.type = "button";
  send.addEventListener("click", async () => {
    send.disabled = true;
    status.replaceChildren(el("span", null, "Queueing…"));

    const read = (name) => {
      const raw = String(inputs.get(name).value).trim();
      const parsed = name === "learning_rate" ? Number(raw) : parseInt(raw, 10);
      return Number.isFinite(parsed) ? parsed : undefined;
    };

    const changes = { model_spec: {}, hyperparameters: {} };
    fields.forEach(([name]) => {
      const value = read(name);
      if (value !== undefined) changes.model_spec[name] = value;
    });
    ["steps", "learning_rate"].forEach((name) => {
      const value = read(name);
      if (value !== undefined) changes.hyperparameters[name] = value;
    });

    try {
      const res = await fetch(`/retry-task/${encodeURIComponent(job.task_id)}`, {
        method: "POST",
        headers: submitterHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(changes),
      });
      const data = await res.json();
      if (!res.ok || data.status !== "success") {
        throw new Error(data?.detail?.detail || data?.detail || "The coordinator refused it.");
      }

      status.replaceChildren(el("span", "success-message",
        `Queued as ${data.task_id}. It will appear above when it finishes.`));
      (data.notes || []).forEach((text) =>
        status.appendChild(el("p", "field-advice", text)));
      // The list refreshes on its own; reload now so the new run is immediate.
      await loadMyJobs();
    } catch (error) {
      console.error("Could not queue the adjusted run:", error);
      status.replaceChildren(el("span", "error-message", error.message));
      send.disabled = false;
    }
  });

  const cancel = el("button", "btn-ghost", "Cancel");
  cancel.type = "button";
  cancel.addEventListener("click", () => {
    box.remove();
    if (onClose) onClose();
  });

  const row = el("div", "job-actions");
  row.appendChild(send);
  row.appendChild(cancel);
  box.appendChild(row);
  box.appendChild(status);

  return box;
}


function buildJobRow(job) {
  const row = el("details", "job-row");
  row.open = openJobs.has(job.task_id);
  row.addEventListener("toggle", () => {
    if (row.open) openJobs.add(job.task_id);
    else openJobs.delete(job.task_id);
  });

  const summary = el("summary", "job-summary");

  summary.appendChild(el("span", `job-status job-status-${job.status || "unknown"}`,
    STATUS_LABEL[job.status] || job.status || "Unknown"));

  const title = el("div", "job-title");
  title.appendChild(el("span", "job-name",
    job.task_data?.model_name || job.task_data?.task_type || "job"));
  summary.appendChild(title);

  const verdict = job.verification?.verdict;
  if (verdict) {
    summary.appendChild(el("span", `job-verdict job-verdict-${verdict}`,
      VERDICT_LABEL[verdict] || verdict));

    const strength = job.verification?.strength;
    if (strength) {
      summary.appendChild(el("span", `job-strength job-strength-${strength}`,
        STRENGTH_LABEL[strength] || strength));
    }
  }

  // The thing the owner is actually waiting for.
  if (job.status === "completed" && job.weights_id) {
    summary.appendChild(el("span", "job-headline", "model ready"));
  }

  summary.appendChild(el("span", "job-time", relativeTime(job.submitted_at)));
  row.appendChild(summary);

  const body = el("div", "job-body");
  body.appendChild(el("div", "job-id", job.task_id));

  const metrics = job.metrics || {};
  if (Object.keys(metrics).length) {
    const grid = el("div", "job-metrics");
    if (metrics.final_loss !== undefined && metrics.final_loss !== null) {
      grid.appendChild(metric("Loss", `${metrics.initial_loss} → ${metrics.final_loss}`));
    }
    if (metrics.achieved_tflops) grid.appendChild(metric("Achieved", `${metrics.achieved_tflops} TFLOPS`));
    if (metrics.steps) grid.appendChild(metric("Steps", metrics.steps));
    if (metrics.dataset_rows) grid.appendChild(metric("Rows", metrics.dataset_rows.toLocaleString()));
    body.appendChild(grid);
  }

  if (job.result) body.appendChild(el("p", "job-result", job.result));

  const grading = strengthNote(job.verification);
  if (grading) body.appendChild(el("p", "job-grade", grading));

  // What the model actually writes. This is the only question anyone has
  // about a finished language model, and answering it used to require a
  // download, a Python install and a script.
  const samples = job.metrics?.samples;
  if (Array.isArray(samples) && samples.length) {
    const box = el("details", "job-samples");
    box.appendChild(el("summary", null,
      `What it writes now (${samples.length} samples)`));

    const inner = el("div", "job-samples-body");
    inner.appendChild(el("p", "job-field-hint",
      "Each starts from a short snippet of your own text — shown in bold — "
      + "and the model continued it."));

    samples.forEach((sample) => {
      const block = el("p", "job-sample");
      block.appendChild(el("strong", null, sample.prompt || ""));
      block.appendChild(el("span", null, sample.continuation || ""));
      inner.appendChild(block);
    });

    box.appendChild(inner);
    body.appendChild(box);
  }

  const actions = el("div", "job-actions");

  if (!FINISHED.includes(job.status)) {
    const cancel = el("button", "btn-ghost", "Cancel job");
    cancel.type = "button";
    cancel.addEventListener("click", () => act(job, cancel, {
      path: "/cancel-task",
      verb: "Cancelling",
      confirm: job.status === "running"
        ? "Stop this job? The work done so far is lost."
        : null,
    }));
    actions.appendChild(cancel);
  } else if (job.can_rerun) {
    const again = el("button", "btn-ghost", "Run again");
    again.type = "button";
    again.addEventListener("click", () => act(job, again, {
      path: "/retry-task",
      verb: "Queueing",
    }));
    actions.appendChild(again);

    // The useful half. "Run again" repeats a result you have already seen;
    // what you want after reading the grade is the same data with different
    // numbers -- and re-uploading the file to get that would score the next
    // run against a different holdout, making the comparison meaningless.
    const tune = el("button", "btn-ghost", "Adjust and run");
    tune.type = "button";
    tune.addEventListener("click", () => {
      tune.disabled = true;
      body.insertBefore(buildTuner(job, () => { tune.disabled = false; }),
                        actions.nextSibling);
    });
    actions.appendChild(tune);
  } else if (job.status === "completed") {
    // Its data has been deleted, so there is nothing to run it against.
    actions.appendChild(el("p", "job-note",
      "The data behind this job has been deleted, so it cannot be run again. "
      + "Send it as a new job with the file."));
  }

  if (actions.childElementCount) body.appendChild(actions);

  if (job.status === "completed" && job.weights_id) {
    const download = el("button", "btn", "Download trained model");
    download.type = "button";
    download.addEventListener("click", () => downloadModel(job, download));
    body.appendChild(download);

    // A downloaded .npz is a bag of arrays until someone knows what to do with
    // it. The file describes itself, so the whole answer is two commands.
    body.appendChild(buildUsage(job));

    if (!verdict && job.has_holdout) {
      body.appendChild(el("p", "job-note", "Verification is still running."));
    } else if (!job.has_holdout) {
      body.appendChild(el("p", "job-note",
        "No dataset was attached, so this result could not be verified."));
    }
  } else if (job.status === "completed") {
    body.appendChild(el("p", "job-note",
      "This job finished without returning a model."));
  }

  row.appendChild(body);
  return row;
}

export function setJobsListener(fn) {
  onJobs = fn;
}

export async function loadMyJobs() {
  const container = document.getElementById("myJobsList");
  if (!container) return;

  // Asking for the key here would mint one for a visitor who has never sent
  // anything, and the panel would claim they have jobs to look at.
  if (!hasSubmitterKey()) {
    onJobs?.([]);
    container.replaceChildren(notice(
      "Nothing sent yet",
      "Jobs you send will appear here, and finished models can be downloaded from this panel."
    ));
    return;
  }

  try {
    const res = await fetch(`/my-tasks?limit=${MAX_JOBS}`, { headers: submitterHeaders() });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const jobs = await res.json();
    if (!Array.isArray(jobs)) throw new Error("Unexpected response from the coordinator.");

    onJobs?.(jobs);

    const count = document.getElementById("myJobsCount");
    if (count) {
      const ready = jobs.filter((j) => j.status === "completed" && j.weights_id).length;
      count.textContent = jobs.length
        ? (ready ? `${jobs.length} sent · ${ready} ready to download` : `${jobs.length} sent`)
        : "";
    }

    if (!jobs.length) {
      container.replaceChildren(notice(
        "Nothing sent yet",
        "Jobs you send will appear here, and finished models can be downloaded from this panel."
      ));
      return;
    }

    const present = new Set(jobs.map((job) => job.task_id));
    [...openJobs].forEach((id) => { if (!present.has(id)) openJobs.delete(id); });

    const list = el("div", "job-list");
    jobs.forEach((job) => list.appendChild(buildJobRow(job)));
    container.replaceChildren(list);
  } catch (error) {
    console.error("Could not load your jobs:", error);
    container.replaceChildren(notice("Could not load your jobs", error.message));
  }
}

export function startMyJobsPolling() {
  if (pollTimer) clearInterval(pollTimer);
  loadMyJobs();
  pollTimer = setInterval(loadMyJobs, POLL_INTERVAL_MS);
}
