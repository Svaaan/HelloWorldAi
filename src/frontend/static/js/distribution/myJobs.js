// The data owner's view of their own jobs, and the download that ends the trip.
//
// Until this existed the product stopped one step short of its own point:
// someone uploaded data, a contributor's GPU trained a model, verification
// passed -- and there was no way to get the model back. The weights sat in
// storage with nothing pointing at them.
//
// Built with createElement + textContent. Job fields pass through nodes and
// back, so treating them as markup would let a node run script here.

import { hasSubmitterKey, isSignedIn, submitterHeaders } from "./submitter.js";

const POLL_INTERVAL_MS = 10000;
const MAX_JOBS = 25;

let pollTimer = null;

// The workspace renders a summary from the same jobs this list draws, so both
// come from one fetch and cannot show different numbers.
let onJobs = null;

// Rows are rebuilt on every poll. Without this an expanded row slams shut a
// few seconds after it is opened, taking the download button with it.
const openJobs = new Set();

// Jobs with an open "Adjust and run" panel.
//
// The list refreshes every ten seconds by rebuilding every row, which is fine
// for a status view and fatal for a form: the panel and everything typed into
// it disappeared mid-sentence. Nobody fills in four fields in ten seconds.
//
// A row being *open* survives a rebuild through openJobs above, because that
// is one boolean. Half-typed input is not something to rebuild -- so while
// somebody is editing, the refresh waits.
//
// Keyed by job *and* panel: a row can have both the settings form and the
// prompt box open, and closing one must not resume the refresh under the
// other.
const editingJobs = new Set();

function beginEditing(taskId, panel) {
  editingJobs.add(`${taskId}:${panel}`);
}

function endEditing(taskId, panel) {
  editingJobs.delete(`${taskId}:${panel}`);
}

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

/** The other runs on this same data, and what they scored.
 *
 * The loop a person actually works in is: change one thing, run it again, did
 * that help. The list answered the first two and not the third -- every run was
 * a separate row, and comparing two of them meant opening both and holding four
 * numbers in your head.
 *
 * A retry keeps the dataset it was made from, so runs on the same data are
 * exactly the family worth putting side by side. Anything else on the page is
 * a different question and stays where it is.
 */
function familyNote(job, byId) {
  if (!job.dataset_id || !byId) return null;

  const family = [...byId.values()]
    .filter((other) => other.dataset_id === job.dataset_id)
    .filter((other) => (other.verification?.measured || {}).holdout_accuracy != null);

  if (family.length < 2) return null;      // nothing to compare against yet

  // Oldest first, so the table reads as the order the work happened in.
  family.sort((a, b) => String(a.submitted_at || "").localeCompare(
    String(b.submitted_at || "")));

  const box = el("div", "job-family");
  box.appendChild(el("h4", "job-family-title", "Runs on this data"));

  const table = el("table", "job-family-table");
  const head = el("tr");
  ["Run", "Steps", "Scored", "Gap closed"].forEach(
    (heading) => head.appendChild(el("th", null, heading)));
  table.appendChild(head);

  const best = Math.max(...family.map(
    (other) => other.verification.measured.holdout_accuracy));

  family.forEach((other) => {
    const measured = other.verification.measured;
    const row = el("tr", other.task_id === job.task_id ? "is-this-one" : null);

    const name = el("td", null,
      other.task_data?.model_name || other.task_id.slice(0, 12));
    if (other.task_id === job.task_id) name.appendChild(el("span", "job-family-here", " this one"));
    row.appendChild(name);

    row.appendChild(el("td", null,
      other.task_data?.hyperparameters?.steps ?? "—"));

    const scored = el("td", null,
      `${(measured.holdout_accuracy * 100).toFixed(1)}%`);
    // Marking the best is the whole point: it is the answer to "did that help".
    if (measured.holdout_accuracy === best && family.length > 1) {
      scored.className = "is-best";
      scored.appendChild(el("span", "job-family-here", " best"));
    }
    row.appendChild(scored);

    row.appendChild(el("td", null, measured.learned_fraction != null
      ? `${(measured.learned_fraction * 100).toFixed(0)}%` : "—"));

    table.appendChild(row);
  });

  box.appendChild(table);
  box.appendChild(el("p", "job-note",
    "Scores here are comparable because every run was judged on the same "
    + "held-back rows. Against a different dataset they are not."));
  return box;
}


/** Why a queued job has not started, for as long as it has not started.
 *
 * A job sent to a machine that is switched off gets moved to another one after
 * a quarter of an hour; requeue_stale_tasks has always done that. What nobody
 * did was say so, and "Queued" on its own for fifteen minutes reads as a
 * service that does not work, which is a poor outcome for a rescue that was
 * coming anyway.
 *
 * The coordinator supplies the numbers on `job.waiting` -- it knows the
 * threshold, and a page that recomputed it would drift the moment it moved.
 */
function waitingNote(job) {
  const waiting = job.waiting;
  if (!waiting || job.status !== "pending") return null;

  if (!waiting.node_known) {
    return el("p", "job-note",
      "The machine this was queued to is no longer registered. It will be "
      + "moved to another one shortly.");
  }

  const machine = waiting.machine || "a machine on the network";
  const silent = waiting.silent_seconds;

  if (waiting.answering) {
    // Nothing is wrong; it is a queue. Say that plainly rather than leave a
    // blank that reads as a fault.
    return el("p", "job-note",
      `Queued on ${machine}, which is reporting in normally. It will start `
      + `when that machine finishes what it is doing.`);
  }

  const minutes = silent == null ? null : Math.round(silent / 60);
  const quiet = minutes == null ? "has stopped reporting in"
    : `has not reported in for ${minutes} minute${minutes === 1 ? "" : "s"}`;

  if (waiting.can_be_moved) {
    const left = silent == null ? null
      : Math.max(0, Math.round((waiting.moves_after_seconds - silent) / 60));
    return el("p", "job-note",
      `${machine} ${quiet}. `
      + (left ? `If it stays quiet the job moves to another machine in about ${left} minute${left === 1 ? "" : "s"}.`
              : "The job is being moved to another machine."));
  }

  // Chosen deliberately, so it is not silently swapped for something else.
  return el("p", "job-note",
    `${machine} ${quiet}. You picked this machine, so the job is not moved `
    + `automatically — cancel and send it again to let the coordinator choose.`);
}


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


/** Which rows were held back, and what that score is worth.
 *
 * "Data the node never saw" is true of both kinds of holdout and reassuring
 * about only one of them. A random slice of rows recorded over time means the
 * model was graded on Wednesday having memorised Tuesday and Thursday, which is
 * a far easier question than the one a submitter with a time series is asking.
 *
 * Measured here, on the same weights: 54.2% on a random holdout against a 50.7%
 * floor, and 51.7% against a 51.6% baseline when graded on the following two
 * years instead. The first number is the one the page used to show, on its own.
 */
function holdoutNote(job) {
  const kind = job.task_data?.holdout_kind;
  if (!kind) return null;

  if (kind === "time-ordered") {
    return el("p", "job-note",
      "Judged on the newest rows, because you said this data is in time order. "
      + "The model was trained on what came before them, so this score is the "
      + "question you actually asked.");
  }

  return el("p", "job-note",
    "Judged on a random slice of the rows, which is right when rows are "
    + "independent of each other. If yours were recorded over time — prices, "
    + "logs, sales — this flatters the model, because it was graded on days it "
    + "had both neighbours of. Tick “These rows are in time order” and send it "
    + "again to see the honest number.");
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
/** Hand a fetched response to the browser as a file. */
function saveResponse(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

/** The model packaged the way models are normally shipped. */
async function downloadBundle(job, button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Packaging…";

  try {
    const res = await fetch(`/my-tasks/${encodeURIComponent(job.task_id)}/bundle`, {
      headers: submitterHeaders(),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => null);
      throw new Error(detail?.detail?.detail || detail?.detail
                      || `Server returned ${res.status}`);
    }

    const name = (job.task_data?.model_name || "model")
      .replace(/[^A-Za-z0-9._-]+/g, "-");
    saveResponse(await res.blob(), `${name}.zip`);
    button.textContent = original;
  } catch (error) {
    console.error("Could not package the model:", error);
    button.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}


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

// What to actually do with the download once it is on disk.
function buildUsage(job) {
  const name = (job.task_data?.model_name || "model")
    .replace(/[^A-Za-z0-9._-]+/g, "-");

  const box = el("details", "job-usage");
  box.appendChild(el("summary", null, "How to use this model"));

  const body = el("div", "job-usage-body");
  body.appendChild(el("p", null,
    "The download is laid out the way models normally are — weights, a config "
    + "describing them, and the loader — so it runs without anything from "
    + "this project."));

  const contents = el("ul", "job-usage-list");
  [
    ["model.safetensors", "the weights. Loading it cannot run code, so it is "
      + "safe to pass to somebody else."],
    ["config.json", "the shape of the model, what its inputs are called, and "
      + "what its answers mean."],
    ["load_model.py", "rebuilds it and runs it. Needs numpy and torch."],
  ].forEach(([file, what]) => {
    const item = el("li");
    item.appendChild(el("code", null, file));
    item.appendChild(el("span", null, ` — ${what}`));
    contents.appendChild(item);
  });
  body.appendChild(contents);

  // The example is real rather than a placeholder now: the column names come
  // back with the model, so the number of values is known and the command can
  // be pasted. It still has to match what was trained -- --input is for a
  // classifier, and printing it under a language model told people to run a
  // command that model refuses.
  const architecture = job.task_data?.model_spec?.architecture || "mlp";
  const isClassifier = ["mlp", "feedforward"].includes(architecture);
  const columns = job.task_data?.dataset_info?.feature_names;
  const example = columns ? columns.map(() => "0").join(" ")
                          : "<one value per column>";

  body.appendChild(el("pre", "job-usage-code",
    `unzip ${name}.zip && cd ${name}\n`
    + `python load_model.py model.safetensors\n`
    + (isClassifier
        ? `python load_model.py model.safetensors --input ${example}`
        : 'python load_model.py model.safetensors --prompt "some text"')));

  body.appendChild(el("p", "job-usage-hint",
    "The first command prints what it reads, what it answers, and how it was "
    + "trained."));

  // safetensors is the format other tools read. ONNX is for the ones that are
  // not Python at all.
  body.appendChild(el("p", "job-field-hint",
    "To run it outside Python, convert it once — .onnx works almost anywhere:"));

  body.appendChild(el("pre", "job-usage-code",
    `python load_model.py model.safetensors --export ${name}.onnx`));

  box.appendChild(body);
  return box;
}

// --- comparing runs on the same data -------------------------------------
//
// Three runs sat in a list and the only way to tell whether the second was
// better than the first was to open both and read numbers off them. Worse, a
// run that changed two settings at once left no record of which one bought the
// improvement -- so the loop produced numbers without producing knowledge.
//
// Every re-run already records the job it came from. That is a chain, and a
// chain of (what changed, what it scored) is the thing worth looking at.

const TRACKED_SETTINGS = [
  ["hidden_dim", "hidden width"], ["depth", "layers"],
  ["d_model", "width"], ["n_head", "heads"], ["n_layer", "layers"],
  ["steps", "steps"], ["batch_size", "batch"], ["learning_rate", "rate"],
];

function settingsOf(job) {
  return { ...(job.task_data?.model_spec || {}),
           ...(job.task_data?.hyperparameters || {}) };
}

/** What this run changed from the one it came from, in words. */
function changesFrom(job, parent) {
  if (!parent) return "first run";

  const before = settingsOf(parent);
  const after = settingsOf(job);

  const changed = TRACKED_SETTINGS
    .filter(([key]) => key in after && before[key] !== after[key])
    .map(([key, label]) => `${label} ${before[key]} → ${after[key]}`);

  return changed.length ? changed.join(", ") : "same settings";
}

/** Runs descended from one original, oldest first. */
function seriesFor(job, byId) {
  const chain = [];
  const seen = new Set();

  let current = job;
  while (current && !seen.has(current.task_id)) {
    seen.add(current.task_id);
    chain.unshift(current);
    current = current.retry_of ? byId.get(current.retry_of) : null;
  }
  return chain;
}

function buildSeries(job, byId) {
  const chain = seriesFor(job, byId).filter(
    (run) => (run.verification?.measured || {}).learned_fraction != null
  );
  if (chain.length < 2) return null;

  const box = el("details", "job-series");
  box.open = true;
  box.appendChild(el("summary", null, `Runs on this data (${chain.length})`));

  const body = el("div", "job-series-body");
  const best = Math.max(...chain.map(
    (run) => run.verification.measured.learned_fraction));

  chain.forEach((run, index) => {
    const captured = run.verification.measured.learned_fraction;
    const row = el("div", "series-run");
    if (run.task_id === job.task_id) row.classList.add("is-current");

    row.appendChild(el("span", "series-change",
      changesFrom(run, index ? chain[index - 1] : null)));

    // A bar, because the point is which one is taller.
    const track = el("span", "series-bar");
    const fill = el("span",
      captured >= best ? "series-fill is-best" : "series-fill");
    fill.style.width = `${Math.max(2, captured * 100).toFixed(1)}%`;
    track.appendChild(fill);
    row.appendChild(track);

    row.appendChild(el("span", "series-score", `${(captured * 100).toFixed(1)}%`));
    body.appendChild(row);
  });

  body.appendChild(el("p", "job-field-hint",
    "How much of the possible improvement over guessing each run captured, "
    + "all scored on the same held-back rows."));

  box.appendChild(body);
  return box;
}


// Answer a spreadsheet with a spreadsheet.
//
// The person who uploads a CSV is a spreadsheet person. Handing them a weights
// file -- in any format -- hands them something they cannot open, and telling
// them to export it to ONNX needs the Python and PyTorch they do not have.
// Using the model has to be possible without leaving the page, so the model
// comes to the data instead of the other way round.
function buildScorer(job) {
  const info = job.task_data?.dataset_info || {};
  const columns = info.feature_names || [];

  const box = el("details", "job-prompt");
  box.appendChild(el("summary", null, "Use it on new rows"));

  box.addEventListener("toggle", () => {
    if (box.open) beginEditing(job.task_id, "score");
    else endEditing(job.task_id, "score");
  });

  const body = el("div", "job-prompt-body");
  body.appendChild(el("p", "job-field-hint",
    "Send a CSV of rows and get the same file back with the answer added. "
    + "No download, no Python."));

  if (columns.length) {
    const needed = el("p", "job-field-hint");
    needed.appendChild(el("strong", null, "Columns it reads: "));
    needed.appendChild(el("span", null, columns.join(", ")));
    needed.appendChild(el("span", null,
      " — matched by name, so the order does not matter and extra columns are "
      + "left alone. You can send the file you trained on."));
    body.appendChild(needed);
  }

  const file = document.createElement("input");
  file.type = "file";
  file.accept = ".csv,text/csv";
  body.appendChild(file);

  const status = el("div", "field-status");

  file.addEventListener("change", async () => {
    const chosen = file.files?.[0];
    if (!chosen) return;

    status.replaceChildren(el("span", null, `Reading ${chosen.name}…`));

    try {
      const res = await fetch(`/my-tasks/${encodeURIComponent(job.task_id)}/predict`, {
        method: "POST",
        headers: submitterHeaders({ "Content-Type": "text/csv" }),
        body: await chosen.text(),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail?.detail || detail?.detail
                        || `Server returned ${res.status}`);
      }

      const name = chosen.name.replace(/\.csv$/i, "");
      saveResponse(await res.blob(), `${name}-answered.csv`);
      status.replaceChildren(el("span", "success-message",
        "Downloaded. Every row has a predicted column and a confidence."));
    } catch (error) {
      console.error("Could not score the file:", error);
      status.replaceChildren(el("span", "error-message", error.message));
    } finally {
      file.value = "";
    }
  });

  body.appendChild(status);
  body.appendChild(el("p", "job-field-hint",
    "The rows are read on the coordinator to answer them, exactly as your "
    + "training data was: held in memory, never written down, gone when the "
    + "answer is sent."));

  box.appendChild(body);
  return box;
}


// Ask the model something of your own.
//
// A finished language model arrived as a grade and three continuations of
// prompts the node chose. What it does with *your* sentence is the question
// anyone actually has, and answering it meant downloading the weights,
// installing torch and running a script -- for a forward pass that takes two
// seconds on the machine already holding the file.
function buildPrompt(job) {
  const box = el("details", "job-prompt");
  box.appendChild(el("summary", null, "Ask it to write something"));

  // Open, the panel holds a half-typed prompt and an answer worth reading.
  // The list rebuilds every ten seconds and would take both away.
  box.addEventListener("toggle", () => {
    if (box.open) beginEditing(job.task_id, "prompt");
    else endEditing(job.task_id, "prompt");
  });

  const body = el("div", "job-prompt-body");
  body.appendChild(el("p", "job-field-hint",
    "Type the start of something and it will continue it, in whatever style "
    + "it picked up from your data."));

  const input = document.createElement("textarea");
  input.className = "job-prompt-input";
  input.rows = 2;
  input.placeholder = "The quick brown fox";
  input.maxLength = 500;
  body.appendChild(input);

  const ask = el("button", "btn-ghost", "Continue it");
  ask.type = "button";

  const output = el("p", "job-prompt-output");
  output.hidden = true;

  ask.addEventListener("click", async () => {
    const prompt = input.value.trim();
    if (!prompt) {
      output.hidden = false;
      output.replaceChildren(el("span", "error-message", "Type something first."));
      return;
    }

    ask.disabled = true;
    ask.textContent = "Writing…";
    output.hidden = false;
    output.replaceChildren(el("span", null, "…"));

    try {
      const res = await fetch(`/my-tasks/${encodeURIComponent(job.task_id)}/sample`, {
        method: "POST",
        headers: submitterHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ prompt, length: 200, temperature: 0.7 }),
      });
      const data = await res.json();
      if (!res.ok || data.status !== "success") {
        throw new Error(data?.detail?.detail || data?.detail || "The model could not be run.");
      }

      // The prompt in bold and the model's own words after it, so it is
      // obvious where one ends and the other begins.
      output.replaceChildren();
      output.appendChild(el("strong", null, data.prompt));
      output.appendChild(el("span", null, data.continuation));
    } catch (error) {
      console.error("Could not sample the model:", error);
      output.replaceChildren(el("span", "error-message", error.message));
    } finally {
      ask.disabled = false;
      ask.textContent = "Continue it";
    }
  });

  // Enter sends; Shift+Enter is a newline, since a prompt can be several lines.
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      ask.click();
    }
  });

  body.appendChild(ask);
  body.appendChild(output);
  box.appendChild(body);
  return box;
}


// Same data, different settings. Only the numbers worth changing between two
// runs -- the dataset is fixed here, and its shape already decided the rest.
/** "Data kept for another 3 days", or null when there is nothing to say. */
function dataCountdown(job) {
  if (!job.data_expires_at) return null;

  const expires = new Date(job.data_expires_at);
  if (Number.isNaN(expires.getTime())) return null;

  const left = expires.getTime() - Date.now();
  if (left <= 0) return null;              // the sweep has not caught up yet

  const line = el("p", "job-data-life");

  const minutes = Math.round(left / 60000);
  const when = minutes < 90
    ? `${minutes} min`
    : (minutes < 60 * 36
        ? `${Math.round(minutes / 60)} hours`
        : `${Math.round(minutes / (60 * 24))} days`);

  line.textContent = `Data kept for another ${when}.`;

  // The reason it is a week rather than an hour, offered where somebody is
  // looking at the clock running down and can still do something about it.
  if (job.data_kept_for === "60 minutes") {
    line.appendChild(document.createTextNode(" "));
    const why = el("span", "job-data-hint",
      "Signing in keeps it for days instead, so you can come back to this.");
    line.appendChild(why);
  }

  return line;
}

function buildTuner(job, onClose) {
  const spec = job.task_data?.model_spec || {};
  const hyper = job.task_data?.hyperparameters || {};
  const isText = !["mlp", "feedforward"].includes(spec.architecture || "mlp");

  const box = el("div", "job-tuner");
  box.appendChild(el("p", "job-field-hint",
    "Runs on the same data, scored against the same held-back rows — so the "
    + "difference in the result is your change, not a different split."));

  // Carry on from what this run already learned, rather than paying for it
  // twice. Only offered when there is a model to carry on from.
  let continueFrom = null;
  if (job.weights_id) {
    const wrap = el("label", "job-continue");
    continueFrom = document.createElement("input");
    continueFrom.type = "checkbox";
    wrap.appendChild(continueFrom);

    const words = el("span");
    words.appendChild(el("strong", null, "Carry on from this model"));
    words.appendChild(el("span", null,
      " — start where this run finished instead of from scratch. Keeps the "
      + "same shape, so only the training settings can change."));
    wrap.appendChild(words);
    box.appendChild(wrap);
  }

  const grid = el("div", "job-field-grid");
  const inputs = new Map();

  const fields = isText
    ? [["d_model", "Model width", spec.d_model], ["n_layer", "Layers", spec.n_layer]]
    : [["hidden_dim", "Hidden width", spec.hidden_dim], ["depth", "Hidden layers", spec.depth]];

  const shapeNames = fields.map(([name]) => name);

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

  // Greyed rather than hidden: seeing that the shape is fixed, and why, beats
  // watching two fields vanish.
  function applyContinueState() {
    const on = Boolean(continueFrom?.checked);
    shapeNames.forEach((name) => {
      const input = inputs.get(name);
      input.disabled = on;
      input.closest(".job-field").classList.toggle("is-locked", on);
    });
  }

  if (continueFrom) {
    continueFrom.addEventListener("change", applyContinueState);
    applyContinueState();
  }

  const status = el("div", "field-status");

  beginEditing(job.task_id, "tune");
  const done = () => {
    endEditing(job.task_id, "tune");
    if (onClose) onClose();
  };

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

    const carrying = Boolean(continueFrom?.checked);
    const changes = { model_spec: {}, hyperparameters: {} };
    if (carrying) changes.continue_from = true;

    // Sending the shape at all while carrying on would be refused, and
    // rightly: the weights only fit the model they came from.
    if (!carrying) {
      fields.forEach(([name]) => {
        const value = read(name);
        if (value !== undefined) changes.model_spec[name] = value;
      });
    }
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
        data.continued
          ? `Queued as ${data.task_id}, carrying on from this model.`
          : `Queued as ${data.task_id}. It will appear above when it finishes.`));
      (data.notes || []).forEach((text) =>
        status.appendChild(el("p", "field-advice", text)));

      // Let the panel stand long enough to read what it just said, then let
      // the list come back and show the new run.
      setTimeout(() => {
        endEditing(job.task_id, "tune");
        loadMyJobs();
      }, 4000);
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
    done();
    loadMyJobs();      // catch up on anything the pause missed
  });

  const row = el("div", "job-actions");
  row.appendChild(send);
  row.appendChild(cancel);
  box.appendChild(row);
  box.appendChild(status);

  return box;
}


function buildJobRow(job, byId) {
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

  const waiting = waitingNote(job);
  if (waiting) body.appendChild(waiting);

  if (job.result) body.appendChild(el("p", "job-result", job.result));

  const grading = strengthNote(job.verification);
  if (grading) body.appendChild(el("p", "job-grade", grading));

  // What the score was measured on. Shown next to it, because the number alone
  // means different things depending on the answer.
  if (grading) {
    const holdout = holdoutNote(job);
    if (holdout) body.appendChild(holdout);
  }

  const family = familyNote(job, byId);
  if (family) body.appendChild(family);

  // Where this run sits among the ones before it, and what was changed to get
  // here. Only worth drawing once there is something to compare against.
  const series = byId ? buildSeries(job, byId) : null;
  if (series) body.appendChild(series);

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

    // How long the file behind these buttons is still there.
    //
    // Without it the deletion is discovered by pressing "Adjust and run" and
    // being told the data went an hour after the job finished -- at which
    // point the only way forward is to upload the same file again. Saying it
    // beforehand turns a dead end into a deadline, and a deadline is
    // actionable.
    const until = dataCountdown(job);
    if (until) actions.appendChild(until);
  } else if (job.status === "completed") {
    // Its data has been deleted, so there is nothing to run it against.
    actions.appendChild(el("p", "job-note",
      "The data behind this job has been deleted, so it cannot be run again. "
      + "Send it as a new job with the file."));
  }

  if (actions.childElementCount) body.appendChild(actions);

  if (job.status === "completed" && job.weights_id) {
    const download = el("button", "btn", "Download model");
    download.type = "button";
    download.addEventListener("click", () => downloadBundle(job, download));
    body.appendChild(download);

    // Before the download, not after it: using the thing is the reason to
    // want the file at all, and for half of these people the file is no use.
    const architecture = job.task_data?.model_spec?.architecture || "mlp";
    if (["mlp", "feedforward"].includes(architecture)) {
      body.appendChild(buildScorer(job));
    } else {
      body.appendChild(buildPrompt(job));
    }

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

  // Somebody is filling in a form inside one of these rows. Rebuilding the
  // list would delete it out from under them.
  if (editingJobs.size) return;

  // Asking for the key here would mint one for a visitor who has never sent
  // anything, and the panel would claim they have jobs to look at.
  //
  // Signed in is the other way of being somebody. A laptop that has never sent
  // a job holds no key and has plenty to show, and this guard used to be what
  // stopped it: it returned "Nothing sent yet" without ever asking.
  if (!hasSubmitterKey() && !isSignedIn()) {
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

    const byId = new Map(jobs.map((job) => [job.task_id, job]));

    const list = el("div", "job-list");
    jobs.forEach((job) => list.appendChild(buildJobRow(job, byId)));
    container.replaceChildren(list);
  } catch (error) {
    console.error("Could not load your jobs:", error);
    container.replaceChildren(notice("Could not load your jobs", error.message));

    // null, not []: the panels listening to this draw very different things for
    // "you have no jobs" and "we could not find out". Without this the failure
    // never reached them at all, and a first fetch that failed left them
    // showing the "Loading…" their markup ships with -- forever, and through
    // every retry, since each one fails the same way.
    onJobs?.(null);
  }
}

export function startMyJobsPolling() {
  if (pollTimer) clearInterval(pollTimer);
  loadMyJobs();
  pollTimer = setInterval(loadMyJobs, POLL_INTERVAL_MS);
}
