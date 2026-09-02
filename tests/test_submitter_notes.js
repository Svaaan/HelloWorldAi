// What the workspace tells a submitter while they wait, and about their score.
//
// Three notes were added after running a real workload through this service --
// a stock-signal project that uploaded data, waited, and collected models. Each
// exists because of something that was confusing or misleading the first time
// somebody did that for real:
//
//   waitingNote   a job queued to a switched-off machine is moved after fifteen
//                 minutes, which works. Nothing said so, and a quarter of an
//                 hour of the word "Queued" reads as a broken service.
//
//   holdoutNote   the verification score comes from a random slice of rows.
//                 For a time series that is a much easier question than the one
//                 being asked: measured here, the same weights scored 54.2% on
//                 a random holdout and 51.7% graded on the following two years.
//                 The page showed the first number with an encouraging sentence.
//
//   familyNote    the loop is change one thing, run again, did it help. The
//                 third part needed two scores next to each other.
//
// These check which message appears for which state, because that is the whole
// of what they do. The DOM is a shim: the functions build nodes rather than
// markup, and only the text and the branching matter here.

const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.join(__dirname, "..");
const SOURCE = path.join(ROOT, "src/frontend/static/js/distribution/myJobs.js");

let passed = 0;
const failures = [];

function check(name, fn) {
  try {
    fn();
    console.log("  ok   " + name);
    passed++;
  } catch (error) {
    console.log("  FAIL " + name);
    console.log("       " + error.message);
    failures.push(name);
  }
}

/** Enough of a DOM for functions that only build and fill elements. */
function fakeDocument() {
  const make = (tag) => ({
    tag,
    className: "",
    textContent: "",
    children: [],
    appendChild(child) { this.children.push(child); return child; },
    get text() {
      return [this.textContent, ...this.children.map((c) => c.text)].join("");
    },
  });
  return { createElement: make };
}

/** Load only the helpers, without the module's fetches and timers. */
function load() {
  const source = fs.readFileSync(SOURCE, "utf8");

  // Take the three functions and the small el() they use. Running the whole
  // module would start polling and reach for a page that is not here.
  const wanted = ["function el(", "function waitingNote(", "function holdoutNote(",
                  "function familyNote("];
  const pieces = wanted.map((marker) => {
    const start = source.indexOf(marker);
    if (start < 0) throw new Error(`${marker} is missing from myJobs.js`);
    // Functions in this file end at a closing brace in column 0.
    const end = source.indexOf("\n}", start);
    return source.slice(start, end + 2);
  });

  const context = { document: fakeDocument() };
  vm.createContext(context);
  vm.runInContext(pieces.join("\n\n"), context);
  return vm.runInContext("({ waitingNote, holdoutNote, familyNote })", context);
}

const { waitingNote, holdoutNote, familyNote } = load();

const textOf = (node) => (node ? node.text : null);

// --- waiting ---------------------------------------------------------------

check("a job on a healthy machine says it is simply queued", () => {
  const note = waitingNote({
    status: "pending",
    waiting: { node_known: true, machine: "RTX 3070", silent_seconds: 12,
               answering: true, moves_after_seconds: 900, can_be_moved: true },
  });

  const text = textOf(note);
  assert.match(text, /RTX 3070/);
  assert.match(text, /reporting in normally/);
  // It must not imply a problem: there is a queue, and that is all.
  assert.doesNotMatch(text, /stopped|has not reported/);
});

check("a job on a silent machine says when it will be moved", () => {
  const note = waitingNote({
    status: "pending",
    waiting: { node_known: true, machine: "RTX 3070", silent_seconds: 600,
               answering: false, moves_after_seconds: 900, can_be_moved: true },
  });

  const text = textOf(note);
  assert.match(text, /has not reported in for 10 minutes/);
  assert.match(text, /moves to another machine in about 5 minutes/,
    "the submitter should know the wait ends, and roughly when");
});

check("a machine the submitter chose is not silently swapped", () => {
  const note = waitingNote({
    status: "pending",
    waiting: { node_known: true, machine: "RTX 4090", silent_seconds: 900,
               answering: false, moves_after_seconds: 900, can_be_moved: false },
  });

  const text = textOf(note);
  assert.match(text, /You picked this machine/);
  assert.match(text, /cancel and send it again/,
    "if it will not be moved, say what to do instead");
});

check("nothing is said about a job that is not waiting", () => {
  assert.strictEqual(waitingNote({ status: "running", waiting: {} }), null);
  assert.strictEqual(waitingNote({ status: "completed" }), null);
  assert.strictEqual(waitingNote({ status: "pending" }), null);
});

// --- what the score was measured on ---------------------------------------

check("a random holdout warns that a time series would be flattered", () => {
  const note = holdoutNote({ task_data: { holdout_kind: "random" } });
  const text = textOf(note);

  assert.match(text, /random slice/);
  assert.match(text, /flatters the model/,
    "the caveat is the point; without it the number reads as skill");
  assert.match(text, /time order/,
    "it should name the checkbox that fixes it");
});

check("a time-ordered holdout says the score is the real question", () => {
  const note = holdoutNote({ task_data: { holdout_kind: "time-ordered" } });
  const text = textOf(note);

  assert.match(text, /newest rows/);
  assert.doesNotMatch(text, /flatters/,
    "there is nothing to warn about once the split respects the order");
});

check("nothing is claimed when the split is unknown", () => {
  assert.strictEqual(holdoutNote({ task_data: {} }), null);
  assert.strictEqual(holdoutNote({}), null);
});

// --- comparing runs --------------------------------------------------------

function scored(taskId, datasetId, steps, accuracy) {
  return {
    task_id: taskId,
    dataset_id: datasetId,
    submitted_at: `2026-09-0${steps === 200 ? 1 : 2}T10:00:00`,
    task_data: { model_name: `run-${steps}`, hyperparameters: { steps } },
    verification: { measured: { holdout_accuracy: accuracy, learned_fraction: 0.5 } },
  };
}

check("one run alone has nothing to compare against", () => {
  const only = scored("t1", "d1", 200, 0.6);
  const byId = new Map([["t1", only]]);
  assert.strictEqual(familyNote(only, byId), null);
});

check("runs on the same data are listed together, best marked", () => {
  const first = scored("t1", "d1", 200, 0.61);
  const second = scored("t2", "d1", 4000, 0.72);
  const byId = new Map([["t1", first], ["t2", second]]);

  const text = textOf(familyNote(second, byId));

  assert.match(text, /run-200/);
  assert.match(text, /run-4000/);
  assert.match(text, /61\.0%/);
  assert.match(text, /72\.0%/);
  assert.match(text, /best/, "the answer to 'did that help' should be marked");
  assert.match(text, /this one/, "the row you are looking at should be findable");
});

check("a run on different data is not mixed in", () => {
  const mine = scored("t1", "d1", 200, 0.61);
  const other = scored("t2", "d2", 4000, 0.99);
  const byId = new Map([["t1", mine], ["t2", other]]);

  // Only one scored run on d1, so there is nothing to compare -- and the 99%
  // from a different dataset must not appear beside it.
  assert.strictEqual(familyNote(mine, byId), null);
});

check("a run with no score yet does not join the comparison", () => {
  const done = scored("t1", "d1", 200, 0.61);
  const running = { task_id: "t2", dataset_id: "d1", task_data: {}, verification: {} };
  const byId = new Map([["t1", done], ["t2", running]]);

  assert.strictEqual(familyNote(done, byId), null,
    "a job still training has nothing to contribute to a comparison");
});

console.log();
if (failures.length) {
  console.log(`  ${failures.length} check(s) failed`);
  process.exit(1);
}
console.log(`  ${passed} checks passed`);
