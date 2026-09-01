// The workspace's live view of the machine holding your data.
//
// A submitter could be told their job was "Running" and nothing else. You hand
// your data to a stranger's graphics card and watch a word sit there, with no
// way to tell working from stuck. The contributor lending the card has had the
// full picture on their node page the whole time, and /nodes publishes it to
// anybody who asks -- it was simply never shown to the person whose data was on
// the machine.
//
// These check the parts that turn node readings into something readable,
// because that is where this kind of panel goes wrong: a temperature of
// undefined, a duration of -3 minutes, a memory figure off by 1024.

const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.join(__dirname, "..");
const SOURCE = path.join(
  ROOT, "src/frontend/static/js/workspace/running.js");

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

// The module is an ES module using browser globals. Strip the exports and run
// the pure helpers in a plain context -- no DOM needed for the parts that only
// do arithmetic.
function load() {
  const source = fs.readFileSync(SOURCE, "utf8")
    .replace(/^export /gm, "");
  const context = { module: {}, fetch: () => {}, document: undefined };
  vm.createContext(context);
  vm.runInContext(
    source + "\n;({ elapsedSince, firstGpu, readings });", context);
  return vm.runInContext("({ elapsedSince, firstGpu, readings })", context);
}

const { elapsedSince, firstGpu, readings } = load();

// The module runs in its own realm, so the arrays it returns have that realm's
// Array prototype and deepStrictEqual rejects them as not reference-equal even
// when every value matches. Compare by value.
function sameRows(actual, expected, message) {
  assert.strictEqual(JSON.stringify(actual), JSON.stringify(expected), message);
}

// --- how long it has been running -----------------------------------------

check("a job that just started does not read as zero", () => {
  const now = Date.parse("2026-09-01T12:00:00Z");
  assert.strictEqual(
    elapsedSince("2026-09-01T11:59:40Z", now), "less than a minute");
});

check("minutes and hours are spelled the way people say them", () => {
  const now = Date.parse("2026-09-01T12:00:00Z");
  assert.strictEqual(elapsedSince("2026-09-01T11:59:00Z", now), "1 minute");
  assert.strictEqual(elapsedSince("2026-09-01T11:30:00Z", now), "30 minutes");
  assert.strictEqual(elapsedSince("2026-09-01T11:00:00Z", now), "1 hour");
  assert.strictEqual(elapsedSince("2026-09-01T09:00:00Z", now), "3 hours");
  assert.strictEqual(elapsedSince("2026-09-01T10:25:00Z", now), "1h 35m");
});

check("a clock ahead of the server does not run time backwards", () => {
  // The browser's clock is not the coordinator's. A machine a few seconds fast
  // would otherwise show "Running for -1 minutes".
  const now = Date.parse("2026-09-01T12:00:00Z");
  assert.strictEqual(
    elapsedSince("2026-09-01T12:00:30Z", now), "less than a minute");
});

check("a missing or unparseable start time says nothing at all", () => {
  assert.strictEqual(elapsedSince(null), null);
  assert.strictEqual(elapsedSince(undefined), null);
  assert.strictEqual(elapsedSince("not a date"), null);
});

// --- reading the machine ---------------------------------------------------

check("the card is found whether capabilities hold a list or one object", () => {
  assert.strictEqual(
    firstGpu({ capabilities: { gpu: [{ name: "RTX 3070" }] } }).name,
    "RTX 3070");
  assert.strictEqual(
    firstGpu({ capabilities: { gpu: { name: "RTX 3070" } } }).name,
    "RTX 3070");
});

check("a node with no capabilities does not throw", () => {
  assert.strictEqual(firstGpu(undefined), null);
  assert.strictEqual(firstGpu({}), null);
  assert.strictEqual(firstGpu({ capabilities: {} }), null);
  assert.strictEqual(firstGpu({ capabilities: { gpu: [] } }), null);
});

check("memory is shown as used out of total, in GB", () => {
  // The node reports megabytes. Printing those raw gives "6170 / 8192", which
  // is a number nobody holds in their head against a card they know as 8 GB.
  const rows = readings({
    total_memory: 8192, free_memory: 6170,
    load_percentage: 49, temperature: 39,
  });

  sameRows(rows, [
    ["Load", "49%"],
    ["Temperature", "39 °C"],
    ["Memory", "2.0 / 8.0 GB"],
  ]);
});

check("a reading the node did not send is left out, not rendered empty", () => {
  // Not every field is present on every node record, and "Temperature: °C" or
  // "Load: undefined%" reads as broken rather than as absent.
  sameRows(readings({ load_percentage: 12 }), [["Load", "12%"]]);
  sameRows(readings({}), []);
  sameRows(readings(null), []);
});

check("zero readings are shown, not treated as missing", () => {
  // An idle-but-claimed card legitimately reports 0% load, and 0 is falsy.
  const rows = readings({ load_percentage: 0, temperature: 0 });
  sameRows(rows, [["Load", "0%"], ["Temperature", "0 °C"]]);
});

check("free memory larger than total does not print a negative", () => {
  const rows = readings({ total_memory: 8192, free_memory: 9000 });
  sameRows(rows, [["Memory", "0.0 / 8.0 GB"]]);
});

// --- what it refuses to claim ---------------------------------------------

check("it does not invent training progress", () => {
  // The node tracks epochs and loss, and tells only its own dashboard. None of
  // it reaches the coordinator, so the workspace cannot honestly show it.
  // Deriving a percentage from elapsed time would be a guess wearing the
  // clothes of a measurement.
  const source = fs.readFileSync(SOURCE, "utf8");
  const code = source.replace(/\/\/.*$/gm, "").replace(/\/\*[\s\S]*?\*\//g, "");

  for (const invented of ["epoch", "progress", "percentComplete", "eta"]) {
    assert.ok(!new RegExp("\\b" + invented + "\\b", "i").test(code),
      `running.js refers to ${invented}, which the coordinator does not know`);
  }
});

console.log();
if (failures.length) {
  console.log(`  ${failures.length} check(s) failed`);
  process.exit(1);
}
console.log(`  ${passed} checks passed`);
