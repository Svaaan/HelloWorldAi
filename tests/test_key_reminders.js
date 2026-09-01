// The reminders that say a key exists only in this browser.
//
// Both pages carried one, and both were wrong in the same way and one of them
// in a second way as well.
//
// They were filled amber blocks four or five lines deep, standing above a node
// that was plainly connected and running jobs, or above a workspace listing a
// finished, verified model. Nothing has gone wrong on either page. A warning at
// the volume of an error, on a page where nothing is wrong, every visit, is one
// people stop seeing -- so it takes the space of a real problem and gets read
// as decoration.
//
// The node one also did not work. Its button read "Save the key file →" and was
// an anchor to "/", so it saved nothing and dropped you on the front door --
// which, holding a node identity, offers "Connect to node". You press a button
// promising to save your key and end up somewhere else entirely. The download
// existed the whole time, inside the registration dialog, where it could only
// be reached at the one moment you are least likely to stop and use it.

const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const ROOT = path.join(__dirname, "..");
const JS = path.join(ROOT, "src/frontend/static/js");

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

const read = (p) => fs.readFileSync(path.join(JS, p), "utf8");

// --- the button has to do what it says -------------------------------------

check("saving the node key saves the node key", () => {
  const source = read("nodejs/nodeInfo.js");
  const warning = source.slice(source.indexOf("function keyBackupWarning"));
  const body = warning.slice(0, warning.indexOf("\n    }"));

  assert.ok(body.includes("downloadNodeKeyFile"),
    "the node page's reminder does not call the download");

  // The specific shape of the bug: an anchor pretending to be an action.
  assert.ok(!/\.href\s*=/.test(body),
    "the reminder navigates somewhere instead of saving. That is what it did "
    + "before: href = \"/\", labelled \"Save the key file →\".");
});

check("one implementation writes the node key file", () => {
  // It was private to the registration dialog, which is why the node page had
  // to offer a link instead of a download.
  const files = ["connect/nodeKeyFile.js", "connect/registerNodeModal.js",
                 "nodejs/nodeInfo.js"];
  const owning = files.filter((f) => read(f).includes("node_key_pair.json"));

  assert.deepStrictEqual(owning, ["connect/nodeKeyFile.js"],
    "the download should be built in exactly one place; found it in: "
    + owning.join(", "));
});

// --- and it has to be a reminder, not a wall -------------------------------

check("neither reminder is a block of amber", () => {
  for (const file of ["nodejs/nodeInfo.js", "workspace/identity.js"]) {
    const source = read(file);
    assert.ok(!source.includes("key-warning"),
      `${file} still builds the old block; it should use the one-line reminder`);
    assert.ok(source.includes("key-reminder"),
      `${file} does not render a reminder at all`);
  }
});

check("the reminder is one sentence, not a paragraph", () => {
  // Length is the whole point here: the previous version said everything it
  // could think of, which is why nobody read it.
  const source = read("workspace/identity.js");
  const start = source.indexOf("function renderBackupWarning");
  const body = source.slice(start, source.indexOf("\n}", start));

  const strings = [...body.matchAll(/"([^"\\]{10,})"/g)].map((m) => m[1]);
  const prose = strings.join(" ");

  assert.ok(prose.length < 200,
    `the workspace reminder is ${prose.length} characters of prose. It sits `
    + "beside a panel that already says \"Key loaded\", on a page listing jobs "
    + "the key has demonstrably run.");
});

// --- but it does not lie about being safe ----------------------------------

check("having jobs does not count as having saved the key", () => {
  // Tempting, and wrong: a workspace full of finished models is exactly when
  // losing the key costs the most. Only writing the file counts.
  const source = read("workspace/identity.js");
  const start = source.indexOf("export async function initIdentity");
  const body = source.slice(start);

  assert.ok(/renderBackupWarning\(known && !saved\)/.test(body),
    "the reminder should be shown whenever a key is held and not backed up, "
    + "regardless of what else the page knows");
  assert.ok(!/jobs|tasks|models/i.test(body.slice(0, body.indexOf("renderSettled"))),
    "initIdentity reads something about jobs; whether a key is saved is not "
    + "something jobs can answer");
});

check("saving is what removes it, on both pages", () => {
  const workspace = read("workspace/identity.js");
  assert.ok(/markBackedUp\(\);[\s\S]{0,400}renderBackupWarning\(false\)/
    .test(workspace),
    "saving the workspace key does not clear the reminder straight away");

  const node = read("nodejs/nodeInfo.js");
  const warning = node.slice(node.indexOf("function keyBackupWarning"));
  assert.ok(/if \(ok\)[\s\S]{0,200}remove\(\)/.test(warning),
    "the node reminder stays put after a successful save");
});

console.log();
if (failures.length) {
  console.log(`  ${failures.length} check(s) failed`);
  process.exit(1);
}
console.log(`  ${passed} checks passed`);
