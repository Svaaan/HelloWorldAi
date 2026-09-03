// A contributor with more than one graphics card, and what a data trainer sees.
//
// The compute side has handled this for a while: the node enumerates every
// device through NVML, poolPlanner splits a batch proportionally rather than
// evenly so a fast card is not held to the pace of a slow one, and the trainer
// has a property test showing uneven shards train identically to one big batch.
//
// The part nobody had exercised was the other end. Putting a four-card rig into
// the network -- an RTX 3070, two RTX 3060s and a GTX 1660 Super -- the node
// card on the send-work page read:
//
//   NVIDIA GeForce RTX 3070, NVIDIA GeForce RTX 3060, NVIDIA GeForce RTX 3060,
//   NVIDIA GeForce GTX 1660 Super
//
// which is mostly the word NVIDIA, and it pushed its own row 443 pixels past
// the edge of the card, dragging CPU and Cores out with it. Two separate
// problems: the text was not summarised, and the CSS could not truncate it
// because a flex item will not shrink below its content by default.
//
// Run with:  node tests/test_multi_gpu.js

import assert from "node:assert";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, "..");

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log(`  ok   ${name}`);
  } catch (error) {
    failures += 1;
    console.error(`  FAIL ${name}\n       ${error.message}`);
  }
}

function gpu(name) {
  return { name, total_memory: 8192, theoretical_tflops: 10 };
}

function node(names) {
  return { capabilities: { gpu: names.map(gpu) } };
}

const { describeGpus } = await import(
  path.join(ROOT, "src/frontend/static/js/distribution/fetchNode.js")
    .replace(/\\/g, "/")
    .replace(/^([A-Za-z]):/, "file:///$1:")
);

// --- summarising an array ------------------------------------------------

check("one card reads as its model, without the vendor boilerplate", () => {
  assert.equal(describeGpus(node(["NVIDIA GeForce RTX 3070"])), "RTX 3070");
});

check("identical cards are counted rather than repeated", () => {
  const text = describeGpus(node([
    "NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 3060",
  ]));
  assert.equal(text, "2× RTX 3060");
});

check("a mixed rig lists the commonest first", () => {
  const text = describeGpus(node([
    "NVIDIA GeForce RTX 3070",
    "NVIDIA GeForce RTX 3060",
    "NVIDIA GeForce RTX 3060",
    "NVIDIA GeForce GTX 1660 Super",
  ]));
  assert.equal(text, "2× RTX 3060, GTX 1660 Super, RTX 3070");
});

check("the summary is far shorter than the raw list", () => {
  const names = [
    "NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3060",
    "NVIDIA GeForce RTX 3060", "NVIDIA GeForce GTX 1660 Super",
  ];
  const raw = names.join(", ");
  const summarised = describeGpus(node(names));

  assert.ok(summarised.length < raw.length / 2,
    `summary is ${summarised.length} chars against ${raw.length} raw`);
});

check("a card from another vendor keeps its whole name", () => {
  // Only the words every NVIDIA card shares are dropped; nothing else is
  // assumed about how a device is named.
  assert.equal(describeGpus(node(["AMD Radeon RX 7900"])), "AMD Radeon RX 7900");
});

check("a machine with no GPU says so", () => {
  assert.equal(describeGpus({ capabilities: { gpu: [] } }), "None");
  assert.equal(describeGpus({ capabilities: {} }), "None");
});

check("an unnamed device does not render as undefined", () => {
  assert.equal(describeGpus({ capabilities: { gpu: [{}] } }), "Unknown GPU");
});

// --- and the row can actually shrink -------------------------------------

check("the value column is allowed to truncate", () => {
  // ellipsis without this does nothing: a flex item's default min-width is
  // auto, so it refuses to shrink below its content and overflows instead.
  const css = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/css/distribution.css"), "utf8");

  const start = css.indexOf(".node-spec > span:last-child");
  assert.ok(start !== -1, "the spec value rule is gone");
  const rule = css.slice(start, css.indexOf("}", start));

  assert.ok(/min-width:\s*0/.test(rule),
    "without min-width: 0 the text-overflow rule above it never applies");
  assert.ok(/text-overflow:\s*ellipsis/.test(rule));
});

// --- the four things a walkthrough turned up -----------------------------
//
// Appended here rather than in a file of their own: they are all "what the
// page says versus what is true", which is what this file is already about.

check("the data side produces the file its button promises", () => {
  // "Create key file" made a key and went straight to the workspace, so no
  // file appeared and the label was a small lie. The GPU side has always
  // stopped and insisted you download one first.
  const html = fs.readFileSync(
    path.join(ROOT, "src/frontend/template/start.html"), "utf8");
  const js = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/component/start.js"), "utf8");

  assert.ok(html.includes('id="builderKeyModal"'), "no confirmation modal");
  assert.ok(html.includes('id="builderKeyDownload"'), "no download button");
  assert.ok(js.includes("downloadKeyFile()"),
    "the button does not actually write a file");
  // and it must not slip past to the workspace without offering
  assert.ok(!/getSubmitterKey\(\);\s*window\.location\.href = "\/workspace"/.test(js),
    "still navigating away without offering the file");
});

check("the two row counts explain themselves", () => {
  // "Ready: 12 rows" and "Reads your 10 rows" a few centimetres apart, with
  // the holdout invisible, reads as a bug rather than as a split.
  const js = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/distribution/jobForm.js"), "utf8");

  assert.ok(js.includes("training rows"),
    "the second number should say which rows it counts");
  assert.ok(js.includes("held back to check the result"),
    "nothing accounts for the difference between the two numbers");
});

check("the privacy note mentions the one thing that carries a name", () => {
  // It says column names are not sent, which is true and which makes it easy
  // to assume nothing legible travels. The job's name does.
  const js = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/distribution/modalHandler.js"), "utf8");

  assert.ok(js.includes("name you give the job"),
    "the disclosure does not mention that the model name reaches the contributor");
});

check("a visitor with no node is not shown somebody else's GPU", () => {
  // "No node connected yet" sat beside a live RTX 3070 at 35 degrees and an
  // offer to accept work on it -- the local agent's readings, which are
  // neither theirs nor theirs to control.
  const js = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/nodejs/liveWork.js"), "utf8");

  const start = js.indexOf("export function startLiveWorkPolling");
  const body = js.slice(start, start + 700);
  assert.ok(/if \(!hasNode\(\)\) return;/.test(body),
    "live work polling is not gated on holding a node identity");
});

check("the front door is rendered for its deployment, not adjusted after", () => {
  // What each rendering actually contains is checked in
  // tests/test_front_door.py, which renders the template both ways. Reading the
  // raw template here cannot tell the two apart -- both branches are in the
  // file -- and a check that cannot fail is worse than none.
  //
  // What is worth pinning from this side: the page does not do it in the
  // browser. It did once, and the layout visibly moved a beat after the page
  // appeared.
  const html = fs.readFileSync(
    path.join(ROOT, "src/frontend/template/start.html"), "utf8");
  const js = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/component/start.js"), "utf8");

  assert.ok(/{%\s*if has_local_node\s*%}/.test(html),
    "the template should decide this, so the right markup arrives first paint");
  assert.ok(!js.includes('fetch("/local-node")'),
    "the front door should not re-arrange itself after load");

  // Every handler has to tolerate its element being absent, because on the
  // public rendering the register and connect buttons are simply not there.
  assert.ok(/if \(!document\.getElementById\("registerNodeButton"\)\) return;/
    .test(js), "setupContributor should bail out when the buttons are absent");
});

check("a missing node agent is explained where it gets in the way", () => {
  const errors = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/connect/nodeErrors.js"), "utf8");

  // The flag, not the wording. Matching the sentence means nobody can edit it.
  assert.ok(errors.includes("no_local_node"),
    "it should branch on the marker the proxy sets, not on message text");
  assert.ok(/href = "\/setup"/.test(errors),
    "the message should offer the guide as the next step");

  for (const file of ["registerNodeModal.js", "connectExistingNode.js"]) {
    const source = fs.readFileSync(
      path.join(ROOT, "src/frontend/static/js/connect", file), "utf8");

    assert.ok(source.includes("nodeCallError"),
      `${file} builds its errors by hand, so the marker never survives`);
    // The flag has to reach the call that renders, or the link never appears.
    assert.ok(/offerSetupGuide: err\.offerSetupGuide/.test(source),
      `${file} throws with the flag but drops it before showing the message`);
  }

  // Same dead end, different cause: an actual machine with no NVIDIA card.
  const gpu = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/connect/gpuDetect.js"), "utf8");
  assert.ok(/if \(!supported\)/.test(gpu) && gpu.includes('href = "/setup"'),
    "no usable card should point at the guide too");
});

check("the setup guide does not send you back to itself", () => {
  // The last step read "You are ready" over a green tick and linked "/". On the
  // central server "/" is the front door, which cannot register a node -- so it
  // offered the setup guide, which is the page you were just on.
  const html = fs.readFileSync(
    path.join(ROOT, "src/frontend/template/setup.html"), "utf8");

  const done = html.slice(html.indexOf('class="step step-done"'));
  assert.ok(!/href="\/"/.test(done),
    "the last step still points at the front door, which is where the loop was");
  assert.ok(done.includes("localhost:3000"),
    "it should point at the dashboard that starts beside the node");
  assert.ok(!done.includes("You are ready"),
    "nobody reading a guide has run anything yet");
});


// --- ending a session ----------------------------------------------------
//
// The key in this browser is the account. Until there was a way out, the only
// way to stop being somebody was to clear site data by hand -- which is also
// exactly how you lose a key you never saved.

check("sign-out forgets every key the app stores", () => {
  // The invariant worth guarding. A new key added anywhere in the app and not
  // listed here would survive a sign-out: the person believes they are gone,
  // and the browser still holds part of their identity.
  const header = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/component/header.js"), "utf8");

  const listed = new Set(
    [...header.matchAll(/"([a-zA-Z][a-zA-Z0-9]*)"/g)].map(m => m[1]));

  const jsDir = path.join(ROOT, "src/frontend/static/js");
  const written = new Set();
  const walk = (dir) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) { walk(full); continue; }
      if (!entry.name.endsWith(".js")) continue;
      const src = fs.readFileSync(full, "utf8");
      for (const m of src.matchAll(/localStorage\.setItem\(\s*"([^"]+)"/g)) {
        written.add(m[1]);
      }
      // keys held in a constant, e.g. KEY_STORAGE = "submitterKey"
      for (const m of src.matchAll(/const [A-Z_]+ = "([a-zA-Z][a-zA-Z0-9]*)";/g)) {
        if (/Key|KEY|Token|Id/.test(m[0])) written.add(m[1]);
      }
    }
  };
  walk(jsDir);

  const missed = [...written].filter(k => !listed.has(k));
  assert.deepEqual(missed, [],
    `these are stored but sign-out would leave them behind: ${missed.join(", ")}`);
});

check("signing out of one side leaves the other alone", () => {
  // Somebody can lend a graphics card and train their own models. Those are
  // two keys, and ending one session must not end the other.
  const header = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/component/header.js"), "utf8");

  const builder = header.slice(header.indexOf("builder: ["), header.indexOf("]", header.indexOf("builder: [")));
  const contributor = header.slice(header.indexOf("contributor: ["), header.indexOf("]", header.indexOf("contributor: [")));

  assert.ok(builder.includes("submitterKey"));
  assert.ok(!builder.includes("currentNodeId"), "the data side must not clear node keys");
  assert.ok(contributor.includes("currentNodeId"));
  assert.ok(!contributor.includes("submitterKey"), "the GPU side must not clear the submitter key");
});

check("holding both keys still gives a sign-out an answer", () => {
  // currentRole() returns null for somebody who is both, because it asks
  // "which one thing is this browser". The header asks a narrower question --
  // which side am I leaving from this page -- and that always has an answer.
  const header = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/component/header.js"), "utf8");

  const fn = header.slice(header.indexOf("function signedInAs"));
  assert.ok(/pageRole\(/.test(fn.slice(0, 400)),
    "signedInAs should prefer the role of the page you are on");
  assert.ok(!/return currentRole\(\);/.test(fn.slice(0, 400)),
    "currentRole is null when both keys are present, which hides the button");
});

check("it warns when the key has never been saved", () => {
  const html = fs.readFileSync(
    path.join(ROOT, "src/frontend/template/header.html"), "utf8");
  const header = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/component/header.js"), "utf8");

  assert.ok(html.includes('id="signOutUnsaved"'), "no unsaved warning at all");
  assert.ok(html.includes("no password reset"),
    "the warning should say the loss is permanent");
  assert.ok(header.includes("BACKED_UP_MARKER"),
    "the warning is not conditional on whether the key was saved");
});


// --- one way out ----------------------------------------------------------
//
// There were two. The header had a button that forgot the key -- irreversible,
// nobody can reissue one -- and the workspace had a "Sign out of GitHub" link
// that ended the session, which is reversible in ten seconds. Two controls on
// one page, almost the same words, wildly different consequences, and neither
// saying which was which.

check("there is exactly one sign-out control in the app", () => {
  const files = [
    "src/frontend/template/header.html",
    "src/frontend/template/workspace.html",
    "src/frontend/template/start.html",
    "src/frontend/template/node.html",
    "src/frontend/template/distribution.html",
  ];

  let controls = 0;
  for (const rel of files) {
    const html = fs.readFileSync(path.join(ROOT, rel), "utf8");
    controls += (html.match(/id="signOutButton"/g) || []).length;
  }

  assert.equal(controls, 1,
    controls + " sign-out controls across the pages; there must be one, and "
    + "its dialog must say what it is about to forget");

  const js = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/workspace/account.js"), "utf8");
  const code = js.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*/g, "");
  assert.ok(!/sign-?out/i.test(code),
    "the workspace account module is offering a sign-out again");
});

check("the dialog names both things it ends, separately", () => {
  const header = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/component/header.js"), "utf8");

  assert.ok(header.includes("signOutList"),
    "the dialog should itemise rather than summarise: one of the two is "
    + "permanent and the other is not");
  assert.ok(header.includes("/auth/sign-out"),
    "confirming must end the GitHub session too, or the single control only "
    + "does half of what it says");
  assert.ok(header.includes("signout-permanent"),
    "the irreversible item should be marked as such");
});

check("the key panel stops repeating itself once the key is saved", () => {
  // "Fingerprint 420a4d98 - saved to a file" stood there permanently, under a
  // pill already reading "Key loaded", with no warning outstanding.
  const identity = fs.readFileSync(
    path.join(ROOT, "src/frontend/static/js/workspace/identity.js"), "utf8");

  assert.ok(!identity.includes("saved to a file"),
    "the detail line is announcing a saved key again");
  assert.ok(identity.includes("identityFingerprint"),
    "the fingerprint should still be reachable, under the manage toggle");
});

console.log(failures ? `\n  ${failures} failed` : "\n  all checks passed");
process.exit(failures ? 1 : 0);
