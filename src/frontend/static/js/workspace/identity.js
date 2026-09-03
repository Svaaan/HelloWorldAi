// The AI builder's identity: hold it, save it, carry it to another machine.
//
// A submitter key is generated in the browser and kept in localStorage. That is
// enough to own a job, but on its own it is trapped: switch browsers, clear
// site data, or move to a laptop and the jobs and trained models become
// unreachable, with nothing to recover them from.
//
// So the key gets the same treatment a node's keypair already gets on /connect
// -- it can be written to a file and loaded back somewhere else. That keeps one
// story for the whole product ("your key is the thing you keep") instead of
// bolting accounts onto one half of it.

import {
  getSubmitterKey,
  hasSubmitterKey,
  isSignedIn,
  setSubmitterKey,
} from "../distribution/submitter.js";

const FILE_TYPE = "helloworldai-builder-key";
const FILE_VERSION = 1;

// Whether this key has ever been written to a file.
//
// The key lives in localStorage, which is the same store a browser empties
// when someone clears site data, uses a private window, or switches machines.
// Nothing said so, and there is no recovery: the coordinator keeps only a
// one-way digest, so a lost key is a lost workspace and every model in it.
// A quiet "Save key to a file" button next to that is not a warning.
//
// The flag is stored beside the key, so clearing site data clears it too --
// which is correct. A restored browser with no record of a backup should ask
// again rather than assume.
const BACKED_UP_KEY = "submitterKeyBackedUp";

function markBackedUp() {
  try {
    localStorage.setItem(BACKED_UP_KEY, getSubmitterKey().slice(0, 8));
  } catch {
    // A browser that refuses storage will simply keep asking, which is the
    // safe direction to fail in.
  }
}

/** Whether the key currently held has been written to a file from here. */
function isBackedUp() {
  try {
    const stored = localStorage.getItem(BACKED_UP_KEY);
    return Boolean(stored) && hasSubmitterKey()
      && stored === getSubmitterKey().slice(0, 8);
  } catch {
    return false;
  }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function setStatus(message, kind) {
  const target = document.getElementById("identityStatus");
  if (!target) return;
  target.replaceChildren();
  if (!message) return;
  target.appendChild(el("span", kind ? `${kind}-message` : null, message));
}

/** A short, non-secret fingerprint so two keys can be told apart on screen. */
async function fingerprint(key) {
  const bytes = new TextEncoder().encode(key);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].slice(0, 4)
    .map((b) => b.toString(16).padStart(2, "0")).join("");
}

// --- saving ---------------------------------------------------------------

export function downloadKeyFile() {
  const payload = {
    type: FILE_TYPE,
    version: FILE_VERSION,
    submitterKey: getSubmitterKey(),
    created: new Date().toISOString(),
    note: "Anyone holding this file can see your jobs and download your models.",
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = "builder_key.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);

  markBackedUp();
  // Both the moment it is saved, not on the next page load: the panel has just
  // been used, and leaving it looking unfinished is what this fixes.
  renderBackupWarning(false);
  renderSettled(true);
  setStatus("Key saved. Keep it somewhere safe — it is the only way back to your jobs.", "success");
}

// --- loading --------------------------------------------------------------

async function loadKeyFile(file) {
  const text = await file.text();

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error("That file is not valid JSON.");
  }

  // Accept a bare key too: someone may have copied just the string out.
  const key = typeof parsed === "string" ? parsed : parsed.submitterKey;

  // Someone with a node and a workspace has two key files, and this is the
  // likeliest mix-up. The node file carries no `type`, so recognise it by its
  // shape rather than reporting a vague "no key in that file".
  if (!key && parsed?.privateKey && parsed?.publicKey) {
    throw new Error(
      "That is your node key file, which belongs on the Connect page. " +
      "The builder key is a different file."
    );
  }

  if (parsed.type && parsed.type !== FILE_TYPE) {
    throw new Error("That file is not a builder key.");
  }
  if (!key) throw new Error("No builder key in that file.");

  setSubmitterKey(key);          // throws if it is malformed
  markBackedUp();                // it demonstrably exists in a file
  return key;
}

// --- the one copy problem -------------------------------------------------

function renderBackupWarning(needed) {
  const host = document.getElementById("identityWarning");
  if (!host) return;

  host.replaceChildren();
  host.hidden = !needed;
  if (!needed) return;

  // One line, not a block.
  //
  // This was five lines of amber sitting above a panel that already says
  // "Key loaded", on a page listing jobs the key has demonstrably run. It is
  // true and it is worth saying, and it was saying it at the volume of an
  // error on a page where nothing has gone wrong -- so it reads as noise and
  // gets skipped, which is the opposite of what a warning is for.
  //
  // Having jobs is not the same as having saved the key, so this does not go
  // away on its own. Saving the key removes it, because that is the thing it
  // is asking for. Nothing else does.
  const line = el("p", "key-reminder");
  line.appendChild(document.createTextNode(
    "This key is only in this browser, and it is the only way back to these "
    + "jobs. "));

  const save = el("button", "link-button", "Save it to a file");
  save.type = "button";
  save.addEventListener("click", downloadKeyFile);
  line.appendChild(save);

  host.appendChild(line);
}


// --- settling down --------------------------------------------------------

/** Keep the panel to its fingerprint, with everything else behind a toggle.
 *
 * The panel used to be a "Save key to a file" button as the loudest thing on
 * the page, under a paragraph explaining why you should press it -- standing
 * there whether or not you had, and often directly after the dialog on the
 * front door that offered the very same download a moment before.
 *
 * That was the whole panel, in both states, forever. Now it is compact in both
 * and the difference is one line: unsaved, a reminder sits above it with a link
 * that saves. Saving another copy and loading a different key are still real
 * things to want, so they are one click away rather than gone.
 *
 * `settled` is kept as a parameter because the caller knows the state and this
 * function does not need to; today both states render the same way.
 */
function renderSettled(settled) {
  const actions = document.getElementById("identityActions");
  const note = document.getElementById("identityNote");
  if (!actions) return;

  if (note) note.hidden = true;
  actions.hidden = true;

  let toggle = document.getElementById("identityManage");
  if (toggle) return;                       // already there from a prior render

  toggle = el("button", "link-button", "Save a copy, or load a different key");
  toggle.type = "button";
  toggle.id = "identityManage";
  toggle.addEventListener("click", () => {
    const hidden = actions.hidden;
    actions.hidden = !hidden;
    if (note) note.hidden = !hidden;
    toggle.textContent = hidden
      ? "Hide key options"
      : "Save a copy, or load a different key";
  });

  actions.parentNode.insertBefore(toggle, actions);
}


// --- wiring ---------------------------------------------------------------

/** The pill and the line under it: what this browser is, right now.
 *
 * Separate from initIdentity because being signed in is decided by a request
 * that lands after the panel has been drawn, and the answer changes what these
 * two lines should say. Redrawing them is safe; re-running initIdentity is not,
 * because it would bind the save button's click handler a second time.
 */
async function renderState() {
  const known = hasSubmitterKey();
  const signedIn = isSignedIn();
  const state = document.getElementById("identityState");
  const detail = document.getElementById("identityDetail");

  if (state) {
    // "No key yet" is literally true of a signed-in browser holding no key,
    // and it read as a fault: the panel below it says the account has keys
    // linked and can see the work, so the two lines appeared to contradict
    // each other. Nothing is missing in that case -- the account is the way in.
    state.textContent = known
      ? "Key loaded"
      : (signedIn ? "Using your account" : "No key yet");
    state.className = known || signedIn
      ? "status-pill status-online"
      : "status-pill";
  }

  const saved = known && isBackedUp();

  if (detail) {
    if (!known) {
      detail.textContent = signedIn
        // And no key is created by sending one, either: signed in, work is
        // filed under the account. Saying otherwise would promise a key file
        // that never appears.
        ? "No key in this browser. Your work is reached through your account."
        : "A key is created the first time you send a job.";
    } else {
      // Saying it is saved is the point of the whole panel, so it goes in the
      // line people actually read rather than only in the absence of a warning.
      detail.textContent = `Fingerprint ${await fingerprint(getSubmitterKey())}`
        + (saved ? " · saved to a file" : "");
    }
  }

  return { known, saved };
}

let watchingIdentity = false;

export async function initIdentity({ onChange } = {}) {
  const { known, saved } = await renderState();

  if (!watchingIdentity) {
    watchingIdentity = true;
    document.addEventListener("hw:identity-changed", () => { renderState(); });
  }

  const save = document.getElementById("saveKeyButton");
  if (save) {
    // Minting a key just to save it would hand out an identity to someone who
    // has never sent anything, which is worse than an unavailable button.
    save.disabled = !known;
    save.addEventListener("click", downloadKeyFile);
  }

  renderBackupWarning(known && !saved);
  renderSettled(saved);            // compact either way; see above

  const fileInput = document.getElementById("keyFileInput");
  if (fileInput) {
    fileInput.addEventListener("change", async () => {
      const file = fileInput.files?.[0];
      if (!file) return;

      try {
        await loadKeyFile(file);
        setStatus("Key loaded. Your jobs are below.", "success");
        fileInput.value = "";
        await initIdentity({ onChange });     // refresh the header state
        onChange?.();
      } catch (error) {
        console.error("Could not load the builder key:", error);
        setStatus(error.message, "error");
        fileInput.value = "";
      }
    });
  }
}
