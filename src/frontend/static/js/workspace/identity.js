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
  renderBackupWarning(false);    // the moment it is saved, not on next load
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

  const box = el("div", "key-warning");
  box.appendChild(el("strong", null, "This key exists only in this browser."));
  box.appendChild(el("p", null,
    "Clearing site data, a private window, or a different computer will lose "
    + "it — and with it every job and model on this page. There is no reset: "
    + "the coordinator stores a one-way digest of your key and cannot give it "
    + "back. Save it to a file now."));
  host.appendChild(box);
}


// --- wiring ---------------------------------------------------------------

export async function initIdentity({ onChange } = {}) {
  const known = hasSubmitterKey();
  const state = document.getElementById("identityState");
  const detail = document.getElementById("identityDetail");

  if (state) {
    state.textContent = known ? "Key loaded" : "No key yet";
    state.className = known ? "status-pill status-online" : "status-pill";
  }

  if (detail) {
    detail.textContent = known
      ? `Fingerprint ${await fingerprint(getSubmitterKey())}`
      : "A key is created the first time you send a job.";
  }

  const save = document.getElementById("saveKeyButton");
  if (save) {
    // Minting a key just to save it would hand out an identity to someone who
    // has never sent anything, which is worse than an unavailable button.
    save.disabled = !known;
    save.addEventListener("click", downloadKeyFile);
  }

  renderBackupWarning(known && !isBackedUp());

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
