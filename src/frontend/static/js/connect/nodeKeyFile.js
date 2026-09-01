// Writing a node's keypair to a file, and remembering that it happened.
//
// This lived inside the registration dialog, which meant the only place that
// could actually save the key was the one screen you see once, at the moment
// you are least likely to stop and think about it. The node page, which is
// where somebody goes afterwards and where the "your key exists only in this
// browser" warning lives, had no way to offer the download at all -- so its
// button was an anchor to "/" labelled "Save the key file →", which saved
// nothing and dropped you back on the front door.
//
// One implementation, importable from both.

const BACKED_UP_KEY = "nodeKeyBackedUp";
const PRIVATE_KEY = "nodePrivateKey";
const PUBLIC_KEY = "nodePublicKeyBase64";

/** Whether the keypair in this browser has been written to a file from here.
 *
 * Stored beside the key, so clearing site data clears this too -- which is
 * correct. A restored browser with no record of a backup should ask again
 * rather than assume.
 */
export function isNodeKeyBackedUp() {
  try {
    const stored = localStorage.getItem(BACKED_UP_KEY);
    const publicKey = localStorage.getItem(PUBLIC_KEY);
    return Boolean(stored) && Boolean(publicKey)
      && stored === publicKey.slice(0, 12);
  } catch {
    return false;                    // keep asking rather than assume
  }
}

function markBackedUp(publicKeyBase64) {
  try {
    localStorage.setItem(BACKED_UP_KEY, publicKeyBase64.slice(0, 12));
  } catch {
    // A browser refusing storage will simply keep asking, which is the safe
    // direction to fail in.
  }
}

/**
 * Save this browser's node keypair to a file.
 *
 * @returns {{ok: boolean, message: string}} what to tell the person, so the
 *   caller can put it wherever that page shows things.
 */
export function downloadNodeKeyFile() {
  let privateKeyRaw;
  let publicKeyBase64;
  try {
    privateKeyRaw = localStorage.getItem(PRIVATE_KEY);
    publicKeyBase64 = localStorage.getItem(PUBLIC_KEY);
  } catch {
    return { ok: false, message: "This browser will not let the page read its stored key." };
  }

  if (!privateKeyRaw || !publicKeyBase64) {
    return {
      ok: false,
      message: "No node key in this browser. Load your key file to take this node back.",
    };
  }

  let privateKeyJwk;
  try {
    privateKeyJwk = JSON.parse(privateKeyRaw);
  } catch {
    return { ok: false, message: "The stored key is unreadable." };
  }

  // Exported without key_ops, older keys come back unusable for signing.
  if (!privateKeyJwk.key_ops) privateKeyJwk.key_ops = ["sign"];

  const blob = new Blob(
    [JSON.stringify({ privateKey: privateKeyJwk, publicKeyBase64 }, null, 2)],
    { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = "node_key_pair.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);

  markBackedUp(publicKeyBase64);

  return { ok: true, message: "Key file saved. Keep it somewhere you will find it again." };
}
