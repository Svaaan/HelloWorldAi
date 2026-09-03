// Identity for the person sending work, without an account.
//
// A contributor's node proves itself with a keypair. Whoever supplies the data
// had nothing: a submitted job recorded only an IP, so there was no way to ask
// "which jobs are mine" and no way to decide who may download a trained model.
// The pipeline trained a model and then had nowhere to hand it back.
//
// This browser generates one random secret and keeps it. The coordinator stores
// only its SHA-256 digest, so reading the database does not let anyone claim
// someone else's jobs.
//
// It is a bearer credential: whoever holds it owns those jobs. Lose it and the
// jobs become unreachable -- the honest trade for having no account to recover.

const KEY_STORAGE = "submitterKey";
const KEY_BYTES = 32;

let cached = null;

// Whether this browser is signed in to an account. Set by account.js once
// /auth/me has answered; false until then, which is the safe direction --
// it means a key, and a key has always worked.
let signedIn = false;

/** Told by account.js. Nothing else should call this. */
export function setSignedIn(value) {
  signedIn = Boolean(value);
}

/** Whether an account is standing behind this browser. */
export function isSignedIn() {
  return signedIn;
}

function generateKey() {
  const bytes = new Uint8Array(KEY_BYTES);
  crypto.getRandomValues(bytes);
  return [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** The secret for this browser, creating one on first use. */
export function getSubmitterKey() {
  if (cached) return cached;

  try {
    let key = localStorage.getItem(KEY_STORAGE);
    if (!key) {
      key = generateKey();
      localStorage.setItem(KEY_STORAGE, key);
    }
    cached = key;
    return key;
  } catch (error) {
    // Private modes can refuse storage. Work with a key that lasts as long as
    // the page does: jobs still submit, they just cannot be claimed later.
    console.warn("Could not persist a submitter key:", error);
    cached = cached || generateKey();
    return cached;
  }
}

/** Whether a key already exists, without creating one. */
export function hasSubmitterKey() {
  try {
    return Boolean(localStorage.getItem(KEY_STORAGE));
  } catch {
    return false;
  }
}

/** Merge the submitter key into a headers object.
 *
 * Signed in with no key here, the header is left off deliberately. Minting one
 * would hand this browser a brand new identity at the exact moment somebody
 * has told us who they are: the coordinator prefers a key over a session, so a
 * key created here would file the work under a digest the account has never
 * heard of, and their existing jobs would vanish from the page that just
 * greeted them by name.
 *
 * With no key and no account there is nothing else to be, so one is created --
 * which is what has always happened.
 */
export function submitterHeaders(extra = {}) {
  if (signedIn && !hasSubmitterKey()) return { ...extra };
  return { ...extra, "X-Submitter-Key": getSubmitterKey() };
}

/** Replace the stored key, e.g. to adopt one exported from another browser. */
export function setSubmitterKey(key) {
  const clean = (key || "").trim();
  if (!/^[A-Za-z0-9_-]{32,256}$/.test(clean)) {
    throw new Error("That does not look like a submitter key.");
  }
  try {
    localStorage.setItem(KEY_STORAGE, clean);
  } catch (error) {
    console.warn("Could not persist the submitter key:", error);
  }
  cached = clean;
  return clean;
}
