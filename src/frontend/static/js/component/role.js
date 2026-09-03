// Which of the two sides of the network this browser belongs to.
//
// There are two kinds of person here and they want opposite things:
//
//   a contributor  has an NVIDIA GPU and wants to lend it
//   a builder      has data and wants it trained, and may well own no GPU at all
//
// The product used to assume everyone was a contributor. The front door said
// "Contribute your GPU", ran a GPU check, and offered to register a node --
// so somebody arriving with a dataset and an ordinary laptop was told their
// hardware was unsuitable for something they never wanted to do.
//
// Each side proves itself with a key it keeps: a contributor with the node
// keypair from registration, a builder with the submitter key. Holding one used
// to be the whole of what "signed in" meant -- there were no accounts.
//
// There are now, on the builder side, and they are the other way of being
// somebody. A laptop signed in with GitHub holds no submitter key and still
// owns every job sent from the desktop, so a browser is a builder if it holds
// the key *or* is signed in. Getting this wrong is not subtle: the first
// version of the account shipped with this file unchanged, and a signed-in
// person with two finished models on screen was told by the banner above them
// that "there is nothing here until you set that up", with the navigation
// hidden for good measure.

const NODE_KEY = "currentNodeId";
const BUILDER_KEY = "submitterKey";

// Set by account.js once /auth/me has answered. False until then, which is the
// right default: it is what this browser looks like with no account at all.
let signedInBuilder = false;

/** Told by account.js when the sign-in state is known or changes. */
export function setSignedInBuilder(value) {
  const next = Boolean(value);
  if (next === signedInBuilder) return;
  signedInBuilder = next;

  // The header and the notice were both drawn before this was known. An event
  // rather than a direct call because role.js is the bottom of this stack --
  // header.js imports it, so it cannot import the header back.
  document.dispatchEvent(new CustomEvent("hw:identity-changed"));
}

function stored(name) {
  try {
    return localStorage.getItem(name);
  } catch (error) {
    // Storage can be refused outright in private modes. Treating that as
    // "no role" is right: it means nothing has been set up here.
    console.warn("Could not read local storage:", error);
    return null;
  }
}

/** This browser has a node registered to it. */
export function isContributor() {
  return Boolean(stored(NODE_KEY));
}

/** This browser holds a builder key, so it can own jobs and collect models. */
export function holdsBuilderKey() {
  return Boolean(stored(BUILDER_KEY));
}

/** This browser can own jobs and collect models: by key, or by being signed in. */
export function isBuilder() {
  return holdsBuilderKey() || signedInBuilder;
}

/** Neither side has been set up yet: a first-time visitor. */
export function isNewHere() {
  return !isContributor() && !isBuilder();
}

/** Whether this browser holds the key for `role`. */
export function hasRole(role) {
  if (role === "contributor") return isContributor();
  if (role === "builder") return isBuilder();
  return false;
}

/**
 * Whether a thing tagged for `role` should be shown.
 *
 * "any" is always shown. Everything else is shown to the side that owns it,
 * and to nobody else -- including a visitor who is neither yet. They are sent
 * to the front door, which is the page that asks which side they are on;
 * offering them both sides' navigation asks the same question twice and
 * answers neither.
 */
export function showsFor(role) {
  if (!role || role === "any") return true;
  return hasRole(role);
}

// --- which pages belong to which side ------------------------------------
//
// Kept here rather than in the header, because two different things need it:
// the navigation, which shows you your own side, and the notice that explains
// where you have landed when you follow a link or a bookmark to the other one.
//
// `entry` marks a page whose purpose is to sign somebody up for that side.
// Arriving there without the role is the point, not a mistake. The others are
// interior pages, which without the key are an empty room.
const PAGES = {
  "/setup": { role: "contributor", entry: true },
  "/node": { role: "contributor" },
  "/distribution": { role: "builder", entry: true },
  "/workspace": { role: "builder" },
};

export const ROLE_LABEL = {
  contributor: "lending a GPU",
  builder: "training a model",
};

export const ROLE_HOME = {
  contributor: "/node",
  builder: "/workspace",
};

export const ROLE_ENTRY = {
  // Registering a node happens on the front door, where taking up the other
  // side does too. There is no separate connect page any more.
  contributor: "/",
  builder: "/distribution",
};

/** What this path is for, or null if it belongs to neither side. */
export function pageRole(path) {
  const clean = String(path || "").replace(/\/+$/, "") || "/";
  return PAGES[clean] || null;
}

/** The side this browser is on, when it is on exactly one. */
export function currentRole() {
  if (isContributor() && !isBuilder()) return "contributor";
  if (isBuilder() && !isContributor()) return "builder";
  return null;      // both, or neither
}
