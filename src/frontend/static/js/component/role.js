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
// keypair from registration, a builder with the submitter key. Holding one is
// what "signed in" means here -- there are no accounts, so the presence of a
// key is the whole of it.

const NODE_KEY = "currentNodeId";
const BUILDER_KEY = "submitterKey";

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
export function isBuilder() {
  return Boolean(stored(BUILDER_KEY));
}

/** Neither side has been set up yet: a first-time visitor. */
export function isNewHere() {
  return !isContributor() && !isBuilder();
}

/**
 * Whether a thing tagged for `role` should be shown.
 *
 * "any" is always shown. A page belonging to one side is shown to that side,
 * and also to somebody who is neither yet -- otherwise a new visitor could
 * never reach the page that would sign them in.
 */
export function showsFor(role) {
  if (!role || role === "any") return true;
  if (role === "contributor") return isContributor() || isNewHere();
  if (role === "builder") return isBuilder() || isNewHere();
  return true;
}
