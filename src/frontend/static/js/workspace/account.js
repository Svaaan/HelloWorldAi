// Signing in with GitHub, on top of the key -- never instead of it.
//
// The key model is the good part of this service and it is not being replaced:
// the coordinator stores sha256(key) and nothing else, so reading its database
// gives an attacker nothing they can act with. That is also exactly why an
// account cannot hand your key back to you on a new laptop. It never had it.
//
// So an account holds digests, not keys, and being signed in means the
// coordinator can work out which piles of work are yours without you carrying
// anything. What that buys, concretely:
//
//   * the jobs you sent from the desktop are on the laptop
//   * a lost key file stops being a lost workspace, as long as you signed in
//   * work has an owner, which is what a list of GPUs you like has to hang off
//
// Nothing here is required, and a deployment with no OAuth application
// configured hides all of it rather than offering a button that leads to an
// error page.

import { setAccountLogin } from "../component/header.js";
import { setSignedInBuilder } from "../component/role.js";
import {
  getSubmitterKey, setSignedIn,
} from "../distribution/submitter.js";

// Which key was last linked to which account, so a page load is not another
// request. Beside the key rather than in a cookie: clearing site data clears
// the key too, and then the link genuinely does need making again.
const LINKED_MARKER = "accountLinkedKey";

let account = null;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function stored(name) {
  try {
    return localStorage.getItem(name);
  } catch {
    return null;                        // private mode: ask again, harmlessly
  }
}

/** Ask who is signed in. Answers a shape, never throws. */
async function fetchAccount() {
  try {
    const res = await fetch("/auth/me", { credentials: "same-origin" });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    return await res.json();
  } catch (error) {
    // An older coordinator has no /auth routes at all. That is not a fault
    // worth showing anybody: it means this deployment has no sign-in, which is
    // the same as one that was never configured.
    console.warn("Could not check the sign-in state:", error);
    return { signed_in: false, github: false };
  }
}

/** Attach the key in this browser to the account, once.
 *
 * This is the whole of what makes signing in a wrap. The jobs already sent
 * under this key stay under it -- nothing is migrated, no task is rewritten --
 * and the account simply learns that this digest is one of yours.
 *
 * A browser with no key gets one here rather than being left without.
 *
 * That is not a detail. An account holds digests, so an account with none owns
 * nothing, and the coordinator would resolve a signed-in submission to no
 * submitter at all. It does not refuse that -- an anonymous API client is
 * allowed to queue work nobody can claim -- so a person whose first act was
 * signing in would have sent a job into a state where the model could never be
 * collected, and nothing would have said so.
 *
 * Minting it here and not on the coordinator is the same rule as everywhere
 * else: a submitter_id is the digest of a key some browser holds, and the
 * coordinator does not get to invent identities. What the account changes is
 * that nobody has to look after the file for this to keep working -- lose it,
 * and signing in still reaches the work.
 */
async function linkKey(login) {
  // getSubmitterKey() creates one when there is none. Deliberate here, and
  // deliberately not in submitterHeaders(), which must not mint a key while
  // the answer to "who is this" is still in flight.
  const key = getSubmitterKey();

  const marker = `${login}:${key.slice(0, 8)}`;
  if (stored(LINKED_MARKER) === marker) return account;

  try {
    const res = await fetch("/auth/link", {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-Submitter-Key": key },
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const result = await res.json();
    if (result.linked) {
      try {
        localStorage.setItem(LINKED_MARKER, marker);
      } catch {
        // It will simply be linked again next time, which costs one request.
      }
      account = { ...account, keys_linked: result.keys };
    }
  } catch (error) {
    // Worth saying in the console and nowhere else: everything on the page
    // still works through the key, which is what it did before accounts.
    console.warn("Could not link this key to the account:", error);
  }

  return account;
}

// --- what it looks like ----------------------------------------------------
//
// Almost nothing, now. There was a panel here: signed in it showed a name and a
// "Sign out of GitHub" link, while the header carried a second sign-out with
// nearly the same words that forgot the key instead -- two controls on one page,
// one of them irreversible, neither saying which. The name and the single way
// out are in the header. Signed out, the panel offered a door the front page
// already offers properly.
//
// What is left is one line, in the key panel, for a browser holding a key that
// no account has been told about. That is a real thing to offer and this is
// where it belongs: somebody looking at their key is exactly who might think
// "and if I lose this file?"

function renderSignInLine(host) {
  host.replaceChildren();

  const line = el("p", null,
    "This key only exists in this browser. ");
  line.style.margin = "0";

  const link = el("a", "link-button", "Sign in with GitHub");
  link.href = "/auth/github/start";
  line.appendChild(link);
  line.appendChild(document.createTextNode(
    " to reach the same work from your other machines."));

  host.appendChild(line);
  host.hidden = false;
}

// --- wiring ----------------------------------------------------------------

/** Work out who is signed in, and say so if there is somewhere to say it.
 *
 * Safe to call on a page with no account panel: /distribution does exactly
 * that, so a job submitted from there is filed under the account rather than
 * under a key minted a second earlier.
 *
 * Returns once the answer is known, so callers can await it before asking the
 * coordinator anything that depends on who is asking.
 */
export async function initAccount({ onChange } = {}) {
  const data = await fetchAccount();
  account = data;
  setSignedIn(Boolean(data.signed_in));

  // Two listeners, two jobs. submitter.js decides whether to mint a key;
  // role.js decides what the header and the notice say. A signed-in browser
  // holding no key is a builder, and until this both of them thought it was
  // nobody.
  setSignedInBuilder(Boolean(data.signed_in));

  if (data.signed_in) {
    await linkKey(data.login || "");
    // Linking may have changed what this browser can see.
    onChange?.();
  }

  // Who is signed in belongs in the header, next to the one way out of being
  // them. Told rather than read, because /auth/me lands after the header has
  // already been drawn.
  setAccountLogin(data.signed_in ? (data.login || "Signed in") : null);

  const host = document.getElementById("identitySignIn");
  if (!host) return account;

  // Nothing to offer if this deployment has no OAuth application, and nothing
  // to say to somebody already signed in -- the header has their name.
  host.hidden = true;
  if (data.github && !data.signed_in) renderSignInLine(host);

  return account;
}
