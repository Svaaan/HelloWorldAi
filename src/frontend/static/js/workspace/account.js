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

function renderSignedOut(host) {
  host.replaceChildren();

  const line = el("p", "acct-line");
  line.appendChild(document.createTextNode(
    "Your key is what owns this work, and it only exists in this browser. "
    + "Signing in adds a second way back to it, and lets the same jobs show "
    + "up on your other machines."));
  host.appendChild(line);

  // An anchor, not a fetch: the sign-in is a redirect to GitHub and back.
  const button = el("a", "btn acct-signin", "Sign in with GitHub");
  button.href = "/auth/github/start";
  host.appendChild(button);

  host.appendChild(el("p", "acct-fineprint",
    "Only your GitHub username is read, and no repository access is asked "
    + "for. Your key is never sent to GitHub and never stored here."));
}

function renderSignedIn(host, data) {
  host.replaceChildren();

  // The username, and no avatar.
  //
  // An avatar would need `img-src` opened up to avatars.githubusercontent.com,
  // and would fetch from GitHub on every workspace load -- telling them when
  // somebody is using this service, from what address, for a picture that says
  // nothing the name beside it does not. This service's whole claim is that it
  // holds as little as it can get away with; leaking the same fact outward
  // instead would be the same trade with extra steps.
  const who = el("p", "acct-who");
  who.appendChild(el("strong", null, data.login || "Signed in"));
  host.appendChild(who);

  const keys = Number(data.keys_linked || 0);
  host.appendChild(el("p", "acct-line", keys === 1
    ? "One key is linked to this account. Sign in on another machine and this "
      + "work will be there."
    : keys
      ? `${keys} keys are linked to this account, and you can see the work `
        + "from all of them here."
      : "No key is linked yet. One will be, the first time you send a job."));

  const out = el("button", "link-button", "Sign out of GitHub");
  out.type = "button";
  out.addEventListener("click", async () => {
    out.disabled = true;
    try {
      await fetch("/auth/sign-out", {
        method: "POST", credentials: "same-origin",
      });
    } catch (error) {
      console.warn("Could not sign out:", error);
    }
    // Deliberately not touching the key. Ending a session should not destroy
    // an identity that nobody can reissue -- that is what the header's sign
    // out is for, and it asks first.
    window.location.reload();
  });
  host.appendChild(out);
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

  const panel = document.getElementById("accountPanel");
  const host = document.getElementById("accountBody");
  if (!panel || !host) return account;

  // No OAuth application on this deployment: there is nothing to offer, and a
  // panel explaining an absent feature is worse than no panel.
  if (!data.github) {
    panel.hidden = true;
    return account;
  }

  panel.hidden = false;
  if (data.signed_in) renderSignedIn(host, account);
  else renderSignedOut(host);

  return account;
}
