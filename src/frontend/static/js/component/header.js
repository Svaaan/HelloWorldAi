// Shared site header.
//
// Injected into #header-placeholder on every page, so this is also where the
// app's navigation lives. Two things it has to get right: mark the page you
// are on, and fail visibly rather than leaving a page with no header at all.

import {
  ROLE_LABEL, hasRole, holdsBuilderKey, isNewHere, pageRole, showsFor,
} from "./role.js";
import { showRoleNotice } from "./roleNotice.js";

const COUNT_INTERVAL_MS = 60000;

// Set by the connect flow (see connect/nodeSession.js) once a node has proved
// ownership. Its presence is what "this browser has a node" means everywhere.
const NODE_ID_KEY = "currentNodeId";
const BUILDER_KEY = "submitterKey";

let countTimer = null;

// The signed-in login, or null. Told by account.js once /auth/me answers, which
// is after this header has already been drawn -- so setting it redraws.
let accountLogin = null;

/** Told by account.js. Nothing else should call this. */
export function setAccountLogin(login) {
  const next = login || null;
  if (next === accountLogin) return;
  accountLogin = next;
  refreshNav();
}

// Everything that makes up an identity in this browser, per side. Signing out
// of one must leave the other alone: somebody can lend a graphics card and
// train their own models, and those are two separate keys.
//
// The backed-up markers go with their key. Left behind, the next key created
// here would look as though it had already been saved, and the warning that
// matters most would not appear.
const IDENTITY_KEYS = {
  // accountLinkedKey records that this key was linked to a GitHub account, so
  // a page load does not re-link it every time. It is a note about the key, so
  // it goes when the key does -- left behind, it would name a key that is gone
  // and suppress the link the next one actually needs.
  builder: ["submitterKey", "submitterKeyBackedUp", "accountLinkedKey"],
  contributor: [
    "currentNodeId", "nodePrivateKey", "nodePublicKeyBase64",
    "nodeSessionToken", "nodeKeyBackedUp",
  ],
};

const BACKED_UP_MARKER = {
  builder: "submitterKeyBackedUp",
  contributor: "nodeKeyBackedUp",
};

const SAVE_PAGE = { builder: "/workspace", contributor: "/node" };

export async function loadHeader() {
  const slot = document.getElementById("header-placeholder");
  if (!slot) return;

  try {
    const res = await fetch("/template/header.html");
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    // Our own template from our own origin, so parsing it as markup is safe.
    slot.innerHTML = await res.text();
  } catch (error) {
    // Without this the page loses its only navigation and says nothing about
    // why, which reads as a broken page rather than a failed request.
    console.error("Could not load the header:", error);
    slot.replaceChildren(fallbackHeader());
    return;
  }

  refreshNav();
  wireSignOut();

  // The front door is a sign-in page: two doors, and nothing else to do. A
  // count of machines online means nothing to somebody who has not yet chosen
  // a side, and polling for it is a request per minute nobody asked for.
  if (!isFrontDoor()) startCountPolling();

  // Another tab connecting or disconnecting a node changes what this nav
  // should show, and so does coming back to a tab left open for a while.
  window.addEventListener("storage", (event) => {
    if (!event.key || event.key === NODE_ID_KEY || event.key === BUILDER_KEY) {
      refreshNav();
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshNav();
  });

  // Signing in is decided by a request, so the answer lands after this header
  // has already been drawn. Without this a signed-in browser holding no key
  // renders as a first-time visitor -- no navigation, and a banner telling it
  // there is nothing here.
  document.addEventListener("hw:identity-changed", refreshNav);
}

function refreshNav() {
  markCurrentPage();
  applyRoles();
  showRoleNotice();
  updateSignOut();
}

function stored(name) {
  try {
    return localStorage.getItem(name);
  } catch {
    return null;                          // private mode: nothing to forget
  }
}

/** Which side a sign-out here would end.
 *
 * Not currentRole(): that answers null for somebody holding both keys, because
 * it is asking "which one thing is this browser" and there are two. The
 * question here is narrower -- which side am I leaving *from this page* -- and
 * that has an answer even when both are present.
 */
/** Keys only, and not accounts.
 *
 * This button forgets a key. A browser signed in with GitHub and holding no
 * key has nothing for it to do: offering "sign out of training a model" there
 * would promise to destroy something that is not there, and leave the session
 * they actually have running. Ending that is the account panel's job, and it
 * says so in those words.
 */
function holdsKeyFor(role) {
  return role === "builder" ? holdsBuilderKey() : hasRole(role);
}

function signedInAs() {
  const here = pageRole(window.location.pathname);
  if (here && holdsKeyFor(here.role)) return here.role;   // the side you are on

  const roles = ["builder", "contributor"].filter(holdsKeyFor);
  return roles.length === 1 ? roles[0] : null;        // ambiguous: offer neither
}

function updateSignOut() {
  const button = document.getElementById("signOutButton");
  const chip = document.getElementById("headerAccount");

  if (chip) {
    chip.hidden = !accountLogin;
    chip.textContent = accountLogin || "";
  }

  if (!button) return;

  // Either kind of identity is something to sign out of. It used to be keys
  // only, so a browser holding no key but signed in with GitHub had no way out
  // of the header at all -- the only sign-out was in the workspace panel.
  const role = signedInAs();
  button.hidden = !role && !accountLogin;

  // An icon, with the words in the accessible name rather than beside it. The
  // label used to name the side -- "Sign out of training a model" -- which was
  // precise and also the longest thing in the header, sitting next to a second
  // control in the workspace with almost the same words.
  const leaving = [
    accountLogin ? "GitHub" : null,
    role ? ROLE_LABEL[role] : null,
  ].filter(Boolean).join(" and ");

  // Guarded because the button is hidden when there is neither, and a hidden
  // control still has an accessible name -- "Sign out of " with nothing after
  // it is what a screen reader would have read out.
  const label = leaving ? `Sign out of ${leaving}` : "Sign out";
  button.setAttribute("aria-label", label);
  button.setAttribute("title", label);
}

function wireSignOut() {
  const button = document.getElementById("signOutButton");
  const modal = document.getElementById("signOutModal");
  const confirmButton = document.getElementById("signOutConfirm");
  if (!button || !modal) return;

  button.addEventListener("click", () => {
    const role = signedInAs();
    if (!role && !accountLogin) return;

    // Signed in, this is simply reversible, and it should say so in one line.
    //
    // It clears the key as well -- which used to be the frightening half, and
    // is not any more. The account links that key's digest and reads span every
    // digest an account owns, so signing back in reaches the same work. Leaving
    // the key behind would be the wrong trade anyway: on a shared machine the
    // next person would still hold the thing that owns your jobs.
    //
    // "Am I finished signing out?" now has an answer. Yes, and nothing is lost.
    const lede = document.getElementById("signOutLede");
    const list = document.getElementById("signOutList");
    if (list) list.replaceChildren();

    if (accountLogin) {
      if (lede) {
        lede.textContent =
          `Signs you out of ${accountLogin} on this browser and clears what it `
          + "holds. Sign in again and your work is here.";
      }
    } else if (role) {
      // No account, so the key really is the only thing, and the dialog has to
      // be honest about that rather than reassuring.
      if (lede) lede.textContent = "This cannot be undone.";

      if (list) {
        const item = document.createElement("li");
        item.className = "signout-permanent";
        item.textContent =
          `Forgets the key for ${ROLE_LABEL[role]} in this browser. `
          + "It is not stored anywhere else and nobody can issue you another "
          + "one.";
        list.appendChild(item);
      }
    }

    // The warning that matters, and only when it matters: a key that has never
    // been written to a file, about to be forgotten, with no account to reach
    // the work afterwards.
    const unsaved = document.getElementById("signOutUnsaved");
    if (unsaved) {
      unsaved.hidden = Boolean(accountLogin) || !role
                       || Boolean(stored(BACKED_UP_MARKER[role]));
    }

    const saveLink = document.getElementById("signOutSaveFirst");
    if (saveLink && role) saveLink.href = SAVE_PAGE[role];

    // Red is for the version that destroys something. Signed in it does not,
    // and a red button asking you to confirm a reversible thing teaches people
    // to ignore red buttons.
    if (confirmButton) {
      confirmButton.className = accountLogin ? "btn" : "btn signout-danger";
    }

    modal.style.display = "flex";
  });

  if (confirmButton) {
    confirmButton.addEventListener("click", async () => {
      const role = signedInAs();
      confirmButton.disabled = true;

      if (role) {
        for (const key of IDENTITY_KEYS[role]) {
          try {
            localStorage.removeItem(key);
          } catch {
            /* nothing stored to remove */
          }
        }
      }

      if (accountLogin) {
        try {
          await fetch("/auth/sign-out", {
            method: "POST", credentials: "same-origin",
          });
        } catch (error) {
          // The keys are already gone from this browser, which is the half
          // that matters here. Say so and carry on rather than leaving
          // somebody on a dialog that appears to have done nothing.
          console.warn("Could not end the GitHub session:", error);
        }
      }

      // Back to the front door, which is where somebody with no identity
      // belongs.
      window.location.href = "/";
    });
  }

  for (const closer of modal.querySelectorAll('[data-close="signOutModal"]')) {
    closer.addEventListener("click", () => { modal.style.display = "none"; });
  }
}

// Highlight the link for the page being viewed. Compared on pathname so query
// strings and cache-busting params do not defeat the match.
function markCurrentPage() {
  document.querySelectorAll(".header-nav a").forEach((link) => {
    if (linkPath(link) === currentPath()) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

// Show each side of the network only its own pages.
//
// Two rules, and both of them are about not asking two questions at once.
//
// A link is yours if you hold the key for its side. It used to also un-hide
// the current page's own link so that something was always marked, which meant
// a GPU owner who opened /workspace grew a "Your workspace" link. Being
// somewhere that is not yours is explained by the page now (roleNotice.js)
// rather than by the header changing shape.
//
// And a link belongs *here* if it is for the same side as the page. Standing
// on /connect -- which asks you to register a graphics card -- while the
// header offers "Send work" and "Your workspace" puts the two profiles in
// front of somebody at the moment they are being asked about one of them.
// The front door belongs to neither side, so it is where both appear and
// where you go to change sides.
//
function applyRoles() {
  const here = pageRole(currentPath());

  document.querySelectorAll(".header-nav a").forEach((link) => {
    const role = link.dataset.role;
    link.hidden = !showsFor(role) || Boolean(here && role && role !== here.role);
  });

  // Somebody who has set nothing up has no side, so there is nothing to
  // navigate. This used to apply only on the front door and the connect page,
  // so a first-time visitor who followed a link to /node was shown both
  // sides' navigation at once -- five destinations, none of them theirs.
  //
  // And the front door has none either, whoever you are. The two cards on it
  // say "Open your workspace" and "Open your node" in full sentences; a row of
  // links above them saying the same thing smaller is the same choice offered
  // twice, on the one page whose whole job is asking it once.
  const nav = document.querySelector(".header-nav");
  if (nav) nav.hidden = isNewHere() || isFrontDoor();

  // Nothing to count for somebody deciding which side they are on.
  const count = document.querySelector(".header-details");
  if (count) count.hidden = isFrontDoor();
}

function isFrontDoor() {
  return currentPath() === "/";
}

function currentPath() {
  // "/" must survive: stripping its slash leaves an empty string, and the old
  // fallback made the front door look like another page -- which matters more
  // now that the front door is where both sides are taken up.
  const path = window.location.pathname.replace(/\/+$/, "");
  return path || "/";
}

function linkPath(link) {
  return new URL(link.href, window.location.origin).pathname.replace(/\/+$/, "");
}

function fallbackHeader() {
  const header = document.createElement("header");
  header.className = "header";

  const brand = document.createElement("a");
  brand.className = "header-logo";
  brand.href = "/";
  brand.textContent = "HelloWorldAi";
  header.appendChild(brand);

  return header;
}

function startCountPolling() {
  const label = document.getElementById("connectedNodesCount");
  if (!label) return;

  // loadHeader() can run more than once on a long-lived page; without this
  // each call would leave another timer behind.
  if (countTimer) clearInterval(countTimer);

  async function update() {
    try {
      const res = await fetch("/get-connected-nodes-count");
      if (!res.ok) throw new Error(`Server returned ${res.status}`);

      const data = await res.json();
      const count = data.connected_nodes_count;

      label.textContent = count ?? "—";
      label.classList.toggle("is-stale", count === null || count === undefined);
    } catch (error) {
      console.error("Error updating connected nodes count:", error);
      // A dash reads as "unknown"; a 0 would read as "nobody is online".
      label.textContent = "—";
      label.classList.add("is-stale");
    }
  }

  update();
  countTimer = setInterval(update, COUNT_INTERVAL_MS);

  // Polling a hidden tab is wasted work on both ends.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (countTimer) clearInterval(countTimer);
      countTimer = null;
    } else if (!countTimer) {
      update();
      countTimer = setInterval(update, COUNT_INTERVAL_MS);
    }
  });
}
