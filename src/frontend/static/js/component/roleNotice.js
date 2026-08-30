// What to say when somebody is standing on the other side's page.
//
// The two sides are separate but the pages are not locked: a link, a bookmark
// or a typed address reaches any of them. That is deliberate -- a person can
// genuinely be both, lending a graphics card and training their own models --
// so arriving somewhere is never refused.
//
// What was wrong is that it was never explained. A GPU owner who opened
// /workspace got a working page that said "No key yet" and "Nothing sent yet",
// which reads as a bug in their account rather than as somebody else's half of
// the product. A visitor with no GPU who opened /connect was offered node
// registration and a hardware check they were going to fail.
//
// So the page says where you are, offers the way in if you want this side, and
// the way back if you do not.

import {
  ROLE_ENTRY, ROLE_HOME, ROLE_LABEL,
  currentRole, hasRole, isNewHere, pageRole,
} from "./role.js";

const NOTICE_ID = "roleNotice";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function action(href, text, primary) {
  const link = el("a", primary ? "role-notice-go is-primary" : "role-notice-go", text);
  link.href = href;
  return link;
}

/** The sentence for somebody who has not chosen a side at all. */
function forNewcomer(page, role) {
  const box = el("div", "role-notice");
  box.appendChild(el("strong", null,
    `This page is for ${ROLE_LABEL[role]}.`));

  box.appendChild(el("p", null, page.entry
    ? "You have not set anything up in this browser yet, which is fine — this "
      + "is one of the places to start."
    : "There is nothing here until you set that up, because everything on this "
      + "page belongs to a key this browser does not have yet."));

  const row = el("div", "role-notice-actions");
  if (!page.entry) {
    row.appendChild(action(ROLE_ENTRY[role], "Set that up", true));
  }
  row.appendChild(action("/", "See both sides"));
  box.appendChild(row);
  return box;
}

/** The sentence for somebody who is signed in to the other side. */
function forOtherSide(page, role, mine) {
  const box = el("div", "role-notice");
  box.appendChild(el("strong", null,
    `This is the other side of the network: ${ROLE_LABEL[role]}.`));

  box.appendChild(el("p", null, page.entry
    ? `You are set up for ${ROLE_LABEL[mine]}. Setting this up as well adds `
      + "the other side — it does not replace what you already have."
    : `You are set up for ${ROLE_LABEL[mine]}, so this page has nothing of `
      + "yours on it. You can take up this side as well; nothing you already "
      + "have goes away."));

  const row = el("div", "role-notice-actions");
  row.appendChild(action(ROLE_HOME[mine], "Back to your side", true));
  if (!page.entry) {
    row.appendChild(action(ROLE_ENTRY[role], `Also start ${ROLE_LABEL[role]}`));
  }
  box.appendChild(row);
  return box;
}

/**
 * Put the notice at the top of the page, or take it away.
 *
 * Called on every page load and again whenever a key appears or disappears,
 * so that registering a node while standing on /node clears the notice
 * without a refresh.
 */
export function showRoleNotice() {
  document.getElementById(NOTICE_ID)?.remove();

  const page = pageRole(window.location.pathname);
  if (!page) return;                    // belongs to neither side
  if (hasRole(page.role)) return;       // it is yours; nothing to explain

  // An entry page introduces itself. "Contribute your GPU -- join the network
  // and put an idle graphics card to work" says everything a first-time
  // visitor needs, and a banner above it saying "this page is for lending a
  // GPU" is the same sentence twice. The notice earns its place only when
  // somebody is signed in to the other side and would otherwise wonder why
  // none of this is theirs.
  if (page.entry && isNewHere()) return;

  const wrap = el("div", null);
  wrap.id = NOTICE_ID;

  const mine = currentRole();
  wrap.appendChild(isNewHere() || !mine
    ? forNewcomer(page, page.role)
    : forOtherSide(page, page.role, mine));

  // Above <main>, not inside it. Pages lay their main element out as a grid or
  // as a centring flex row, and a banner dropped in at the top became one more
  // item competing for space -- on the connect page it landed beside the card
  // rather than above it.
  const main = document.querySelector("main");
  if (main && main.parentNode) main.parentNode.insertBefore(wrap, main);
  else document.body.insertBefore(wrap, document.body.firstChild);
}
