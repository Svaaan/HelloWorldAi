// Shared site header.
//
// Injected into #header-placeholder on every page, so this is also where the
// app's navigation lives. Two things it has to get right: mark the page you
// are on, and fail visibly rather than leaving a page with no header at all.

import { isNewHere, pageRole, showsFor } from "./role.js";
import { showRoleNotice } from "./roleNotice.js";

const COUNT_INTERVAL_MS = 60000;

// Set by the connect flow (see connect/nodeSession.js) once a node has proved
// ownership. Its presence is what "this browser has a node" means everywhere.
const NODE_ID_KEY = "currentNodeId";
const BUILDER_KEY = "submitterKey";

let countTimer = null;

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
}

function refreshNav() {
  markCurrentPage();
  applyRoles();
  showRoleNotice();
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
