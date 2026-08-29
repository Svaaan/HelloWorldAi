// Shared site header.
//
// Injected into #header-placeholder on every page, so this is also where the
// app's navigation lives. Two things it has to get right: mark the page you
// are on, and fail visibly rather than leaving a page with no header at all.

const COUNT_INTERVAL_MS = 60000;

// Set by the connect flow (see connect/nodeSession.js) once a node has proved
// ownership. Its presence is what "this browser has a node" means everywhere.
const NODE_ID_KEY = "currentNodeId";

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
  startCountPolling();

  // Another tab connecting or disconnecting a node changes what this nav
  // should show, and so does coming back to a tab left open for a while.
  window.addEventListener("storage", (event) => {
    if (!event.key || event.key === NODE_ID_KEY) refreshNav();
  });
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) refreshNav();
  });
}

function refreshNav() {
  markCurrentPage();
  hideConnectWhenConnected();
}

// Highlight the link for the page being viewed. Compared on pathname so query
// strings and cache-busting params do not defeat the match.
function markCurrentPage() {
  document.querySelectorAll(".header-nav a").forEach((link) => {
    if (linkPath(link) === currentPath()) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
}

// Once this browser has a node, "Connect" is an entry point you have already
// been through, so it just takes up room in the nav.
//
// The one exception is the /connect page itself: hiding the link for the page
// you are looking at leaves the header with nothing marked, which reads as a
// bug. Someone on /connect is also most likely there to add or swap a node.
function hideConnectWhenConnected() {
  const connected = hasNode();

  document.querySelectorAll(".header-nav a").forEach((link) => {
    if (linkPath(link) !== "/connect") return;
    link.hidden = connected && currentPath() !== "/connect";
  });
}

function hasNode() {
  try {
    return Boolean(localStorage.getItem(NODE_ID_KEY));
  } catch (error) {
    // Storage can be unavailable outright in some privacy modes. Showing the
    // link is the safe default -- it is never harmful, only redundant.
    console.warn("Could not read the node session:", error);
    return false;
  }
}

function currentPath() {
  return window.location.pathname.replace(/\/+$/, "") || "/connect";
}

function linkPath(link) {
  return new URL(link.href, window.location.origin).pathname.replace(/\/+$/, "");
}

function fallbackHeader() {
  const header = document.createElement("header");
  header.className = "header";

  const brand = document.createElement("a");
  brand.className = "header-logo";
  brand.href = "/connect";
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
