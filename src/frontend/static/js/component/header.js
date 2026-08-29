// Shared site header.
//
// Injected into #header-placeholder on every page, so this is also where the
// app's navigation lives. Two things it has to get right: mark the page you
// are on, and fail visibly rather than leaving a page with no header at all.

const COUNT_INTERVAL_MS = 60000;

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

  markCurrentPage();
  startCountPolling();
}

// Highlight the link for the page being viewed. Compared on pathname so query
// strings and cache-busting params do not defeat the match.
function markCurrentPage() {
  const here = window.location.pathname.replace(/\/+$/, "") || "/connect";

  document.querySelectorAll(".header-nav a").forEach((link) => {
    const target = new URL(link.href, window.location.origin).pathname
      .replace(/\/+$/, "");
    if (target === here) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
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
