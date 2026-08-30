// The list of nodes offering their GPUs, on the distribution page.
//
// Everything here is built with createElement + textContent rather than
// innerHTML. Node ids, CPU brands and GPU names are reported by whoever
// registered the node, so treating them as markup would let a machine register
// itself under a name like `<img onerror=...>` and run script in the dashboard
// of everyone browsing this page.

import { showNodeModal } from "./modalHandler.js";

// Whether a machine is free changes minute to minute, so a minute-long poll
// would show a stale answer for most of its life.
const POLL_INTERVAL_MS = 10000;

let pollTimer = null;
let inFlight = false;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

// Exported so the summarising can be tested directly; a four-card rig is
// awkward to arrange otherwise.
export function describeGpus(node) {
  const gpus = node.capabilities?.gpu;
  if (!Array.isArray(gpus) || !gpus.length) return gpus?.name || "None";

  // Group identical cards and drop the vendor words every NVIDIA GPU shares.
  // A four-card rig read as "NVIDIA GeForce RTX 3070, NVIDIA GeForce RTX 3060,
  // NVIDIA GeForce RTX 3060, NVIDIA GeForce GTX 1660 Super", which is mostly
  // the word NVIDIA and does not fit anywhere. "2x RTX 3060, RTX 3070,
  // GTX 1660 Super" says the same thing and can be read at a glance.
  const counts = new Map();
  for (const g of gpus) {
    const name = (g.name || "Unknown GPU")
      .replace(/^NVIDIA\s+/i, "")
      .replace(/^GeForce\s+/i, "");
    counts.set(name, (counts.get(name) || 0) + 1);
  }

  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([name, n]) => (n > 1 ? `${n}× ${name}` : name))
    .join(", ");
}

function countGpus(node) {
  const gpus = node.capabilities?.gpu;
  return Array.isArray(gpus) ? gpus.length : (gpus ? 1 : 0);
}

function specRow(label, value) {
  const row = el("div", "node-spec");
  row.appendChild(el("span", "spec-label", label));
  row.appendChild(el("span", null, value));
  return row;
}

function notice(title, detail, kind) {
  const box = el("div", kind ? `empty-message ${kind}` : "empty-message");
  box.appendChild(el("strong", null, title));
  if (detail) box.appendChild(el("p", null, detail));
  return box;
}

// The headline number: what this machine can actually contribute. It is the
// reason someone is on this page, so it belongs on the card and not only
// behind a click into the modal.
function computeBadge(node) {
  const tflops = Number(node.total_gpu_tflops);
  if (!tflops || Number.isNaN(tflops)) return null;

  const badge = el("span", "node-compute");
  badge.appendChild(el("strong", null, tflops.toFixed(1)));
  badge.appendChild(el("span", null, " TFLOPS"));
  return badge;
}

function buildNodeCard(node) {
  const card = el("article", "node-item");
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.dataset.nodeId = node.node_id;

  const gpuCount = countGpus(node);
  card.setAttribute(
    "aria-label",
    `Send work to node ${node.node_id}, ${gpuCount} GPU${gpuCount === 1 ? "" : "s"}`
    + (node.busy ? ", currently busy" : "")
  );

  const head = el("div", "node-header");
  head.appendChild(el("span", "node-id", node.node_id));

  // A machine that is training is still online, but it is not free -- and a
  // green "Online" beside a card already flat out reads as "send it here".
  // Work sent to it will queue behind what it is doing.
  const busy = Boolean(node.busy);
  head.appendChild(el(
    "span",
    `node-status ${busy ? "status-busy" : "status-online"}`,
    busy ? "Busy" : "Online",
  ));
  card.appendChild(head);

  const badge = computeBadge(node);
  if (badge) card.appendChild(badge);

  const specs = el("div", "node-specs");
  specs.appendChild(specRow("GPU", describeGpus(node)));
  if (gpuCount > 1) specs.appendChild(specRow("GPUs pooled", gpuCount));

  const cpu = node.capabilities?.cpu || {};
  specs.appendChild(specRow("CPU", cpu.brand || "Unknown"));
  specs.appendChild(specRow("Cores", cpu.cores ?? "—"));

  // No price row. Nothing anywhere sets price_per_hour, so every card read
  // "Price/h — Free", which is not a fact about this node but an advertisement
  // for a payment system that does not exist. A row with the same value on
  // every card is either a promise or noise, and this one managed both. It
  // comes back when somebody can actually be paid.

  const queued = Number(node.queued) || 0;
  if (queued > 0) {
    specs.appendChild(specRow(
      "Queue",
      `${queued} job${queued === 1 ? "" : "s"} ahead`,
    ));
  }
  card.appendChild(specs);

  card.appendChild(el("span", "node-cta", "Send a job →"));

  const open = () => showNodeModal(node);
  card.addEventListener("click", open);
  card.addEventListener("keydown", (event) => {
    // A card that only responds to a mouse is unreachable by keyboard.
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  });

  return card;
}

function updateCount(nodes) {
  const label = document.getElementById("nodeCount");
  if (!label) return;

  if (!nodes || !nodes.length) {
    label.textContent = "";
    return;
  }

  // Count only what is actually free: totalling the compute of machines
  // already training overstates what the network can start on right now.
  const free = nodes.filter((n) => !n.busy);
  const tflops = free.reduce((sum, n) => sum + (Number(n.total_gpu_tflops) || 0), 0);

  const busyCount = nodes.length - free.length;
  const parts = [`${nodes.length} online`];
  if (busyCount) parts.push(`${busyCount} busy`);
  if (tflops > 0) parts.push(`${tflops.toFixed(1)} TFLOPS free`);

  label.textContent = parts.join(" · ");
}

export async function fetchAvailableNodes() {
  const list = document.getElementById("nodesList");
  if (!list || inFlight) return;

  inFlight = true;

  // Only show a spinner on the first paint. Replacing a populated list with
  // "Loading…" every minute makes a working page look like it keeps resetting.
  if (!list.childElementCount) {
    list.replaceChildren(notice("Looking for nodes…", null, "is-loading"));
  }

  try {
    const res = await fetch("/available-nodes");
    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const nodes = await res.json();
    if (!Array.isArray(nodes)) {
      throw new Error(nodes?.error || "Unexpected response from the coordinator.");
    }

    updateCount(nodes);

    if (!nodes.length) {
      list.replaceChildren(notice(
        "No nodes are online",
        "A contributor needs to connect a machine before work can be sent."
      ));
      return;
    }

    const cards = nodes.map(buildNodeCard);
    list.replaceChildren(...cards);
  } catch (error) {
    // This used to only reach the console, leaving an empty panel that looked
    // like "nobody is online" rather than "the request failed".
    console.error("Error loading available nodes:", error);
    updateCount([]);
    list.replaceChildren(notice(
      "Could not reach the coordinator",
      error.message,
      "is-error"
    ));
  } finally {
    inFlight = false;
  }
}

export function startNodePolling() {
  if (pollTimer) clearInterval(pollTimer);
  fetchAvailableNodes();
  pollTimer = setInterval(fetchAvailableNodes, POLL_INTERVAL_MS);
}

export function stopNodePolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}
