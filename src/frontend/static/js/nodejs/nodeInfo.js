import { authHeaders } from "../connect/nodeSession.js";

const USAGE_POLL_MS = 2000;
const RETRY_POLL_MS = 5000;
const MAX_RETRIES = 5;

/** Build a labelled detail row without going through innerHTML.
 *  Node names, countries and GPU strings come from other people's machines,
 *  so they are inserted as text, never as markup. */
function detailRow(label, valueNode) {
    const row = document.createElement("div");
    row.className = "node-detail";

    const key = document.createElement("span");
    key.className = "node-detail-label";
    key.textContent = label;

    row.append(key);
    if (valueNode) row.append(valueNode);
    return row;
}

function textSpan(value, className) {
    const span = document.createElement("span");
    if (className) span.className = className;
    span.textContent = value;
    return span;
}

export function initNodeInfoManager() {
    let currentNodeId = localStorage.getItem("currentNodeId");

    // Exactly one of each timer may exist. Both used to be re-armed without
    // being cleared, so every refresh added another /usage poller.
    let usageTimer = null;
    let retryTimer = null;
    let currentGpuList = [];

    // --- usage polling ---------------------------------------------------

    function stopUsagePolling() {
        if (usageTimer) {
            clearInterval(usageTimer);
            usageTimer = null;
        }
    }

    function startUsagePolling(gpuList = []) {
        currentGpuList = Array.isArray(gpuList) ? gpuList : [];
        stopUsagePolling();                 // never stack pollers
        if (!currentNodeId || document.hidden) return;

        fetchUsage();
        usageTimer = setInterval(fetchUsage, USAGE_POLL_MS);
    }

    async function fetchUsage() {
        try {
            const res = await fetch("/usage");
            if (!res.ok) throw new Error(`usage returned ${res.status}`);
            const data = await res.json();

            const cpu = Number(data.cpu_usage ?? 0);
            setMeter("cpuUsagePercent", "cpuUsageBar", cpu);

            const gpuData = Array.isArray(data.gpu_data) ? data.gpu_data : [];
            const loads = gpuData
                .map(g => Number.parseFloat(g.gpu_usage))
                .filter(v => Number.isFinite(v));
            const avgGpu = loads.length
                ? Math.round(loads.reduce((a, b) => a + b, 0) / loads.length)
                : 0;
            setMeter("gpuUsagePercent", "gpuUsageBar", avgGpu);

            renderGpuDetails(gpuData);
        } catch (err) {
            console.error("Error fetching usage details:", err);
        }
    }

    function setMeter(labelId, barId, value) {
        const rounded = Math.max(0, Math.min(100, Math.round(value)));
        const label = document.getElementById(labelId);
        const bar = document.getElementById(barId);
        if (label) label.textContent = `${rounded}%`;
        if (bar) bar.style.width = `${rounded}%`;
    }

    function renderGpuDetails(liveGpus) {
        const container = document.getElementById("gpuDetailsContainer");
        if (!container || currentGpuList.length === 0) return;

        container.replaceChildren();

        currentGpuList.forEach((gpu, index) => {
            const live = liveGpus[index] || {};
            const temp = live.gpu_temperature ?? gpu.temperature;
            const critical = live.critical_temperature ?? gpu.temperature_critical ?? 85;
            const load = live.gpu_usage ?? gpu.load_percentage;

            const value = document.createElement("span");
            value.className = "gpu-line";

            const specs = [
                gpu.total_memory != null ? `${gpu.total_memory} MB total` : null,
                gpu.free_memory != null ? `${gpu.free_memory} MB free` : null,
                load != null ? `${load}% load` : null,
            ].filter(Boolean).join(" · ");

            value.append(document.createTextNode(specs ? `${specs} · ` : ""));

            const tempSpan = document.createElement("strong");
            tempSpan.className = "gpu-temp " + temperatureClass(temp, critical);
            tempSpan.textContent = temp != null ? `${temp}°C` : "—";
            value.append(tempSpan);

            container.append(detailRow(gpu.name || "Unknown GPU", value));
        });
    }

    function temperatureClass(temp, critical) {
        if (temp == null) return "is-unknown";
        if (temp >= critical - 5) return "is-critical";
        if (temp >= 70) return "is-warm";
        return "is-cool";
    }

    // --- node details ----------------------------------------------------

    async function fetchNodeInfo(retryCount = 0) {
        const details = document.getElementById("nodeDetails");
        const toggle = document.getElementById("availabilityToggle");

        if (!currentNodeId) {
            showNotice(
                "No node connected yet",
                "Register a node, or reconnect one with its key file.",
                false,
                { label: "Go to connect", href: "/connect" }
            );
            if (toggle) toggle.disabled = true;
            return;
        }

        try {
            const res = await fetch(`/nodes?node_id=${encodeURIComponent(currentNodeId)}`);
            const data = await res.json();
            const node = Array.isArray(data)
                ? data.find(n => n.node_id === currentNodeId)
                : null;

            if (!node) return handleRetry(retryCount);

            if (retryTimer) {
                clearInterval(retryTimer);
                retryTimer = null;          // nulled, so retries can re-arm later
            }

            renderNodeDetails(details, node);

            updateAvailabilityStatus(node.isAvailable);
            if (toggle) {
                toggle.checked = Boolean(node.isAvailable);
                toggle.disabled = false;
            }

            startUsagePolling(node.capabilities?.gpu || []);
        } catch (err) {
            console.error("Error fetching node details:", err);
            showTemporaryUnavailable();
        }
    }

    function renderNodeDetails(container, node) {
        const capabilities = node.capabilities || { cpu: {}, gpu: [] };
        const cpu = capabilities.cpu || {};
        const tflops = node.total_gpu_tflops;

        const group = document.createElement("div");
        group.className = "node-detail-group";

        const id = textSpan(node.node_id, "mono");
        group.append(detailRow("Node ID", id));
        group.append(detailRow("Country", textSpan(node.country || "Unknown")));

        const status = document.createElement("span");
        status.append(textSpan(
            node.isConnected ? "Connected" : "Disconnected",
            node.isConnected ? "status-connected" : "status-disconnected"
        ));

        if (!node.isConnected) {
            const why = document.createElement("span");
            why.className = "status-hint";
            why.textContent =
                "The node has not reported in for over 5 minutes. If it restarted, " +
                "reconnect it from the connect page to hand it a new session.";
            status.append(why);
        }
        group.append(detailRow("Status", status));

        const cpuText = [cpu.brand || "Unknown", cpu.cores != null ? `${cpu.cores} cores` : null]
            .filter(Boolean).join(" · ");
        group.append(detailRow("CPU", textSpan(cpuText)));

        const measured = capabilities.measured_tflops;
        const isMeasured = Number.isFinite(Number(measured));
        const shown = isMeasured ? Number(measured) : Number(tflops);

        const compute = document.createElement("span");
        compute.append(document.createTextNode(
            Number.isFinite(shown) ? `${shown.toFixed(2)} TFLOPS` : "Not measured"
        ));

        // Spec-sheet peak and benchmarked throughput differ a lot, so say which.
        const tag = document.createElement("span");
        tag.className = isMeasured ? "compute-tag is-measured" : "compute-tag";
        tag.textContent = isMeasured ? "measured" : "theoretical";
        compute.append(tag);

        const theoretical = capabilities.theoretical_tflops ?? tflops;
        if (isMeasured && Number.isFinite(Number(theoretical))) {
            compute.title =
                `Benchmarked ${Number(measured).toFixed(2)} TFLOPS; ` +
                `spec-sheet peak is ${Number(theoretical).toFixed(2)}.`;
        }
        group.append(detailRow("GPU compute", compute));

        const gpuHeading = document.createElement("div");
        gpuHeading.className = "node-detail-heading";
        gpuHeading.textContent = "GPUs";
        group.append(gpuHeading);

        const gpuContainer = document.createElement("div");
        gpuContainer.id = "gpuDetailsContainer";
        group.append(gpuContainer);

        container.replaceChildren(group);
        const warning = keyBackupWarning();
        if (warning) container.prepend(warning);
    }

    // A node's keypair lives in this browser's localStorage, and the download
    // at registration is a button somebody can walk past. Clearing site data
    // then leaves a machine that is still online and still taking jobs, with
    // no way for its owner to see or control it -- which is exactly what
    // happened while testing this.
    function keyBackupWarning() {
        let backedUp = false;
        try {
            const stored = localStorage.getItem("nodeKeyBackedUp");
            const publicKey = localStorage.getItem("nodePublicKeyBase64");
            backedUp = Boolean(stored) && Boolean(publicKey)
                && stored === publicKey.slice(0, 12);
        } catch (e) {
            backedUp = false;         // keep asking rather than assume
        }

        if (backedUp) return null;

        const box = document.createElement("div");
        box.className = "key-warning";

        const title = document.createElement("strong");
        title.textContent = "This node's key exists only in this browser.";
        box.append(title);

        const text = document.createElement("p");
        text.textContent =
            "Clear your site data and you lose control of this machine — it "
            + "keeps running and taking jobs, and this page can no longer see "
            + "it. There is no reset. Save the key file from the connect page.";
        box.append(text);

        const link = document.createElement("a");
        link.className = "panel-notice-action";
        link.href = "/connect";
        link.textContent = "Save the key file →";
        box.append(link);

        return box;
    }

    function showNotice(title, detail, isError, action) {
        const container = document.getElementById("nodeDetails");
        if (!container) return;

        const box = document.createElement("div");
        box.className = isError ? "panel-notice is-error" : "panel-notice";

        const strong = document.createElement("strong");
        strong.textContent = title;
        box.append(strong);

        if (detail) {
            const p = document.createElement("span");
            p.textContent = detail;
            box.append(p);
        }

        if (action) {
            const link = document.createElement("a");
            link.className = "panel-notice-action";
            link.href = action.href;
            link.textContent = action.label;
            box.append(link);
        }

        container.replaceChildren(box);
    }

    function handleRetry(retryCount) {
        if (retryCount < MAX_RETRIES) {
            setTimeout(() => fetchNodeInfo(retryCount + 1), 2000);
        } else {
            showTemporaryUnavailable();
        }
    }

    function showTemporaryUnavailable() {
        // The offer to reconnect matters when the saved node is gone for good
        // (deleted, or a database that has since been reset): the header hides
        // its Connect link once a node id is stored, so without this there is
        // no obvious way out of a node that will never come back.
        showNotice("Node unavailable", "Retrying…", true,
                   { label: "Connect a different node", href: "/connect" });

        const toggle = document.getElementById("availabilityToggle");
        if (toggle) toggle.disabled = true;

        stopUsagePolling();     // nothing to poll while the node is gone
        if (!retryTimer) {
            retryTimer = setInterval(() => fetchNodeInfo(), RETRY_POLL_MS);
        }
    }

    // --- availability ----------------------------------------------------

    function updateAvailabilityStatus(isAvailable) {
        const el = document.getElementById("availabilityStatus");
        if (!el) return;
        el.replaceChildren(textSpan(
            isAvailable ? "Available" : "Not available",
            isAvailable ? "status-available" : "status-unavailable"
        ));
    }

    async function toggleAvailability(isAvailable) {
        if (!currentNodeId) {
            showToggleMessage("No node connected", "error");
            return;
        }

        const processing = document.getElementById("toggleProcessing");
        const toggle = document.getElementById("availabilityToggle");

        if (processing) processing.style.display = "flex";
        if (toggle) toggle.disabled = true;

        try {
            const res = await fetch(`/toggle-availability/${encodeURIComponent(currentNodeId)}`, {
                method: "PATCH",
                headers: authHeaders({ "Content-Type": "application/json" })
            });

            if (res.status === 401 || res.status === 403) {
                throw new Error("Session expired. Reconnect this node with its key file.");
            }
            if (!res.ok) throw new Error(`Failed with status: ${res.status}`);

            const result = await res.json();
            if (result.error) throw new Error(result.error);

            updateAvailabilityStatus(isAvailable);
            showToggleMessage(
                isAvailable ? "Node is now available for work" : "Node is now unavailable",
                "success"
            );

            setTimeout(() => fetchNodeInfo(), 500);
        } catch (err) {
            console.error("Error toggling availability:", err);
            if (toggle) toggle.checked = !isAvailable;
            showToggleMessage(err.message || "Failed to update availability", "error");
        } finally {
            if (processing) processing.style.display = "none";
            if (toggle) toggle.disabled = false;
        }
    }

    let toggleMessageTimer = null;
    function showToggleMessage(message, type) {
        const el = document.getElementById("toggleStatusMessage");
        if (!el) return;

        el.textContent = message;
        el.className = `toggle-status-message ${type === "success" ? "success-message" : "error-message"}`;

        clearTimeout(toggleMessageTimer);
        toggleMessageTimer = setTimeout(() => {
            el.textContent = "";
            el.className = "toggle-status-message";
        }, 4000);
    }

    // --- lifecycle -------------------------------------------------------

    function manualRefresh() {
        fetchNodeInfo();
    }

    function init() {
        const toggle = document.getElementById("availabilityToggle");
        if (toggle) {
            toggle.addEventListener("change", e => toggleAvailability(e.target.checked));
        }

        const refreshButton = document.getElementById("refreshNodeInfo");
        if (refreshButton) refreshButton.addEventListener("click", manualRefresh);

        // Stop polling a page nobody is looking at, and catch up on return.
        document.addEventListener("visibilitychange", () => {
            if (document.hidden) {
                stopUsagePolling();
            } else if (currentNodeId) {
                startUsagePolling(currentGpuList);
            }
        });

        window.addEventListener("beforeunload", stopUsagePolling);

        fetchNodeInfo();
    }

    init();

    return {
        manualRefresh,
        fetchNodeInfo,
        toggleAvailability,
        stopUsagePolling,
        getCurrentNodeId: () => currentNodeId
    };
}
