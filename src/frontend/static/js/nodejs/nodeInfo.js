import { authHeaders } from "../connect/nodeSession.js";

export function initNodeInfoManager() {
    let currentNodeId = localStorage.getItem("currentNodeId");
    let retryInterval = null;
    let isRefreshing = false;

    async function fetchNodeInfo(retryCount = 0) {
        const nodeDetailsElement = document.getElementById("nodeDetails");
        const availabilityToggle = document.getElementById("availabilityToggle");

        if (!currentNodeId) {
            nodeDetailsElement.innerHTML = "<p class='status-disconnected'>Access denied: No node ID in localStorage.</p>";
            if (availabilityToggle) availabilityToggle.disabled = true;
            return;
        }

        try {
            // ✅ Fetch static node info (only when user manually refreshes)
            const nodeRes = await fetch(`/nodes?node_id=${currentNodeId}`);
            const nodeData = await nodeRes.json();

            if (!Array.isArray(nodeData) || nodeData.length === 0) {
                handleRetry(retryCount, fetchNodeInfo);
                return;
            }

            const node = nodeData.find(n => n.node_id === currentNodeId);
            if (!node) {
                handleRetry(retryCount, fetchNodeInfo);
                return;
            }

            clearInterval(retryInterval);

            const connectionStatus = node.isConnected
                ? '<span class="status-connected">Connected</span>'
                : '<span class="status-disconnected">Disconnected</span>';

            const capabilities = node.capabilities || { cpu: {}, gpu: [] };
            const gpuTflops = node.total_gpu_tflops ?? '?';

            // GPU tooltip (static calculation)
            const firstGpu = capabilities.gpu?.[0];
            const gpuTooltip = firstGpu && firstGpu.cuda_cores && firstGpu.core_clock_mhz
                ? `Formula: (CUDA Cores: ${firstGpu.cuda_cores} × Clock: ${firstGpu.core_clock_mhz} MHz × 2) ÷ 1,000,000 = ${(firstGpu.cuda_cores * firstGpu.core_clock_mhz * 2 / 1_000_000).toFixed(2)} TFLOPS`
                : 'No calculation available';

            const nodeDetailsHTML = `
                <div class="node-detail-group">
                    <div class="node-detail"><span class="node-detail-label">Node ID:</span> <span>${node.node_id}</span></div>
                    <div class="node-detail"><span class="node-detail-label">Country:</span> <span>${node.country || 'Unknown'}</span></div>
                    <div class="node-detail"><span class="node-detail-label">Status:</span> ${connectionStatus}</div>
                    <div class="node-detail"><span class="node-detail-label">CPU:</span> <span>${capabilities.cpu.brand || 'Unknown'}, ${capabilities.cpu.cores ?? '?'} cores</span></div>
                    <div class="node-detail">
                        <span class="node-detail-label">GPU Compute:</span>
                        <span title="${gpuTooltip}">${gpuTflops !== '?' ? Number(gpuTflops).toFixed(2) : '?'} TFLOPS</span>
                    </div>
                    <div class="node-detail"><span class="node-detail-label">GPUs:</span></div>
                    <div id="gpuDetailsContainer">Loading...</div>
                    <hr />
                </div>
            `;

            nodeDetailsElement.innerHTML = nodeDetailsHTML;

            updateAvailabilityStatus(node.isAvailable);
            if (availabilityToggle) {
                availabilityToggle.checked = node.isAvailable;
                availabilityToggle.disabled = false;
            }

            // ✅ Start fast GPU usage live update
            startUsageFastUpdate(capabilities.gpu);

        } catch (err) {
            console.error("Error fetching node details:", err);
            showTemporaryUnavailable();
        }
    }

    function startUsageFastUpdate(nodeGpuList = []) {
        if (!currentNodeId) return;

        async function fetchUsageInfoFast() {
            try {
                const res = await fetch(`/usage`);
                if (!res.ok) throw new Error(`Failed to fetch usage info with status: ${res.status}`);

                const data = await res.json();

                // CPU Update
                const cpuUsagePercent = document.getElementById("cpuUsagePercent");
                const cpuUsageBar = document.getElementById("cpuUsageBar");
                const cpuUsage = data.cpu_usage ?? 0;
                if (cpuUsagePercent) cpuUsagePercent.textContent = `${cpuUsage}%`;
                if (cpuUsageBar) cpuUsageBar.style.width = `${cpuUsage}%`;

                // GPU Update
                const gpuUsagePercent = document.getElementById("gpuUsagePercent");
                const gpuUsageBar = document.getElementById("gpuUsageBar");

                const usageGpuList = Array.isArray(data.gpu_data) ? data.gpu_data : [];
                const validGpuUsages = usageGpuList
                    .map(gpu => parseInt(gpu.gpu_usage))
                    .filter(val => !isNaN(val));

                const averageGpuUsage = validGpuUsages.length > 0
                    ? Math.round(validGpuUsages.reduce((sum, val) => sum + val, 0) / validGpuUsages.length)
                    : 0;

                if (gpuUsagePercent) gpuUsagePercent.textContent = `${averageGpuUsage}%`;
                if (gpuUsageBar) gpuUsageBar.style.width = `${averageGpuUsage}%`;

                // Detailed GPU Info
                const gpuDetailsContainer = document.getElementById("gpuDetailsContainer");
                if (gpuDetailsContainer && nodeGpuList.length > 0) {
                    nodeGpuList.forEach((gpu, index) => {
                        const liveGpu = usageGpuList[index];
                        if (liveGpu) {
                            gpu.load_percentage = liveGpu.gpu_usage;
                            gpu.temperature = liveGpu.gpu_temperature;
                            gpu.temperature_critical = liveGpu.critical_temperature;
                        }
                    });

                    const gpuDetailsHTML = nodeGpuList.map(gpu => {
                        const temp = gpu.temperature ?? '?';
                        const tempCritical = gpu.temperature_critical ?? 'N/A';
                        const temperatureColor = temp === '?' ? 'var(--text-faint)'
                            : temp < 50 ? 'var(--success)'
                            : temp < 70 ? 'var(--warning)'
                            : 'var(--danger)';

                        return `
                            <div class="node-detail">
                                <span class="node-detail-label">→ ${gpu.name || 'Unknown'}</span>
                                <span>
                                    ${gpu.total_memory ?? '?'} MB total,
                                    ${gpu.free_memory ?? '?'} MB free,
                                    ${gpu.used_memory ?? '?'} MB used,
                                    ${gpu.load_percentage ?? '?'}% load,
                                    <span style="color:${temperatureColor}; font-weight:bold;">${temp}°C</span>
                                    <span style="color:var(--text-faint);">(Critical: ${tempCritical}°C)</span>
                                </span>
                            </div>
                        `;
                    }).join('');

                    gpuDetailsContainer.innerHTML = gpuDetailsHTML;
                }

            } catch (err) {
                console.error("Error fetching usage details:", err);
            }
        }

        // Faster interval for smooth updates 🚀
        setInterval(fetchUsageInfoFast, 2000);
    }

    function handleRetry(retryCount, callback) {
        if (retryCount < 5) {
            setTimeout(() => callback(retryCount + 1), 2000);
        } else {
            showTemporaryUnavailable();
        }
    }

    function showTemporaryUnavailable() {
        const nodeDetailsElement = document.getElementById("nodeDetails");
        const availabilityToggle = document.getElementById("availabilityToggle");

        if (nodeDetailsElement) {
            nodeDetailsElement.innerHTML = "<p class='status-disconnected'>Node temporarily unavailable. Retrying connection…</p>";
        }

        if (availabilityToggle) {
            availabilityToggle.disabled = true;
        }

        if (!retryInterval) {
            retryInterval = setInterval(fetchNodeInfo, 5000);
        }
    }

    function updateAvailabilityStatus(isAvailable) {
        const availabilityStatus = document.getElementById("availabilityStatus");
        if (availabilityStatus) {
            availabilityStatus.innerHTML = isAvailable
                ? '<span class="status-available">Available</span>'
                : '<span class="status-unavailable">Not Available</span>';
        }
    }

    async function toggleAvailability(isAvailable) {
        if (!currentNodeId) {
            showToggleMessage("No node connected", "error");
            return;
        }

        const toggleProcessing = document.getElementById("toggleProcessing");
        const availabilityToggle = document.getElementById("availabilityToggle");

        if (toggleProcessing) toggleProcessing.style.display = "block";
        if (availabilityToggle) availabilityToggle.disabled = true;

        try {
            const res = await fetch(`/toggle-availability/${currentNodeId}`, {
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
            showToggleMessage(isAvailable ? "Node is now available" : "Node is now unavailable", "success");

            setTimeout(() => fetchNodeInfo(), 500);
        } catch (err) {
            console.error("Error toggling availability:", err);
            if (availabilityToggle) availabilityToggle.checked = !isAvailable;
            showToggleMessage(err.message || "Failed to update availability", "error");
        } finally {
            if (toggleProcessing) toggleProcessing.style.display = "none";
            if (availabilityToggle) availabilityToggle.disabled = false;
        }
    }

    function showToggleMessage(message, type) {
        const messageElement = document.getElementById("toggleStatusMessage");
        if (!messageElement) return;

        messageElement.textContent = message;
        messageElement.className = `toggle-status-message ${type === "success" ? "success-message" : "error-message"}`;
        setTimeout(() => {
            messageElement.textContent = "";
            messageElement.className = "toggle-status-message";
        }, 3000);
    }

    function manualRefresh() {
        fetchNodeInfo();
    }

    function init() {
        if (!currentNodeId) {
            currentNodeId = localStorage.getItem("currentNodeId");
            if (!currentNodeId) console.warn("No node ID found in localStorage.");
        }

        const availabilityToggle = document.getElementById("availabilityToggle");
        if (availabilityToggle) {
            availabilityToggle.addEventListener("change", (e) => {
                toggleAvailability(e.target.checked);
            });
        }

        const refreshButton = document.getElementById("refreshNodeInfo");
        if (refreshButton) {
            refreshButton.addEventListener("click", manualRefresh);
        }

        // Manual fetch once on load
        setTimeout(() => {
            manualRefresh();
        }, 1000);

        window.addEventListener("beforeunload", () => {
            if (!currentNodeId) return;
            try {
                navigator.sendBeacon(`/toggle-availability/${currentNodeId}`);
            } catch (err) {
                console.warn("Error setting availability to false on unload:", err);
            }
        });
    }

    init();

    return {
        manualRefresh,
        fetchNodeInfo,
        toggleAvailability,
        getCurrentNodeId: () => currentNodeId
    };
}
