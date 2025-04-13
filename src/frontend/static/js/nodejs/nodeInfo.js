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
            // ✅ Fetch static node info (capabilities etc.)
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

            // ✅ Fetch dynamic usage info
            const usageRes = await fetch(`/usage`);
            const usageData = await usageRes.json();

            const connectionStatus = node.isConnected
                ? '<span class="status-connected">Connected</span>'
                : '<span class="status-disconnected">Disconnected</span>';

            const cpuUsage = usageData.cpu_usage ?? '?';
            const gpuUsage = usageData.gpu_usage ?? '?';
            const memoryUsage = usageData.memory_usage ?? '?';

            const capabilities = node.capabilities || { cpu: {}, gpu: [] };
            const gpuTflops = node.total_gpu_tflops ?? '?';

            // Tooltip calculation string for the first GPU
            const firstGpu = Array.isArray(capabilities.gpu) && capabilities.gpu.length > 0 ? capabilities.gpu[0] : null;
            const gpuTooltip = firstGpu && firstGpu.cuda_cores && firstGpu.core_clock_mhz
                ? `Formula: (CUDA Cores: ${firstGpu.cuda_cores} × Clock: ${firstGpu.core_clock_mhz} MHz × 2) ÷ 1,000,000 = ${(firstGpu.cuda_cores * firstGpu.core_clock_mhz * 2 / 1_000_000).toFixed(2)} TFLOPS`
                : 'No calculation available';

            const nodeGpuList = Array.isArray(capabilities.gpu) ? capabilities.gpu : [];
            const gpuDetailsHTML = nodeGpuList.map(gpu => {
                const gpuDetails = `
                    ${gpu.total_memory ?? '?'} MB total,
                    ${gpu.free_memory ?? '?'} MB free,
                    ${gpu.used_memory ?? '?'} MB used,
                    ${gpu.load_percentage ?? '?'}% load,
                    ${gpu.temperature ?? '?'}°C
                `;
                return `
                    <div class="node-detail">
                        <span class="node-detail-label">→ ${gpu.name || 'Unknown'}</span>
                        <span>${gpuDetails}</span>
                    </div>
                `;
            }).join('');

            const nodeDetailsHTML = `
                <div class="node-detail-group">
                    <div class="node-detail"><span class="node-detail-label">Node ID:</span> <span>${node.node_id}</span></div>
                    <div class="node-detail"><span class="node-detail-label">Country:</span> <span>${node.country || 'Unknown'}</span></div>
                    <div class="node-detail"><span class="node-detail-label">Status:</span> ${connectionStatus}</div>
                    <div class="node-detail"><span class="node-detail-label">CPU:</span> <span>${capabilities?.cpu?.brand || 'Unknown'}, ${capabilities?.cpu?.cores ?? '?'} cores</span></div>
                    <div class="node-detail"><span class="node-detail-label">CPU Usage:</span> <span>${cpuUsage}%</span></div>
                    <div class="node-detail"><span class="node-detail-label">Memory Usage:</span> <span>${memoryUsage}%</span></div>
                    <div class="node-detail"><span class="node-detail-label">GPU Usage:</span> <span>${gpuUsage}%</span></div>
                    <div class="node-detail">
                        <span class="node-detail-label">GPU Compute:</span>
                        <span title="${gpuTooltip}">${gpuTflops !== '?' ? Number(gpuTflops).toFixed(2) : '?'} TFLOPS</span>
                    </div>
                    <div class="node-detail"><span class="node-detail-label">GPUs:</span></div>
                    ${gpuDetailsHTML}
                    <hr />
                </div>
            `;

            nodeDetailsElement.innerHTML = nodeDetailsHTML;

            updateAvailabilityStatus(node.isAvailable);
            if (availabilityToggle) {
                availabilityToggle.checked = node.isAvailable;
                availabilityToggle.disabled = false;
            }

        } catch (err) {
            console.error("Error fetching node details:", err);
            showTemporaryUnavailable();
        }
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
                headers: { "Content-Type": "application/json" }
            });

            if (!res.ok) throw new Error(`Failed with status: ${res.status}`);
            const result = await res.json();
            if (result.error) throw new Error(result.error);

            updateAvailabilityStatus(isAvailable);
            showToggleMessage(isAvailable ? "Node is now available" : "Node is now unavailable", "success");

            setTimeout(() => fetchNodeInfo(), 500);
        } catch (err) {
            console.error("Error toggling availability:", err);
            if (availabilityToggle) availabilityToggle.checked = !isAvailable;
            showToggleMessage("Failed to update availability", "error");
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

    async function fetchUsageInfo() {
        if (!currentNodeId) return;

        try {
            const res = await fetch(`/usage`);
            if (!res.ok) throw new Error(`Failed to fetch usage info with status: ${res.status}`);

            const data = await res.json();

            const cpuUsagePercent = document.getElementById("cpuUsagePercent");
            const gpuUsagePercent = document.getElementById("gpuUsagePercent");
            const cpuUsageBar = document.getElementById("cpuUsageBar");
            const gpuUsageBar = document.getElementById("gpuUsageBar");

            if (cpuUsagePercent) cpuUsagePercent.textContent = `${data.cpu_usage ?? 0}%`;
            if (gpuUsagePercent) gpuUsagePercent.textContent = `${data.gpu_usage ?? 0}%`;
            if (cpuUsageBar) cpuUsageBar.style.width = `${data.cpu_usage ?? 0}%`;
            if (gpuUsageBar) gpuUsageBar.style.width = `${data.gpu_usage ?? 0}%`;

        } catch (err) {
            console.error("Error fetching usage details:", err);
        }
    }

    function manualRefresh() {
        fetchNodeInfo();
        fetchUsageInfo();
    }

    function startPeriodicRefresh() {
        setInterval(async () => {
            if (isRefreshing) return;
            isRefreshing = true;
            try {
                await Promise.all([fetchNodeInfo(), fetchUsageInfo()]);
            } catch (err) {
                console.error("Periodic refresh error:", err);
            } finally {
                isRefreshing = false;
            }
        }, 60000);
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

        setTimeout(() => {
            manualRefresh();
            startPeriodicRefresh();
        }, 2000);

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
        fetchUsageInfo,
        toggleAvailability,
        getCurrentNodeId: () => currentNodeId
    };
}
