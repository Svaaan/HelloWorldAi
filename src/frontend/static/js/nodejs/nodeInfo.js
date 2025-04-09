// File: /static/js/nodejs/nodeInfo.js

export function initNodeInfoManager() {
    let currentNodeId = localStorage.getItem("currentNodeId");
    let retryInterval = null;
    let isRefreshing = false;

    async function fetchNodeInfo(retryCount = 0) {
        const nodeDetailsElement = document.getElementById("nodeDetails");
        const availabilityToggle = document.getElementById("availabilityToggle");
    
        if (!currentNodeId) {
            nodeDetailsElement.innerHTML =
                "<p class='status-disconnected'>Access denied: No node ID in localStorage.</p>";
            if (availabilityToggle) availabilityToggle.disabled = true;
            return;
        }
    
        try {
            // Fetch node info from database
            const nodeRes = await fetch(`/nodes?node_id=${currentNodeId}`);
            const nodeData = await nodeRes.json();
    
            // Safety check
            if (!Array.isArray(nodeData) || nodeData.length === 0) {
                console.warn(`Node not found (attempt ${retryCount + 1}). Retrying in 2 seconds...`, nodeData);
                if (retryCount < 5) {  // 🔁 Try up to 5 times
                    setTimeout(() => fetchNodeInfo(retryCount + 1), 2000);
                } else {
                    showTemporaryUnavailable();
                }
                return;
            }
    
            const node = nodeData.find(n => n.node_id === currentNodeId);
    
            if (!node) {
                console.warn(`Node data missing (attempt ${retryCount + 1}). Retrying in 2 seconds...`, nodeData);
                if (retryCount < 5) {
                    setTimeout(() => fetchNodeInfo(retryCount + 1), 2000);
                } else {
                    showTemporaryUnavailable();
                }
                return;
            }
    
            // Fetch live usage info (CPU & GPU usage)
            const usageRes = await fetch("/usage");
            const usageData = await usageRes.json();
    
            clearInterval(retryInterval); // ✅ Clear retry if successful
    
            const connectionStatus = node.isConnected
                ? '<span class="status-connected">Connected</span>'
                : '<span class="status-disconnected">Disconnected</span>';
    
            const cpuUsage = usageData.cpu_usage ?? '?';
            const gpuUsage = usageData.gpu_usage ?? '?';
            const memoryUsage = usageData.memory_usage ?? '?';
    
            const nodeDetailsHTML = `
                <div class="node-detail-group">
                    <div class="node-detail">
                        <span class="node-detail-label">Node ID:</span> <span>${node.node_id}</span>
                    </div>
                    <div class="node-detail">
                        <span class="node-detail-label">Country:</span> <span>${node.country || 'Unknown'}</span>
                    </div>
                    <div class="node-detail">
                        <span class="node-detail-label">Status:</span> ${connectionStatus}
                    </div>
                    <div class="node-detail">
                        <span class="node-detail-label">CPU:</span>
                        <span>${node.capabilities?.cpu?.brand || 'Unknown'}, ${node.capabilities?.cpu?.cores ?? '?'} cores</span>
                    </div>
                    <div class="node-detail">
                        <span class="node-detail-label">CPU Usage:</span> <span>${cpuUsage}%</span>
                    </div>
                    <div class="node-detail">
                        <span class="node-detail-label">Memory Usage:</span> <span>${memoryUsage}%</span>
                    </div>
                    <div class="node-detail">
                        <span class="node-detail-label">GPU Usage:</span> <span>${gpuUsage}%</span>
                    </div>
                    <div class="node-detail">
                        <span class="node-detail-label">GPUs:</span>
                    </div>
                    ${(Array.isArray(node.capabilities?.gpu) ? node.capabilities.gpu : [node.capabilities.gpu || {}])
                        .map(gpu => `
                            <div class="node-detail">
                                <span class="node-detail-label">→ ${gpu.name || 'Unknown'}</span>
                                <span>
                                    ${gpu.total_memory ?? '?'} MB total,
                                    ${gpu.free_memory ?? '?'} MB free,
                                    ${gpu.used_memory ?? '?'} MB used,
                                    ${gpu.load_percentage ?? '?'}% load,
                                    ${gpu.temperature ?? '?'}°C
                                </span>
                            </div>
                        `).join('')}
                    <hr />
                </div>
            `;
    
            nodeDetailsElement.innerHTML = nodeDetailsHTML;
    
            updateAvailabilityStatus(node.isAvailable);
    
            if (availabilityToggle) {
                availabilityToggle.checked = node.isAvailable;
            }
    
        } catch (err) {
            console.error("Error fetching node details:", err);
            showTemporaryUnavailable();
        }
    }
    
    function showTemporaryUnavailable() {
        const nodeDetailsElement = document.getElementById("nodeDetails");
        const availabilityToggle = document.getElementById("availabilityToggle");
        
        if (nodeDetailsElement) {
            nodeDetailsElement.innerHTML = 
                "<p class='status-disconnected'>Node temporarily unavailable. Retrying connection…</p>";
        }
        
        if (availabilityToggle) {
            availabilityToggle.disabled = true;
        }

        if (!retryInterval) {
            retryInterval = setInterval(fetchNodeInfo, 5000); // ✅ Retry every 5s
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
            const res = await fetch(`/toggle-availability/${currentNodeId}`, { method: "PATCH" });
            const result = await res.json();

            if (!res.ok || result.error) {
                throw new Error(result.error || `Failed with status: ${res.status}`);
            }

            updateAvailabilityStatus(isAvailable);
            showToggleMessage(isAvailable ? "Node is now available" : "Node is now unavailable", "success");

        } catch (err) {
            console.error(`Error toggling availability:`, err);
            if (availabilityToggle) availabilityToggle.checked = !isAvailable;
            showToggleMessage("Failed to update availability", "error");
        }

        if (toggleProcessing) toggleProcessing.style.display = "none";
        if (availabilityToggle) availabilityToggle.disabled = false;
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
        try {
            const res = await fetch("/usage");
            const data = await res.json();
            
            const cpuUsagePercent = document.getElementById("cpuUsagePercent");
            const gpuUsagePercent = document.getElementById("gpuUsagePercent");
            const cpuUsageBar = document.getElementById("cpuUsageBar");
            const gpuUsageBar = document.getElementById("gpuUsageBar");
            
            if (cpuUsagePercent) cpuUsagePercent.textContent = `${data.cpu_usage || 0}%`;
            if (gpuUsagePercent) gpuUsagePercent.textContent = `${data.gpu_usage || 0}%`;
            if (cpuUsageBar) cpuUsageBar.style.width = `${data.cpu_usage || 0}%`;
            if (gpuUsageBar) gpuUsageBar.style.width = `${data.gpu_usage || 0}%`;
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
            }
            isRefreshing = false;
        }, 60000); // Refresh every minute
    }

    // Initialize event listeners specific to node info
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
    
        setTimeout(() => {
            manualRefresh();
            startPeriodicRefresh();
        }, 2000); 
    
        // Optional: catch before unload
        window.addEventListener("beforeunload", (event) => {
            if (!currentNodeId) return;
            try {
                navigator.sendBeacon(`/toggle-availability/${currentNodeId}`);
            } catch (err) {
                console.warn("Error setting availability to false on unload:", err);
            }
        });

        manualRefresh();
        startPeriodicRefresh();
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