import { showNodeModal } from "./modalhandler.js";

export async function fetchAvailableNodes() {
    try {
        const res = await fetch("/nodes");
        const nodes = await res.json();

        const availableNodes = nodes.filter(node => node.isConnected && node.isAvailable);
        const nodesList = document.getElementById("nodesList");
        nodesList.innerHTML = "";

        if (availableNodes.length === 0) {
            nodesList.innerHTML = `<div class="empty-message">
                <p>No nodes currently available.</p>
            </div>`;
            return;
        }

        availableNodes.forEach(node => {
            const gpuInfo = Array.isArray(node.capabilities?.gpu)
                ? node.capabilities.gpu.map(g => g.name).join(", ")
                : (node.capabilities?.gpu?.name || "None");

            const nodeHTML = `
                <div class="node-item" data-node-id="${node.node_id}">
                    <div class="node-header">
                        <span class="node-id">${node.node_id}</span>
                        <span class="node-status status-online">Online</span>
                    </div>
                    <div class="node-specs">
                        <div class="node-spec">
                            <span class="spec-label">CPU:</span>
                            <span>${node.capabilities?.cpu?.brand || "Unknown"}</span>
                        </div>
                        <div class="node-spec">
                            <span class="spec-label">Cores:</span>
                            <span>${node.capabilities?.cpu?.cores ?? "-"}</span>
                        </div>
                        <div class="node-spec">
                            <span class="spec-label">GPU:</span>
                            <span>${(gpuInfo && gpuInfo !== "No GPU") ? gpuInfo : "None"}</span>
                        </div>
                        <div class="node-spec">
                            <span class="spec-label">Price/h:</span>
                            <span>${node.price_per_hour != null ? node.price_per_hour + " SEK/h" : "N/A"}</span>
                        </div>
                    </div>
                </div>
            `;
            nodesList.insertAdjacentHTML("beforeend", nodeHTML);
        });

        document.querySelectorAll(".node-item").forEach(item => {
            item.addEventListener("click", () => {
                const nodeId = item.dataset.nodeId;
                const node = availableNodes.find(n => n.node_id === nodeId);
                if (node) {
                    showNodeModal(node);
                }
            });
        });

    } catch (error) {
        console.error("Error loading available nodes:", error);
    }
}
