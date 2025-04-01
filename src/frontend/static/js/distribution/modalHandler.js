export function showNodeModal(node) {
    const modal = document.getElementById("nodeModal");
    const modalContent = document.getElementById("modalNodeDetails");

    const gpuInfo = Array.isArray(node.capabilities?.gpu)
        ? node.capabilities.gpu.map(g => g.name).join(", ")
        : (node.capabilities?.gpu?.name || "None");

    modalContent.innerHTML = `
        <h3>Node ID: ${node.node_id}</h3>
        <p><strong>CPU:</strong> ${node.capabilities?.cpu?.brand || "Unknown"} (${node.capabilities?.cpu?.cores ?? "-"} cores)</p>
        <p><strong>GPU:</strong> ${gpuInfo}</p>
        <p><strong>Status:</strong> ${node.isAvailable ? "🟢 Available" : "🔴 Unavailable"}</p>
        <p><strong>Available time:</strong> ${node.available_time || "N/A"}</p>
        <p><strong>Price per hour:</strong> ${node.price_per_hour != null ? node.price_per_hour + " SEK/h" : "N/A"}</p>
        <p><strong>Price per day:</strong> ${node.price_per_hour != null ? (node.price_per_hour * 24).toFixed(2) + " SEK/day" : "N/A"}</p>
    `;

    modal.classList.remove("hidden");
}

export function initModalCloseHandler() {
    const closeBtn = document.getElementById("modalClose");
    if (closeBtn) {
        closeBtn.addEventListener("click", () => {
            document.getElementById("nodeModal").classList.add("hidden");
        });
    }

    // Optional: ESC key to close modal
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            document.getElementById("nodeModal").classList.add("hidden");
        }
    });
}
