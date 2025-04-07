// src/frontend/static/js/distribution/modalHandler.js

export function showNodeModal(node) {
    console.log("✅ showNodeModal loaded");

    const modal = document.getElementById("nodeModal");
    const modalContent = document.getElementById("nodeModalDetails") || document.getElementById("modalNodeDetails");

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

        <textarea id="taskDataInput" placeholder='Paste your task JSON here' style="width: 100%; height: 150px; margin-top: 10px;"></textarea>
        <button id="sendTaskButton" style="margin-top: 10px;">🚀 Send Task Request</button>
        <div id="taskResponseMessage" style="margin-top: 10px;"></div>
    `;

    modal.classList.remove("hidden");

    const sendTaskButton = document.getElementById("sendTaskButton");
    const taskResponseMessage = document.getElementById("taskResponseMessage");

    sendTaskButton.addEventListener("click", async () => {
        const taskDataInput = document.getElementById("taskDataInput").value;

        let taskPayload;
        try {
            taskPayload = JSON.parse(taskDataInput);
        } catch (error) {
            taskResponseMessage.style.color = "red";
            taskResponseMessage.textContent = "❌ Invalid JSON format.";
            return;
        }

        sendTaskButton.disabled = true;
        taskResponseMessage.textContent = "Sending task... ⏳";

        try {
            const response = await fetch(`/queue-task/${node.node_id}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(taskPayload)
            });

            const result = await response.json();

            if (result.status === "success") {
                taskResponseMessage.style.color = "green";
                taskResponseMessage.textContent = `✅ Task request sent: ${result.message}`;
            } else {
                taskResponseMessage.style.color = "red";
                taskResponseMessage.textContent = `⚠️ ${result.message || "Error sending task."}`;
            }

        } catch (error) {
            console.error("Error sending task:", error);
            taskResponseMessage.style.color = "red";
            taskResponseMessage.textContent = "❌ Failed to send task.";
        }

        sendTaskButton.disabled = false;
    });
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
