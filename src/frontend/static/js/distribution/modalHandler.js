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

        <button id="sendTaskButton">🚀 Send Test Task to Node</button>
        <div id="taskResponseMessage" style="margin-top: 10px;"></div>
    `;

    modal.classList.remove("hidden");

    const sendTaskButton = document.getElementById("sendTaskButton");
    const taskResponseMessage = document.getElementById("taskResponseMessage");

    sendTaskButton.addEventListener("click", async () => {
        sendTaskButton.disabled = true;
        taskResponseMessage.textContent = "Sending task... ⏳";

        try {
            const response = await fetch(`/execute-task/${node.node_id}`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    task_type: "llm_training",
                    model_name: "gpt2",
                    data: {
                        texts: ["Hello world", "Distributed computing is here!"]
                    },
                    hyperparameters: {
                        learning_rate: 0.001,
                        epochs: 2
                    },
                    response_required: true
                })
            });

            const result = await response.json();

            if (result.status === "success") {
                taskResponseMessage.style.color = "green";
                taskResponseMessage.textContent = `✅ Task processed: ${result.message}`;
            } else {
                taskResponseMessage.style.color = "red";
                taskResponseMessage.textContent = `⚠️ ${result.message || "Error processing task."}`;
            }

        } catch (error) {
            console.error("Error sending task:", error);
            taskResponseMessage.style.color = "red";
            taskResponseMessage.textContent = "❌ Failed to send task to node.";
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
