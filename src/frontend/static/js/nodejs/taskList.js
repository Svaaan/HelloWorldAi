// src/frontend/static/js/nodejs/taskList.js

export async function loadPendingTasks() {
    const taskListContainer = document.getElementById("taskList");
    if (!taskListContainer) {
        console.error("❌ Task list container not found.");
        return;
    }

    taskListContainer.innerHTML = "<p>Loading pending tasks...</p>";

    try {
        const res = await fetch("/get-pending-tasks");
        const tasks = await res.json();

        if (!tasks.length) {
            taskListContainer.innerHTML = "<p>No pending tasks at the moment ✅</p>";
            return;
        }

        taskListContainer.innerHTML = ""; // Clear loading text

        tasks.forEach(task => {
            const taskElement = document.createElement("div");
            taskElement.className = "task-item";
            taskElement.innerHTML = `
                <pre>${JSON.stringify(task.task_data, null, 2)}</pre>
                <div class="task-actions">
                    <button class="accept-task-btn" data-task-id="${task.task_id}">✅ Accept</button>
                    <button class="reject-task-btn" data-task-id="${task.task_id}">❌ Reject</button>
                </div>
                <hr />
            `;

            taskListContainer.appendChild(taskElement);
        });

        attachTaskActionHandlers();

    } catch (error) {
        console.error("❌ Error loading tasks:", error);
        taskListContainer.innerHTML = "<p>⚠️ Failed to load pending tasks.</p>";
    }
}

function attachTaskActionHandlers() {
    document.querySelectorAll(".accept-task-btn").forEach(button => {
        button.addEventListener("click", async () => {
            const taskId = button.getAttribute("data-task-id");
            await handleTaskAction(taskId, "accept");
        });
    });

    document.querySelectorAll(".reject-task-btn").forEach(button => {
        button.addEventListener("click", async () => {
            const taskId = button.getAttribute("data-task-id");
            await handleTaskAction(taskId, "reject");
        });
    });
}

async function handleTaskAction(taskId, action) {
    if (action === "accept") {
        // ✅ Navigate to task session page, no need to process yet
        window.location.href = `/task-session.html?taskId=${taskId}`;
        return;
    }

    const endpoint = `/reject-task/${taskId}`;

    try {
        const res = await fetch(endpoint, { method: "POST" });
        const result = await res.json();

        if (result.status === "rejected") {
            alert(`✅ Task ${taskId} rejected.`);
        } else {
            alert(`⚠️ Failed to reject task: ${result.message}`);
        }

        // Refresh task list
        loadPendingTasks();

    } catch (error) {
        console.error(`Error performing ${action} on task ${taskId}:`, error);
        alert(`❌ Error: Could not reject task.`);
    }
}


// Auto-refresh every 15 seconds
setInterval(loadPendingTasks, 15000);

// Initial load
window.addEventListener("load", loadPendingTasks);
