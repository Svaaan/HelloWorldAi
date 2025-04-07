async function loadTaskDetails(taskId) {
    const response = await fetch('/get-pending-tasks');
    const tasks = await response.json();
    const task = tasks.find(t => t.task_id === taskId);

    const taskDetails = document.getElementById('taskDetails');
    const statusElement = document.getElementById('taskStatus');

    if (!task) {
        taskDetails.innerHTML = `<p>⚠️ Task not found or already processed.</p>`;
        statusElement.innerHTML = `<span style="color: blue;">Processing or Completed ✅</span>`;
        document.getElementById("startTaskButton").disabled = true;
        return;
    }

    taskDetails.innerHTML = `
        <p><strong>Task ID:</strong> ${task.task_id}</p>
        <p><strong>Task Type:</strong> ${task.task_data.task_type}</p>
        <p><strong>Model:</strong> ${task.task_data.model_name}</p>
        <p><strong>Hyperparameters:</strong> ${JSON.stringify(task.task_data.hyperparameters)}</p>
    `;

    statusElement.innerHTML = `<span style="color: orange;">Pending</span>`;
}

async function startTask(taskId) {
    try {
        const response = await fetch(`/process-task/${taskId}`, { method: "POST" });
        const result = await response.json();

        const statusElement = document.getElementById("taskStatus");

        if (result.status === "processing") {
            statusElement.innerHTML = `<span style="color: green;">✅ Task started: ${result.message}</span>`;
            document.getElementById("startTaskButton").disabled = true;

            // Start polling for status updates
            startStatusPolling(taskId);
        } else {
            statusElement.innerHTML = `<span style="color: red;">❌ Failed to start: ${result.message}</span>`;
        }
    } catch (error) {
        console.error("Error starting task:", error);
    }
}

function startStatusPolling(taskId) {
    const statusElement = document.getElementById("taskStatus");

    setInterval(async () => {
        const response = await fetch('/get-pending-tasks');
        const tasks = await response.json();
        const task = tasks.find(t => t.task_id === taskId);

        if (!task) {
            statusElement.innerHTML = `<span style="color: blue;">🚀 Task is now Processing or Completed!</span>`;
        } else {
            statusElement.innerHTML = `<span style="color: orange;">Pending</span>`;
        }
    }, 3000);
}

// Read taskId from URL (✅ correct param!)
const urlParams = new URLSearchParams(window.location.search);
const taskId = urlParams.get("taskId");

if (taskId) {
    loadTaskDetails(taskId);

    document.getElementById("startTaskButton").addEventListener("click", () => {
        startTask(taskId);
    });
} else {
    document.getElementById("taskDetails").innerHTML = `<p>⚠️ No task ID found in URL.</p>`;
}
