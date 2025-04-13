/**
 * Task Management Dashboard JS
 * Handles loading, displaying, and processing of task data
 */

// Cache DOM elements to reduce lookups
const elements = {
  taskDetails: document.getElementById('taskDetails'),
  statusElement: document.getElementById('taskStatus'),
  resultElement: document.getElementById('taskResult'),
  startTaskButton: document.getElementById('startTaskButton')
};

// Store any active intervals for cleanup
let statusPollingInterval = null;

/**
* Loads and displays task details for a given task ID
* @param {string} taskId - The ID of the task to load
*/
async function loadTaskDetails(taskId) {
  try {
      const response = await fetch('/get-pending-tasks');
      if (!response.ok) throw new Error(`Failed to fetch tasks: ${response.status}`);

      const tasks = await response.json();
      const task = tasks.find(t => t.task_id === taskId);

      if (!task) {
          handleTaskNotFound(taskId);
          return;
      }

      displayTaskDetails(task);

      // Also check for existing result
      loadTaskResult(taskId);
  } catch (error) {
      console.error("Error loading task details:", error);
      elements.taskDetails.innerHTML = `<p>Error loading task details: ${error.message}</p>`;
  }
}

/**
* Handles the case when a task is not found in pending tasks
* @param {string} taskId - The ID of the task that wasn't found
*/
function handleTaskNotFound(taskId) {
  elements.taskDetails.innerHTML = `<p>Task not found or already processed.</p>`;
  elements.statusElement.innerHTML = `<span style="color: blue;">Completed</span>`;
  elements.startTaskButton.disabled = true;

  // Try to load results if task not found in pending
  loadTaskResult(taskId);
}

/**
* Displays the details of a task in the UI
* @param {Object} task - The task object to display
*/
function displayTaskDetails(task) {
  const formattedParams = JSON.stringify(task.task_data.hyperparameters, null, 2);

  elements.taskDetails.innerHTML = `
    <p><strong>Task ID:</strong> ${task.task_id}</p>
    <p><strong>Task Type:</strong> ${task.task_data.task_type}</p>
    <p><strong>Model:</strong> ${task.task_data.model_name}</p>
    <p><strong>Hyperparameters:</strong> <pre style="background-color: #f5f5f5; padding: 8px; border-radius: 4px;">${formattedParams}</pre></p>
  `;

  elements.statusElement.innerHTML = `<span style="color: orange;">Pending</span>`;
}

/**
* Loads and displays the result of a task
* @param {string} taskId - The ID of the task to load results for
*/
async function loadTaskResult(taskId) {
  elements.resultElement.innerHTML = "<p>🔍 Checking for result...</p>";

  try {
      const res = await fetch('/get-task-results');
      if (!res.ok) throw new Error(`Failed to fetch results: ${res.status}`);

      const results = await res.json();
      console.log("Received task results:", results);

      // ✅ Search results by '_id', since task_id is not stored in DB
      const result = results.find(r => r._id === taskId);

      if (!result) {
          elements.resultElement.innerHTML = `<p>⏳ No result available yet.</p>`;
          return;
      }

      displayTaskResult(result);
  } catch (error) {
      console.error("Error loading task result:", error);
      elements.resultElement.innerHTML = `<p>❌ Failed to load task result: ${error.message}</p>`;
  }
}

/**
* Displays the result of a task in the UI
* @param {Object} result - The result object to display
*/
function displayTaskResult(result) {
  const statusColor = result.status === 'completed' ? 'green' : 'red';

  const logLines = Array.isArray(result.logs) && result.logs.length > 0
      ? result.logs.map(line => `<li>${line}</li>`).join("")
      : "<li>No logs available</li>";

  elements.resultElement.innerHTML = `
      <p><strong>Status:</strong> <span style="color: ${statusColor};">${result.status}</span></p>
      <p><strong>Log Output:</strong></p>
      <ul style="background-color: #f8f9fa; padding: 12px; border-radius: 4px; max-height: 300px; overflow-y: auto;">
        ${logLines}
      </ul>
  `;
}

/**
* Starts processing a task
* @param {string} taskId - The ID of the task to start
*/
async function startTask(taskId) {
  try {
      elements.startTaskButton.disabled = true;
      elements.startTaskButton.innerHTML = 'Starting task...';

      const response = await fetch(`/process-task/${taskId}`, {
          method: "POST",
          headers: {
              'Content-Type': 'application/json'
          }
      });

      if (!response.ok) throw new Error(`Failed to start task: ${response.status}`);
      const result = await response.json();

      handleTaskStartResult(result, taskId);
  } catch (error) {
      console.error("Error starting task:", error);
      elements.statusElement.innerHTML = `<span style="color: red;">Error: ${error.message}</span>`;

      elements.startTaskButton.disabled = false;
      elements.startTaskButton.innerHTML = 'Start Task';
  }
}

/**
* Handles the result of starting a task
* @param {Object} result - The result of the start task operation
* @param {string} taskId - The ID of the task that was started
*/
function handleTaskStartResult(result, taskId) {
  if (result.status === "processing") {
      elements.statusElement.innerHTML = `<span style="color: green;">Task started: ${result.message}</span>`;
      elements.startTaskButton.disabled = true;
      elements.startTaskButton.innerHTML = 'Task Running';

      startStatusPolling(taskId);
  } else {
      elements.statusElement.innerHTML = `<span style="color: red;">Failed to start: ${result.message}</span>`;

      elements.startTaskButton.disabled = false;
      elements.startTaskButton.innerHTML = 'Retry';
  }
}

/**
* Starts polling for status updates for a task
* @param {string} taskId - The ID of the task to poll for
*/
function startStatusPolling(taskId) {
  if (statusPollingInterval) {
      clearInterval(statusPollingInterval);
  }

  statusPollingInterval = setInterval(async () => {
      try {
          // ✅ 1. First check task status
          const response = await fetch('/get-pending-tasks');
          if (!response.ok) throw new Error(`Failed to fetch tasks: ${response.status}`);

          const tasks = await response.json();
          const task = tasks.find(t => t.task_id === taskId);

          updateTaskStatus(task, taskId);

          // ✅ 2. Also check usage info for protection!
          const usageResponse = await fetch('/usage');
          if (!usageResponse.ok) throw new Error(`Failed to fetch usage info: ${usageResponse.status}`);

          const usageData = await usageResponse.json();

          const gpuList = Array.isArray(usageData.gpu_data) ? usageData.gpu_data : [];

          const warnings = gpuList.map(gpu => {
              const warningsForGpu = [];

              if (gpu.gpu_usage > 95) {
                  warningsForGpu.push(`⚠️ GPU ${gpu.index} usage is very high: ${gpu.gpu_usage}%`);
              }

              if (gpu.gpu_temperature > gpu.critical_temperature - 5) {
                  warningsForGpu.push(`🔥 GPU ${gpu.index} temperature critical: ${gpu.gpu_temperature}°C`);
              }

              return warningsForGpu.join('<br>');
          }).filter(w => w.length > 0);

          const warningElement = document.getElementById('taskWarning');
          if (warningElement) {
              warningElement.innerHTML = warnings.length > 0 ? warnings.join('<br>') : '';
          }

      } catch (error) {
          console.error("Error during polling:", error);
      }
  }, 3000);
}


/**
* Updates the task status in the UI based on polling results
* @param {Object|null} task - The task object if found, null otherwise
* @param {string} taskId - The ID of the task
*/
function updateTaskStatus(task, taskId) {
  if (!task) {
      elements.statusElement.innerHTML = `<span style="color: blue;"> Completed!</span>`;
      loadTaskResult(taskId);
  } else {
      elements.statusElement.innerHTML = `<span style="color: orange;">Pending</span>`;
  }
}

// Initialize the page
function initPage() {
  const urlParams = new URLSearchParams(window.location.search);
  const taskId = urlParams.get("taskId");

  if (taskId) {
      loadTaskDetails(taskId);

      if (elements.startTaskButton) {
          elements.startTaskButton.addEventListener("click", () => {
              startTask(taskId);
          });
      }

      setInterval(monitorHardware, 2000);

  } else {
      elements.taskDetails.innerHTML = `<p>No task ID found in URL.</p>`;
      if (elements.statusElement) elements.statusElement.innerHTML = '';
      if (elements.resultElement) elements.resultElement.innerHTML = '';
  }
}


async function monitorHardware() {
  try {
      const res = await fetch('/usage');
      if (!res.ok) throw new Error(`Failed to fetch usage info: ${res.status}`);

      const data = await res.json();

      const cpuUsage = data.cpu_usage ?? 0;
      const gpuData = Array.isArray(data.gpu_data) ? data.gpu_data : [];

      const cpuUsagePercent = document.getElementById("cpuUsagePercent");
      if (cpuUsagePercent) cpuUsagePercent.textContent = `${cpuUsage}%`;

      const gpuUsagePercent = document.getElementById("gpuUsagePercent");
      const gpuTemperature = document.getElementById("gpuTemperature");

      if (gpuData.length > 0) {
          const avgGpuUsage = Math.round(gpuData.reduce((sum, gpu) => sum + (gpu.gpu_usage ?? 0), 0) / gpuData.length);
          const avgTemperature = Math.round(gpuData.reduce((sum, gpu) => sum + (gpu.gpu_temperature ?? 0), 0) / gpuData.length);

          if (gpuUsagePercent) gpuUsagePercent.textContent = `${avgGpuUsage}%`;
          if (gpuTemperature) gpuTemperature.textContent = `${avgTemperature}`;

          const warningElement = document.getElementById("taskWarning");
          const highTempGpu = gpuData.find(gpu => gpu.gpu_temperature >= (gpu.critical_temperature ?? 85));
          const highLoadGpu = gpuData.find(gpu => gpu.gpu_usage >= 95);

          if (warningElement) {
              if (highTempGpu || highLoadGpu) {
                  let warningMsg = '';
                  if (highLoadGpu) warningMsg += `⚠️ GPU ${highLoadGpu.index} usage is very high: ${highLoadGpu.gpu_usage}% `;
                  if (highTempGpu) warningMsg += `🔥 GPU ${highTempGpu.index} temperature critical: ${highTempGpu.gpu_temperature}°C `;
                  warningElement.textContent = warningMsg.trim();
              } else {
                  warningElement.textContent = '';
              }
          }
      }

  } catch (err) {
      console.error("Hardware monitor error:", err);
  }
}


// Cleanup function to prevent memory leaks
function cleanup() {
  if (statusPollingInterval) {
      clearInterval(statusPollingInterval);
  }
}

document.addEventListener('DOMContentLoaded', initPage);
window.addEventListener('beforeunload', cleanup);
