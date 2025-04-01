// In testNode.js
const testNode = {
    // Run CPU test
    runCpuTest: function(nodeId) {
        return this._runTest(nodeId, 'cpu');
    },
    
    // Run GPU test
    runGpuTest: function(nodeId) {
        return this._runTest(nodeId, 'gpu');
    },
    
    // Generic test runner
    _runTest: function(nodeId, testType) {
        // Show loading state
        const button = document.getElementById(`run-${testType}-test-btn`);
        const originalText = button.textContent;
        button.textContent = `Testing ${testType.toUpperCase()}...`;
        button.disabled = true;
        
        // Clear previous results
        const resultElement = document.getElementById(`${testType}-test-result`);
        if (resultElement) {
            resultElement.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div> Running test...';
        }
        
        // Call the verification endpoint
        return fetch(`/verify-node/${nodeId}/${testType}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                this._pollForResults(nodeId, testType);
                return true;
            } else {
                if (resultElement) {
                    resultElement.innerHTML = `<span class="text-danger">Failed: ${data.message}</span>`;
                }
                return false;
            }
        })
        .catch(error => {
            console.error(`Error running ${testType} test:`, error);
            if (resultElement) {
                resultElement.innerHTML = '<span class="text-danger">Test failed to start</span>';
            }
            return false;
        })
        .finally(() => {
            // Reset button state after a delay
            setTimeout(() => {
                button.textContent = originalText;
                button.disabled = false;
            }, 2000);
        });
    },
    
    _pollForResults: function (nodeId, testType) {
        const resultElement = document.getElementById(`${testType}-test-result`);
        const pollInterval = setInterval(() => {
            fetch(`/node-performance/${nodeId}`)
                .then((response) => response.json())
                .then((data) => {
                    if (data.status === "success") {
                        const isVerified =
                            testType === "cpu"
                                ? data.cpu_verified
                                : data.gpu_verified;
                        const usage =
                            testType === "cpu"
                                ? data.cpu_usage
                                : data.gpu_usage;
                        const benchmark =
                            testType === "cpu"
                                ? data.cpu_benchmark
                                : data.gpu_benchmark;
    
                        if (isVerified) {
                            clearInterval(pollInterval);
    
                            let benchmarkLabel = "N/A";
                            if (typeof benchmark === "number") {
                                benchmarkLabel =
                                    testType === "cpu"
                                        ? `${benchmark.toLocaleString()} operations`
                                        : `${benchmark}x${benchmark} tensor`;
                            } else if (typeof benchmark === "string") {
                                benchmarkLabel = benchmark;
                            }
    
                            resultElement.innerHTML = `
                                <div class="alert alert-success">
                                    <strong>${testType.toUpperCase()} Test Passed</strong>
                                    <div>Usage during test: ${usage.toFixed(2)}%</div>
                                    <div>${benchmarkLabel !== "N/A" ? "Result: " + benchmarkLabel : ""}</div>
                                </div>
                            `;
                        }
                    }
                })
                .catch((error) => {
                    console.error("Error polling for results:", error);
                });
        }, 1000);
    
        // Timeout after 30s
        setTimeout(() => {
            clearInterval(pollInterval);
            if (
                resultElement &&
                resultElement.innerHTML.includes("Running test")
            ) {
                resultElement.innerHTML =
                    '<div class="alert alert-warning">Test timed out. Please try again.</div>';
            }
        }, 30000);
    },
    
    
    // Initialize buttons
    init: function() {
        document.addEventListener('DOMContentLoaded', () => {
            // CPU test button
            const cpuButton = document.getElementById('run-cpu-test-btn');
            if (cpuButton) {
                cpuButton.addEventListener('click', () => {
                    const nodeId = cpuButton.getAttribute('data-node-id');
                    if (nodeId) {
                        this.runCpuTest(nodeId);
                    } else {
                        alert('No node selected');
                    }
                });
            }
            
            // GPU test button
            const gpuButton = document.getElementById('run-gpu-test-btn');
            if (gpuButton) {
                gpuButton.addEventListener('click', () => {
                    const nodeId = gpuButton.getAttribute('data-node-id');
                    if (nodeId) {
                        this.runGpuTest(nodeId);
                    } else {
                        alert('No node selected');
                    }
                });
            }
        });
    },
    
    // Update buttons with current node ID
    updateTestButtons: function(nodeId) {
        const cpuButton = document.getElementById('run-cpu-test-btn');
        const gpuButton = document.getElementById('run-gpu-test-btn');
        
        if (cpuButton) {
            cpuButton.setAttribute('data-node-id', nodeId);
            cpuButton.disabled = !nodeId;
        }
        
        if (gpuButton) {
            gpuButton.setAttribute('data-node-id', nodeId);
            gpuButton.disabled = !nodeId;
        }
    }
};

// Initialize the module
testNode.init();

// Export for global access
window.testNode = testNode;