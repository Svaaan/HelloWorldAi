// header.js

export function loadHeader() {
    fetch('/template/header.html')
        .then(res => res.text())
        .then(html => {
            document.getElementById('header-placeholder').innerHTML = html;
            initHeaderStats();
        });
}

function initHeaderStats() {
    async function updateHeaderStats() {
        try {
            const [powerRes, nodesRes] = await Promise.all([
                fetch("/get-total-power"),
                fetch("/get-connected-nodes-count")
            ]);
            const powerData = await powerRes.json();
            const nodesData = await nodesRes.json();

            document.getElementById("totalPower").textContent =
                typeof powerData.total_compute_score === "number"
                    ? `${powerData.total_compute_score.toFixed(2)} compute units`
                    : "N/A";

            document.getElementById("connectedNodesCount").textContent =
                nodesData.connected_nodes_count ?? "N/A";

        } catch (error) {
            console.error("Error updating header stats:", error);
            document.getElementById("totalPower").textContent = "Error";
            document.getElementById("connectedNodesCount").textContent = "Error";
        }
    }

    updateHeaderStats();
    setInterval(updateHeaderStats, 60000);
}
