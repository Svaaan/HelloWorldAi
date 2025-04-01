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
            const res = await fetch("/get-connected-nodes-count");
            const data = await res.json();

            document.getElementById("connectedNodesCount").textContent =
                data.connected_nodes_count ?? "N/A";
        } catch (error) {
            console.error("Error updating connected nodes count:", error);
            document.getElementById("connectedNodesCount").textContent = "Error";
        }
    }

    updateHeaderStats();
    setInterval(updateHeaderStats, 60000);
}
