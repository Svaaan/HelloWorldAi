// Page bootstrap for distribution.
//
// This lived inline in distribution.html. It was moved out so the Content-Security
// Policy can say `script-src 'self'` and mean it: with an inline block on the
// page the policy needs 'unsafe-inline', which permits any script an attacker
// manages to inject as well. Nothing else about it changed.

import { loadHeader } from "/static/js/component/header.js";
import { fetchAvailableNodes, startNodePolling } from "/static/js/distribution/fetchNode.js";
import { initModalCloseHandler, showNodeModal } from "/static/js/distribution/modalHandler.js";

loadHeader();
initModalCloseHandler();
startNodePolling();

document.getElementById("refreshNodes")
    .addEventListener("click", () => fetchAvailableNodes());

// No node argument: the coordinator chooses at submit time.
document.getElementById("sendAnywhere")
    .addEventListener("click", () => showNodeModal(null));
