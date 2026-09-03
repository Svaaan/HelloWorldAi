// Page bootstrap for distribution.
//
// This lived inline in distribution.html. It was moved out so the Content-Security
// Policy can say `script-src 'self'` and mean it: with an inline block on the
// page the policy needs 'unsafe-inline', which permits any script an attacker
// manages to inject as well. Nothing else about it changed.

import { loadHeader } from "/static/js/component/header.js";
// No panel on this page. It is here because submitting is where it matters:
// without it, a signed-in browser holding no key would mint one at the moment
// of sending, and file the job under a digest the account has never heard of.
import { initAccount } from "/static/js/workspace/account.js";
import { fetchAvailableNodes, startNodePolling } from "/static/js/distribution/fetchNode.js";
// showNodeModal is no longer imported here: the only caller on this page was
// the "send to any node" button. fetchNode.js still opens it, with the machine
// somebody clicked.
import { initModalCloseHandler } from "/static/js/distribution/modalHandler.js";

loadHeader();
initAccount();
initModalCloseHandler();
startNodePolling();

document.getElementById("refreshNodes")
    .addEventListener("click", () => fetchAvailableNodes());

// There is no "send to any node" button any more. Work goes to a machine
// somebody picked from the list, and this page no longer offers to choose one
// on their behalf.
//
// POST /submit-task still places automatically -- it is how an API client
// submits without naming a machine, and the retry and rescue paths depend on
// jobs being marked as coordinator-placed. Only the button is gone.
