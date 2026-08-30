// Page bootstrap for node.
//
// This lived inline in node.html. It was moved out so the Content-Security
// Policy can say `script-src 'self'` and mean it: with an inline block on the
// page the policy needs 'unsafe-inline', which permits any script an attacker
// manages to inject as well. Nothing else about it changed.

import { loadHeader } from "/static/js/component/header.js";
import { startJobPolling } from "/static/js/nodejs/jobList.js";
import { initNodeInfoManager } from "/static/js/nodejs/nodeInfo.js";
import { startLiveWorkPolling } from "/static/js/nodejs/liveWork.js";
import { initSelfTest } from "/static/js/nodejs/selfTest.js";

loadHeader();
startJobPolling();
initSelfTest();
startLiveWorkPolling();
window.nodeInfoManager = initNodeInfoManager();
