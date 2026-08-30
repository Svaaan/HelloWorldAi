// Page bootstrap for workspace.
//
// This lived inline in workspace.html. It was moved out so the Content-Security
// Policy can say `script-src 'self'` and mean it: with an inline block on the
// page the policy needs 'unsafe-inline', which permits any script an attacker
// manages to inject as well. Nothing else about it changed.

import { loadHeader } from "/static/js/component/header.js";
import { initIdentity } from "/static/js/workspace/identity.js";
import { renderSummary } from "/static/js/workspace/summary.js";
import {
    loadMyJobs,
    setJobsListener,
    startMyJobsPolling,
} from "/static/js/distribution/myJobs.js";

loadHeader();

// One fetch feeds both the summary and the rows underneath it.
setJobsListener(renderSummary);

initIdentity({ onChange: loadMyJobs });
startMyJobsPolling();
