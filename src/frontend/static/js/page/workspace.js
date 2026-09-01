// Page bootstrap for workspace.
//
// This lived inline in workspace.html. It was moved out so the Content-Security
// Policy can say `script-src 'self'` and mean it: with an inline block on the
// page the policy needs 'unsafe-inline', which permits any script an attacker
// manages to inject as well. Nothing else about it changed.

import { loadHeader } from "/static/js/component/header.js";
import { initIdentity } from "/static/js/workspace/identity.js";
import { renderSummary } from "/static/js/workspace/summary.js";
import { renderRunning } from "/static/js/workspace/running.js";
import {
    loadMyJobs,
    setJobsListener,
    startMyJobsPolling,
} from "/static/js/distribution/myJobs.js";

loadHeader();

// One fetch feeds the summary, the live panel and the rows underneath them.
// A second poll would double the traffic to say the same thing.
setJobsListener((jobs) => {
    renderSummary(jobs);
    renderRunning(jobs);
});

initIdentity({ onChange: loadMyJobs });
startMyJobsPolling();
