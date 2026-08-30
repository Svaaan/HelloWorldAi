// Page bootstrap for start.
//
// This lived inline in start.html. It was moved out so the Content-Security
// Policy can say `script-src 'self'` and mean it: with an inline block on the
// page the policy needs 'unsafe-inline', which permits any script an attacker
// manages to inject as well. Nothing else about it changed.

import { loadHeader } from "/static/js/component/header.js";
import { initStart } from "/static/js/component/start.js";

loadHeader();
initStart();
