// Page bootstrap for setup.
//
// This lived inline in setup.html. It was moved out so the Content-Security
// Policy can say `script-src 'self'` and mean it: with an inline block on the
// page the policy needs 'unsafe-inline', which permits any script an attacker
// manages to inject as well. Nothing else about it changed.

import { loadHeader } from "/static/js/component/header.js";
import { initSetup } from "/static/js/setup/setup.js";

loadHeader();
initSetup();
