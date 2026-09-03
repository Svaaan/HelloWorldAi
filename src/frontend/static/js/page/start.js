// Page bootstrap for start.
//
// This lived inline in start.html. It was moved out so the Content-Security
// Policy can say `script-src 'self'` and mean it: with an inline block on the
// page the policy needs 'unsafe-inline', which permits any script an attacker
// manages to inject as well. Nothing else about it changed.

import { loadHeader } from "/static/js/component/header.js";
import { initStart } from "/static/js/component/start.js";
import { initAccount } from "/static/js/workspace/account.js";

loadHeader();
initStart();

// Only to answer "are they already signed in", so the card does not offer a
// door somebody has already walked through. There is no account panel on this
// page; initAccount finds none and renders nothing.
initAccount().then((account) => {
    if (!account?.signed_in) return;

    const signIn = document.getElementById("builderSignIn");
    if (signIn) signIn.hidden = true;

    // Signing in creates a key, so this reads "Open your workspace" already --
    // but markAlreadySetUp ran before the answer arrived, and on a browser
    // that had nothing it left the card offering to make a key that now
    // exists.
    const button = document.getElementById("builderStart");
    if (button) {
        button.textContent = "Open your workspace";
        button.className = "btn";
    }
});
