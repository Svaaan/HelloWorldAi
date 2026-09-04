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

    // Signed in, this card has one thing to say and it is not about keys.
    //
    // markAlreadySetUp ran before /auth/me answered, so on a browser holding no
    // key it left "Create key file" and "I have a key file" standing there --
    // offering to make an identity to somebody who has just told us who they
    // are, in the vocabulary the account exists to get rid of.
    const signIn = document.getElementById("builderSignIn");
    if (signIn) signIn.hidden = true;

    const alt = document.querySelector("#builderChoice .start-alt");
    if (alt) alt.hidden = true;

    const fine = document.querySelector("#builderChoice .start-fineprint");
    if (fine) fine.hidden = true;

    const button = document.getElementById("builderStart");
    if (button) {
        button.textContent = "Open your workspace";
        button.className = "btn";
        // It opens the save-your-key dialog otherwise, which is the one thing
        // this card should no longer be doing to somebody with an account.
        button.replaceWith(asLink(button, "/workspace"));
    }

    const note = document.querySelector("#builderChoice .start-choice-note");
    if (note) {
        note.replaceChildren();
        const who = document.createElement("strong");
        who.textContent = `Signed in as ${account.login || "you"}.`;
        note.append(who, document.createTextNode(
            " Your work follows this account, on any machine."));
    }
});

/** Same button, as a link that just goes somewhere. */
function asLink(button, href) {
    const link = document.createElement("a");
    link.id = button.id;
    link.className = button.className;
    link.href = href;
    link.textContent = button.textContent;
    return link;
}
