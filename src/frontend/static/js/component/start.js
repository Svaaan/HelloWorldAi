// The front door: which of the two sides of the network are you?
//
// Whichever door is picked, the browser ends up holding a key, and that key is
// what "signed in" means here. A builder's is generated on the spot -- there
// is nothing to verify, since the key is the identity -- while a returning one
// loads the file they saved.
//
// Someone already set up is sent straight through rather than being asked
// again.

import { getSubmitterKey, hasSubmitterKey, setSubmitterKey }
    from "/static/js/distribution/submitter.js";
import { isContributor } from "./role.js";

function setStatus(message, kind) {
  const target = document.getElementById("builderStatus");
  if (!target) return;

  target.replaceChildren();
  if (!message) return;

  const line = document.createElement("span");
  if (kind) line.className = `${kind}-message`;
  line.textContent = message;
  target.appendChild(line);
}

function markAlreadySetUp() {
  // A gentle nudge rather than a redirect: someone may be here to set up the
  // other side, and bouncing them somewhere they did not ask for is worse
  // than an extra click.
  if (hasSubmitterKey()) {
    const choice = document.getElementById("builderChoice");
    choice?.classList.add("is-ready");
    const button = document.getElementById("builderStart");
    if (button) button.textContent = "Open your workspace";
  }

  if (isContributor()) {
    document.getElementById("contributorChoice")?.classList.add("is-ready");

    // Reveal the way back to an existing node without taking away the way to
    // connect another. Replacing one with the other is what left somebody
    // holding a node unable to add a second.
    const node = document.getElementById("contributorNode");
    if (node) node.hidden = false;

    const connect = document.getElementById("contributorPrimary");
    if (connect) connect.textContent = "Connect another GPU";
  }
}

export function initStart() {
  markAlreadySetUp();

  const start = document.getElementById("builderStart");
  if (start) {
    start.addEventListener("click", () => {
      // Creating the key here rather than silently on first submit means the
      // thing that owns your jobs exists before you have any, and can be saved
      // before there is anything to lose.
      getSubmitterKey();
      window.location.href = "/workspace";
    });
  }

  const returning = document.getElementById("builderReturning");
  const fileInput = document.getElementById("builderKeyFile");

  if (returning && fileInput) {
    returning.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files?.[0];
      if (!file) return;

      try {
        const parsed = JSON.parse(await file.text());
        const key = typeof parsed === "string" ? parsed : parsed.submitterKey;

        if (!key && parsed?.privateKey && parsed?.publicKey) {
          throw new Error(
            "That is a node key file. Use it on the Connect page to bring a "
            + "GPU back online."
          );
        }
        if (!key) throw new Error("No builder key in that file.");

        setSubmitterKey(key);
        window.location.href = "/workspace";
      } catch (error) {
        console.error("Could not load the builder key:", error);
        setStatus(
          error instanceof SyntaxError
            ? "That file is not valid JSON."
            : error.message,
          "error",
        );
        fileInput.value = "";
      }
    });
  }
}
