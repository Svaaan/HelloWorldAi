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
import { setupRegisterModal } from "/static/js/connect/registerNodeModal.js";
import { setupConnectExistingNodeModal }
    from "/static/js/connect/connectExistingNode.js";
import { renderGpuStatus } from "/static/js/connect/gpuDetect.js";
import { isContributor } from "./role.js";
import { downloadKeyFile } from "/static/js/workspace/identity.js";

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

// Both cards offer the same two doors -- make a key, or load the one you
// already have -- because that is what this page is: a sign-in for two
// separate profiles. Once a side is set up, its first door becomes the way
// back in, and making another key moves out of the way rather than vanishing.
function markAlreadySetUp() {
  // A gentle nudge rather than a redirect: someone may be here to set up the
  // other side, and bouncing them somewhere they did not ask for is worse
  // than an extra click.
  if (hasSubmitterKey()) {
    const choice = document.getElementById("builderChoice");
    choice?.classList.add("is-ready");

    const button = document.getElementById("builderStart");
    if (button) button.textContent = "Open your workspace";

    // Loading a file still works -- it is how you move an identity between
    // machines -- but it is no longer the obvious thing to press.
    const returning = document.getElementById("builderReturning");
    if (returning) returning.textContent = "Use a different key file";
  }

  if (isContributor()) {
    document.getElementById("contributorChoice")?.classList.add("is-ready");

    const node = document.getElementById("contributorNode");
    const connect = document.getElementById("registerNodeButton");

    // Reveal the way back to an existing node without taking away the way to
    // connect another. Replacing one with the other is what left somebody
    // holding a node unable to add a second.
    //
    // Getting back in leads, on both cards: somebody returning to a machine
    // they already set up is the common case, and adding a second graphics
    // card is not.
    if (node && connect) {
      node.hidden = false;
      node.className = "btn";
      connect.className = "btn-ghost";
      connect.textContent = "Connect GPU";
      connect.parentNode.insertBefore(node, connect);
    }

    // Already reconnected; the file is not what they need from this page.
    const returning = document.getElementById("connectNodeButton");
    if (returning) returning.hidden = true;
  }
}

// One handler for every close button and backdrop click, for the modals that
// registering a node brings with it.
function setupModalDismissal() {
  document.querySelectorAll("[data-close]").forEach((button) => {
    button.addEventListener("click", () => {
      const modal = document.getElementById(button.dataset.close);
      if (modal) modal.style.display = "none";
    });
  });

  document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
    backdrop.addEventListener("mousedown", (event) => {
      if (event.target === backdrop) backdrop.style.display = "none";
    });
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    document.querySelectorAll(".modal-backdrop").forEach((backdrop) => {
      backdrop.style.display = "none";
    });
  });
}

// Setting up a graphics card used to be a second page. It is the same two
// choices as the data card -- make a key, or load the one you have -- so it
// belongs on the same screen, in a dialog, exactly as loading a builder key
// already did.
function setupContributor() {
  if (!document.getElementById("registerNodeButton")) return;

  setupRegisterModal();
  setupConnectExistingNodeModal();
  setupModalDismissal();

  // The check runs when somebody asks to register, not when the page loads.
  // Running it on load is what made the front door interrogate the hardware
  // of every visitor, including the ones who came with data and a laptop.
  let checked = false;
  document.getElementById("registerNodeButton").addEventListener("click", () => {
    if (checked) return;
    checked = true;
    renderGpuStatus(document.getElementById("gpuInfo"));
  });
}

export function initStart() {
  markAlreadySetUp();
  setupContributor();

  const start = document.getElementById("builderStart");
  if (start) {
    start.addEventListener("click", () => {
      // Creating the key here rather than silently on first submit means the
      // thing that owns your jobs exists before you have any, and can be saved
      // before there is anything to lose.
      getSubmitterKey();

      // And then say so, rather than walking straight past it. The button says
      // "Create key file"; it used to produce no file at all and land you in
      // the workspace, where a panel you had to notice offered to save one.
      // The GPU side has always stopped and insisted. Both sides now do.
      const modal = document.getElementById("builderKeyModal");
      if (!modal) {                       // markup missing: do not trap anyone
        window.location.href = "/workspace";
        return;
      }
      modal.style.display = "flex";
    });
  }

  const keyDownload = document.getElementById("builderKeyDownload");
  if (keyDownload) {
    keyDownload.addEventListener("click", () => {
      downloadKeyFile();
      const said = document.getElementById("builderKeyResult");
      if (said) {
        said.textContent = "Saved. Keep it somewhere you will find it again.";
        said.className = "field-status is-ok";
      }
    });
  }

  const keyClose = document.querySelector('[data-close="builderKeyModal"]');
  if (keyClose) {
    keyClose.addEventListener("click", () => {
      const modal = document.getElementById("builderKeyModal");
      if (modal) modal.style.display = "none";
    });
  }

  const keyOnward = document.getElementById("builderKeyOnward");
  if (keyOnward) {
    keyOnward.addEventListener("click", () => {
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
