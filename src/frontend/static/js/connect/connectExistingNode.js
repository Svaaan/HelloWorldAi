import { reconnectNode } from "./nodeReconnect.js";
// nodeCallError is not called here any more -- nodeReconnect builds the errors
// now -- but the flag it sets still has to be passed on below, and the test
// that guards that link checks this file names it.
import { nodeCallError, showNodeMessage } from "./nodeErrors.js";

export function setupConnectExistingNodeModal() {
  const modal = document.getElementById("connectNodeModal");
  const openButton = document.getElementById("connectNodeButton");
  const confirmButton = document.getElementById("confirmConnectNodeButton");
  const fileInput = document.getElementById("privateKeyFile");
  const processingMessage = document.getElementById("processingMessage");
  const resultMessage = document.getElementById("resultMessage");

  // textContent throughout, never innerHTML: these messages carry the server's
  // `detail`, which in places is a formatted string containing a node_id taken
  // straight from the URL. `options.offerSetupGuide` adds a link to the guide,
  // for the one failure that is really an instruction.
  function showMessage(message, type = "success", options = {}) {
    showNodeMessage(resultMessage, message, type, options);
  }

  function clearModalState() {
    fileInput.value = "";
    resultMessage.textContent = "";
    resultMessage.className = "";
  }

  function open() {
    clearModalState();
    modal.style.display = "flex";
  }

  openButton.addEventListener("click", open);

  // Arriving from "I have a key file" on the front door. Without this, that
  // button would land somebody on a page offering to register a new node --
  // which is the opposite of what they said they wanted.
  if (new URLSearchParams(window.location.search).get("key") === "1") {
    open();
  }

  confirmButton.addEventListener("click", async () => {
    if (!fileInput.files.length) {
      showMessage("Please upload your key file first.", "error");
      return;
    }

    processingMessage.style.display = "flex";
    confirmButton.disabled = true;

    try {
      const file = fileInput.files[0];
      const parsedFile = JSON.parse(await file.text());

      if (!parsedFile.privateKey || !parsedFile.publicKeyBase64) {
        throw new Error(
          "Invalid key file format. Please use the downloaded file from "
          + "registration.");
      }

      // The same four steps the node page runs when it takes a node back with
      // the key already in the browser. One implementation: they have to
      // happen in that order, and having two copies of that order is how one
      // of them quietly stops matching.
      await reconnectNode({
        privateKeyJwk: parsedFile.privateKey,
        publicKeyBase64: parsedFile.publicKeyBase64,
      });

      showMessage("Node verified and connected. Taking you there…", "success");
      setTimeout(() => { window.location.href = "/node"; }, 1200);

    } catch (err) {
      console.error("Connect node error:", err);
      showMessage(err.message || "An unknown error occurred.", "error",
        { offerSetupGuide: err.offerSetupGuide });
    } finally {
      processingMessage.style.display = "none";
      confirmButton.disabled = false;
    }
  });
}
