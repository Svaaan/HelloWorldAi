import { establishNodeSession } from "./nodeSession.js";

export function setupConnectExistingNodeModal() {
  const modal = document.getElementById("connectNodeModal");
  const openButton = document.getElementById("connectNodeButton");
  const confirmButton = document.getElementById("confirmConnectNodeButton");
  const fileInput = document.getElementById("privateKeyFile");
  const processingMessage = document.getElementById("processingMessage");
  const resultMessage = document.getElementById("resultMessage");

  let privateKey = null;

  function showMessage(message, type = "success") {
    // textContent, not innerHTML. Every message here is plain text, but one of
    // them is `err.message`, and those errors are built from the server's
    // `detail` field -- which in places is a formatted string containing a
    // node_id taken straight from the URL. That is markup arriving from
    // outside and being parsed, which is the one thing this file must not do.
    resultMessage.textContent = message;
    resultMessage.className = type === "success" ? "success-message" : "error-message";
  }

  function clearModalState() {
    privateKey = null;
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
      // Step 1: Load and validate the uploaded key file
      const file = fileInput.files[0];
      const text = await file.text();
      const parsedFile = JSON.parse(text);

      const privateKeyJwk = parsedFile.privateKey;
      const publicKeyBase64 = parsedFile.publicKeyBase64;

      if (!privateKeyJwk || !publicKeyBase64) {
        throw new Error("Invalid key file format. Please use the downloaded file from registration.");
      }

      if (!privateKeyJwk.key_ops) {
        privateKeyJwk.key_ops = ["sign"];
      }

      privateKey = await window.crypto.subtle.importKey(
        "jwk",
        privateKeyJwk,
        { name: "ECDSA", namedCurve: "P-256" },
        true,
        ["sign"]
      );

      console.log("✅ Private key imported successfully");

      // Step 2: Prepare node memory with this public key (also validates the GPU).
      // Goes through the dashboard proxy — the node is not reachable from the browser.
      const connectResponse = await fetch("/connect-node", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          node_name: "Imported Node",
          public_key: publicKeyBase64
        })
      });

      const connectData = await connectResponse.json();

      if (!connectResponse.ok) {
        throw new Error(connectData.detail || connectData.error || "Failed to reach the node process.");
      }

      if (connectData.status === "rejected") {
        throw new Error(connectData.reason || "Node rejected the connection.");
      }

      // Step 3: Resolve node ID from the public key
      const nodeIdResponse = await fetch("/find-node-id", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ public_key: publicKeyBase64 }),
      });

      const nodeIdData = await nodeIdResponse.json();

      if (!nodeIdResponse.ok || !nodeIdData.node_id) {
        throw new Error("Node not found. Please register this key before connecting.");
      }

      const nodeId = nodeIdData.node_id;
      console.log("✅ Found node ID:", nodeId);

      // Step 4: Prove ownership of the key and obtain a session token
      await establishNodeSession(nodeId, privateKey);
      console.log("✅ Node verified and session established");

      // Step 5: Finalize — refreshes capabilities and marks the node connected
      const finalizeResponse = await fetch("/finalize-connection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ public_key: publicKeyBase64 })
      });

      const finalizeData = await finalizeResponse.json();

      if (!finalizeResponse.ok || !finalizeData.node_id) {
        throw new Error(finalizeData.detail || finalizeData.error || "Failed to finalize node connection.");
      }

      console.log("✅ Finalization successful:", finalizeData.node_id);

      showMessage("✅ Node verified and connected! Redirecting...", "success");
      setTimeout(() => window.location.href = "/node", 2000);

    } catch (err) {
      console.error("Connect node error:", err);
      showMessage(err.message || "An unknown error occurred.", "error");
    } finally {
      processingMessage.style.display = "none";
      confirmButton.disabled = false;
    }
  });
}
