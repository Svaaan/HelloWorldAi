import { establishNodeSession } from "./nodeSession.js";

let privateKey = null;

// ✅ Generate key pair and save both private & public keys
async function generateKeyPair() {
    const keyPair = await window.crypto.subtle.generateKey(
        {
            name: "ECDSA",
            namedCurve: "P-256",
        },
        true,
        ["sign", "verify"]
    );

    const publicKeyBuffer = await window.crypto.subtle.exportKey("spki", keyPair.publicKey);
    const publicKeyBase64 = btoa(String.fromCharCode(...new Uint8Array(publicKeyBuffer)));

    const privateKeyJwk = await window.crypto.subtle.exportKey("jwk", keyPair.privateKey);
    localStorage.setItem("nodePrivateKey", JSON.stringify(privateKeyJwk));
    localStorage.setItem("nodePublicKeyBase64", publicKeyBase64);

    privateKey = keyPair.privateKey;

    console.log("✅ New keypair generated and saved");
    return publicKeyBase64;
}

// ✅ Manual download for private key
function offerPrivateKeyDownload() {
    const savedPrivateKey = localStorage.getItem("nodePrivateKey");
    const savedPublicKeyBase64 = localStorage.getItem("nodePublicKeyBase64");

    if (!savedPrivateKey || !savedPublicKeyBase64) {
        showMessage("❌ Key data not found. Please try generating again.", "error");
        return;
    }

    const privateKeyJwk = JSON.parse(savedPrivateKey);

    if (!privateKeyJwk.key_ops) {
        privateKeyJwk.key_ops = ["sign"];
    }

    const exportData = {
        privateKey: privateKeyJwk,
        publicKeyBase64: savedPublicKeyBase64
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'node_key_pair.json';
    document.body.appendChild(a);
    a.click();

    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showMessage("✅ Key pair downloaded successfully. Keep it safe!", "success");
}

// ✅ Utility to show messages
function showMessage(message, type = "success") {
    const resultMessageElement = document.getElementById("resultMessage");
    resultMessageElement.innerHTML = message;
    resultMessageElement.className = type === "success" ? "success-message" : "error-message";
}

// ✅ Utility to reset result message area (before next registration flow)
function clearResultMessage() {
    const resultMessageElement = document.getElementById("resultMessage");
    resultMessageElement.innerHTML = "";
    resultMessageElement.className = "";
}

// ✅ Show modal input field
export function showNodeNameInput() {
    clearResultMessage(); // ✅ Clear old messages
    document.getElementById("nodeNameModal").style.display = "flex";
}

// ✅ Hide modal input field
function hideNodeNameInput() {
    const modal = document.getElementById("nodeNameModal");
    if (modal) modal.style.display = "none";
}

// ✅ Show post-registration success actions
function showSuccessActions() {
    const resultMessageElement = document.getElementById("resultMessage");
    resultMessageElement.innerHTML = "✅ Node registered successfully! Please download your private key for future authentication.";
    resultMessageElement.className = "success-message";

    const downloadButton = document.createElement("button");
    downloadButton.textContent = "⬇️ Download Private Key";
    downloadButton.onclick = offerPrivateKeyDownload;

    const continueButton = document.createElement("button");
    continueButton.textContent = "➡️ Continue to Dashboard";
    continueButton.onclick = () => window.location.href = "/node";

    resultMessageElement.appendChild(document.createElement("br"));
    resultMessageElement.appendChild(downloadButton);
    resultMessageElement.appendChild(continueButton);
}

// ✅ Register node flow
// ✅ Register node flow
async function registerNode() {
    const nodeNameInput = document.getElementById("nodeName");
    const nodeName = nodeNameInput.value.trim();
    const processingMessage = document.getElementById("processingMessage");

    if (!nodeName) {
        showMessage("Please enter a node name before registering.", "error");
        return;
    }

    // ✅ Ensure fresh keypair per registration
    privateKey = null;
    localStorage.removeItem("nodePrivateKey");
    localStorage.removeItem("nodePublicKeyBase64");

    processingMessage.style.display = "flex";
    clearResultMessage();

    try {
        const publicKey = await generateKeyPair();

        // ✅ Step 1: Call /connect-node (initial node info storage)
        const response = await fetch("/connect-node", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                node_name: nodeName,
                public_key: publicKey
            })
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || result.error || `Registration failed: ${response.status}`);
        }

        // The node returns 200 with status "rejected" when it has no usable GPU.
        // Bail out here rather than letting the coordinator retry loop time out.
        if (result.status === "rejected") {
            throw new Error(result.reason || "Node rejected the connection.");
        }

        console.log("✅ Node initial registration completed");

        const finalizeResponse = await fetch("/finalize-connection", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              public_key: publicKey
            })
          });
          

        const finalizeData = await finalizeResponse.json();

        if (!finalizeResponse.ok) {
            throw new Error(finalizeData.detail || finalizeData.error || "Failed to finalize connection.");
        }

        if (!finalizeData.node_id) {
            throw new Error("Missing node ID in finalize connection response.");
        }

        console.log("✅ Node finalized connection, received node ID:", finalizeData.node_id);

        // ✅ Step 3: Prove ownership of the freshly generated key to obtain a
        // session token. Without this the node cannot heartbeat or be toggled.
        await establishNodeSession(finalizeData.node_id, privateKey);
        console.log("✅ Node session established");

        hideNodeNameInput();
        showSuccessActions();

    } catch (err) {
        console.error("Register node error:", err);
        showMessage(err.message, "error");
    } finally {
        processingMessage.style.display = "none";
    }
}


// ✅ Setup modal button listeners
export function setupRegisterModal() {
    document.getElementById("registerNodeButton").addEventListener("click", showNodeNameInput);
    document.getElementById("confirmRegisterNodeButton").addEventListener("click", registerNode);
    // Closing is handled centrally by the page (close button, backdrop click and
    // Escape, for every modal). This used to bind querySelector(".modal-close"),
    // which only ever matched the first modal on the page.
}
