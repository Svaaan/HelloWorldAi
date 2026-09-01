import { establishNodeSession } from "./nodeSession.js";
import { nodeCallError, showNodeMessage } from "./nodeErrors.js";
import { downloadNodeKeyFile } from "./nodeKeyFile.js";

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

// Saving the keypair moved to connect/nodeKeyFile.js, because the node page
// needs it too: its "your key exists only in this browser" reminder used to
// offer a link to "/" rather than a download, since the only implementation
// was in here.
function offerPrivateKeyDownload() {
    const { ok, message } = downloadNodeKeyFile();
    showMessage(message, ok ? "success" : "error");
}

// ✅ Utility to show messages
//
// `options.offerSetupGuide` adds a link to the guide. It is set when the
// failure was that no node agent is running here, which is not really an error
// so much as the next thing to go and do.
function showMessage(message, type = "success", options = {}) {
    showNodeMessage(document.getElementById("resultMessage"), message, type, options);
}

// ✅ Utility to reset result message area (before next registration flow)
function clearResultMessage() {
    const resultMessageElement = document.getElementById("resultMessage");
    resultMessageElement.textContent = "";
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
    // Finish in the dialog that started it.
    //
    // This used to write into a panel on the page behind, which was fine when
    // registering had a page of its own and is not now: on the front door it
    // landed inside one of the two cards, shoved the layout around, and left
    // the buttons you had just pressed sitting above the result of pressing
    // them.
    const modal = document.getElementById("nodeNameModal");
    const container = modal?.querySelector(".modal-container");
    if (!container) return;

    modal.style.display = "flex";
    container.replaceChildren();

    const title = document.createElement("h3");
    title.textContent = "Node registered";
    container.appendChild(title);

    const lede = document.createElement("p");
    lede.className = "modal-lede";
    lede.textContent =
        "Save the key file now. It is the only proof this node is yours, and "
        + "nobody can issue you another one.";
    container.appendChild(lede);

    const download = document.createElement("button");
    download.type = "button";
    download.className = "btn-primary";
    download.textContent = "Download key file";
    download.addEventListener("click", offerPrivateKeyDownload);
    container.appendChild(download);

    const onward = document.createElement("button");
    onward.type = "button";
    onward.className = "btn-secondary";
    onward.textContent = "Go to your node";
    onward.addEventListener("click", () => { window.location.href = "/node"; });
    container.appendChild(onward);

    // Where the download confirmation lands, now that the page behind is no
    // longer the place for it.
    const result = document.getElementById("resultMessage");
    if (result) container.appendChild(result);
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
            throw nodeCallError(response, result,
                `Registration failed: ${response.status}`);
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
        showMessage(err.message, "error",
            { offerSetupGuide: err.offerSetupGuide });
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
