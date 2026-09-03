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

// After registering: save the file, prove you have it, then go.
//
// It used to offer "Download key file" and "Go to your node" side by side, both
// optional. So the common path was to press the second one -- the flow was
// finished, the node was working, and the dialog was in the way -- and the key
// that is the only proof this node is yours stayed in one browser's local
// storage with no copy anywhere.
//
// Now the door out only opens once the file has been loaded back. Not to be
// strict for its own sake: downloading a file proves a click happened, loading
// it back proves the file exists, is readable, and is the right one. That is
// the difference between believing you have a backup and having one.
function showSuccessActions() {
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
        "Save the key file, then load it back. It is the only proof this node "
        + "is yours -- nobody can issue you another one, and this is the one "
        + "moment where checking the file costs nothing.";
    container.appendChild(lede);

    const step1 = document.createElement("button");
    step1.type = "button";
    step1.className = "btn-primary";
    step1.textContent = "Download key file";
    container.appendChild(step1);

    // Step two appears once step one has been pressed. Shown before that, it
    // asks for a file that does not exist yet.
    const check = document.createElement("div");
    check.hidden = true;
    container.appendChild(check);

    const checkLabel = document.createElement("label");
    checkLabel.setAttribute("for", "confirmKeyFile");
    checkLabel.textContent = "Now load it back, to be sure it saved";
    check.appendChild(checkLabel);

    const checkInput = document.createElement("input");
    checkInput.type = "file";
    checkInput.id = "confirmKeyFile";
    checkInput.accept = ".json,application/json";
    check.appendChild(checkInput);

    const onward = document.createElement("button");
    onward.type = "button";
    onward.className = "btn-secondary";
    onward.textContent = "Go to your node";
    onward.disabled = true;
    container.appendChild(onward);

    const result = document.getElementById("resultMessage");
    if (result) container.appendChild(result);

    step1.addEventListener("click", () => {
        offerPrivateKeyDownload();
        check.hidden = false;
        checkInput.focus();
    });

    checkInput.addEventListener("change", async () => {
        const file = checkInput.files?.[0];
        if (!file) return;

        try {
            const parsed = JSON.parse(await file.text());
            const publicKeyBase64 = parsed.publicKeyBase64;

            if (!parsed.privateKey || !publicKeyBase64) {
                throw new Error("That file is not a node key file.");
            }

            // The right key, not merely a valid one. Somebody with two nodes
            // has two of these files and they look identical.
            if (publicKeyBase64 !== localStorage.getItem("nodePublicKeyBase64")) {
                throw new Error(
                    "That is a key file for a different node. Load the one you "
                    + "just downloaded.");
            }

            showMessage("That is the right file. Your node is ready.", "success");
            onward.disabled = false;
            onward.focus();
        } catch (error) {
            console.error("Key file check failed:", error);
            showMessage(
                error instanceof SyntaxError
                    ? "That file is not valid JSON."
                    : error.message,
                "error");
            checkInput.value = "";
        }
    });

    onward.addEventListener("click", () => { window.location.href = "/node"; });
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
