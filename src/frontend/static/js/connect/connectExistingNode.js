export function setupConnectExistingNodeModal() {
    const modal = document.getElementById("connectNodeModal");
    const openButton = document.getElementById("connectNodeButton");
    const confirmButton = document.getElementById("confirmConnectNodeButton");
    const fileInput = document.getElementById("privateKeyFile");
    const processingMessage = document.getElementById("processingMessage");
    const resultMessage = document.getElementById("resultMessage");
  
    let privateKey = null;
  
    function showMessage(message, type = "success") {
      resultMessage.innerHTML = message;
      resultMessage.className = type === "success" ? "success-message" : "error-message";
    }
  
    openButton.addEventListener("click", () => {
        privateKey = null; 
        fileInput.value = ""; 
        modal.style.display = "flex";
      });
  
    confirmButton.addEventListener("click", async () => {
      if (!fileInput.files.length) {
        showMessage("Please upload your key file first.", "error");
        return;
      }
  
      processingMessage.style.display = "flex";
  
      try {
        // ✅ Step 1: Read the uploaded key file
        const file = fileInput.files[0];
        const text = await file.text();
        const parsedFile = JSON.parse(text);
  
        const privateKeyJwk = parsedFile.privateKey;
        const publicKeyBase64 = parsedFile.publicKeyBase64;
  
        if (!privateKeyJwk || !publicKeyBase64) {
          throw new Error("Invalid key file format. Please use the correct downloaded key file.");
        }
  
        if (!privateKeyJwk.key_ops) {
          privateKeyJwk.key_ops = ["sign"];
        }
  
        // ✅ Import private key
        privateKey = await window.crypto.subtle.importKey(
          "jwk",
          privateKeyJwk,
          { name: "ECDSA", namedCurve: "P-256" },
          true,
          ["sign"]
        );
  
        console.log("✅ Private key loaded");
  
        // ✅ Step 2: Find node ID from public key
        const nodeIdResponse = await fetch("/find-node-id", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ public_key: publicKeyBase64 }),
        });
  
        const nodeIdData = await nodeIdResponse.json();
  
        if (!nodeIdResponse.ok) {
          throw new Error(nodeIdData.detail || nodeIdData.error || "Failed to find node ID.");
        }
  
        const nodeId = nodeIdData.node_id;
        console.log("✅ Node ID resolved:", nodeId);
  
        // ✅ Step 3: Request challenge
        const challengeResponse = await fetch(`/generate-challenge/${nodeId}`);
        const challengeData = await challengeResponse.json();
  
        if (!challengeResponse.ok || !challengeData.challenge) {
          throw new Error(challengeData.error || "Failed to retrieve challenge from server.");
        }
  
        const challenge = challengeData.challenge;
        console.log("✅ Challenge received:", challenge);
  
        // ✅ Step 4: Sign the challenge
        const encoder = new TextEncoder();
        const challengeBuffer = encoder.encode(challenge);
  
        const signature = await window.crypto.subtle.sign(
          { name: "ECDSA", hash: { name: "SHA-256" } },
          privateKey,
          challengeBuffer
        );
  
        const signatureHex = Array.from(new Uint8Array(signature))
          .map(byte => byte.toString(16).padStart(2, "0"))
          .join("");
  
        console.log("✅ Signature generated:", signatureHex);
  
        // ✅ Step 5: Verify signature with server
        const verifyResponse = await fetch(`/verify-challenge/${nodeId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ signature: signatureHex }),
        });
  
        const verifyData = await verifyResponse.json();
  
        if (!verifyResponse.ok) {
          throw new Error(verifyData.detail || verifyData.error || "Verification failed.");
        }
  
        console.log("✅ Node verified successfully!");
  
        // ✅ Step 6: Finalize connection
        const finalizeResponse = await fetch("/finalize-connection", {
          method: "POST",
        });
  
        const finalizeData = await finalizeResponse.json();
  
        if (!finalizeResponse.ok) {
          throw new Error(finalizeData.detail || finalizeData.error || "Failed to finalize connection.");
        }
  
        console.log("✅ Finalize connection complete:", finalizeData);
  
        showMessage("✅ Node verified and connected successfully! Redirecting...", "success");
  
        setTimeout(() => window.location.href = "/node", 2000);
  
      } catch (err) {
        console.error("Connect node error:", err);
        showMessage(err.message, "error");
      } finally {
        processingMessage.style.display = "none";
      }
    });
  }
  