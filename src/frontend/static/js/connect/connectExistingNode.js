export function setupConnectExistingNodeModal() {
    const modal = document.getElementById("connectNodeModal");
    const openButton = document.getElementById("connectNodeButton");
    const confirmButton = document.getElementById("confirmConnectNodeButton");
    const fileInput = document.getElementById("privateKeyFile");
    const processingMessage = document.getElementById("processingMessage");
    const resultMessage = document.getElementById("resultMessage");
  
    let privateKey = null;
  
    // ✅ Helper: Show messages
    function showMessage(message, type = "success") {
      resultMessage.innerHTML = message;
      resultMessage.className = type === "success" ? "success-message" : "error-message";
    }
  
    // ✅ Open modal
    openButton.addEventListener("click", () => {
      modal.style.display = "flex";
    });
  
    // ✅ Confirm connect
    confirmButton.addEventListener("click", async () => {
      if (!fileInput.files.length) {
        showMessage("Please upload your private key file first.", "error");
        return;
      }
  
      processingMessage.style.display = "flex";
  
      try {
        // ✅ Step 1: Read and import the private key
        const file = fileInput.files[0];
        const text = await file.text();
        const privateKeyJwk = JSON.parse(text);
  
        privateKey = await window.crypto.subtle.importKey(
          "jwk",
          privateKeyJwk,
          { name: "ECDSA", namedCurve: "P-256" },
          true,
          ["sign"]
        );
  
        console.log("✅ Private key loaded!");
  
        // ✅ Optional: Save to localStorage for session
        localStorage.setItem("nodePrivateKey", JSON.stringify(privateKeyJwk));
  
        // ✅ Step 2: Get Node ID
        const nodeId = localStorage.getItem("currentNodeId");
        if (!nodeId) {
          throw new Error("No node ID found in localStorage. Please register your node first.");
        }
  
        // ✅ Step 3: Request challenge from server
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
          {
            name: "ECDSA",
            hash: { name: "SHA-256" },
          },
          privateKey,
          challengeBuffer
        );
  
        const signatureHex = Array.from(new Uint8Array(signature))
          .map(byte => byte.toString(16).padStart(2, '0'))
          .join('');
  
        console.log("✅ Signature generated:", signatureHex);
  
        // ✅ Step 5: Send signature to verify
        const verifyResponse = await fetch(`/verify-challenge/${nodeId}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ signature: signatureHex })
        });
  
        const verifyData = await verifyResponse.json();
  
        if (!verifyResponse.ok) {
          throw new Error(verifyData.detail || verifyData.error || "Verification failed.");
        }
  
        showMessage("✅ Node verified successfully! Redirecting...", "success");
  
        setTimeout(() => window.location.href = "/node", 2000);
  
      } catch (err) {
        console.error("Connect node error:", err);
        showMessage(err.message, "error");
      } finally {
        processingMessage.style.display = "none";
      }
    });
  }
  