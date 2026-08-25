// Shared node session handling.
//
// The private key never leaves the browser. To prove ownership of a node we ask
// the coordinator for a challenge, sign it locally, and exchange the signature
// for a short-lived session token. That token is what authorises every later
// call that mutates this node.

const TOKEN_KEY = "nodeSessionToken";
const NODE_ID_KEY = "currentNodeId";

export function getNodeToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function clearNodeSession() {
  localStorage.removeItem(TOKEN_KEY);
}

// Merge the bearer token into a headers object, if we have one.
export function authHeaders(extra = {}) {
  const token = getNodeToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : { ...extra };
}

function toHex(buffer) {
  return Array.from(new Uint8Array(buffer))
    .map(b => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Run the full challenge-response handshake and store the resulting session.
 *
 * @param {string} nodeId    The node being claimed.
 * @param {CryptoKey} privateKey  ECDSA P-256 private key for that node.
 * @returns {Promise<string>} the session token
 */
export async function establishNodeSession(nodeId, privateKey) {
  // 1. Ask the coordinator for a fresh challenge
  const challengeResponse = await fetch(`/generate-challenge/${nodeId}`);
  const challengeData = await challengeResponse.json();

  if (!challengeResponse.ok || !challengeData.challenge) {
    throw new Error("Failed to retrieve challenge from coordinator.");
  }

  // 2. Sign it with the private key
  const signatureBuffer = await window.crypto.subtle.sign(
    { name: "ECDSA", hash: { name: "SHA-256" } },
    privateKey,
    new TextEncoder().encode(challengeData.challenge)
  );

  // 3. Exchange the signature for a session token
  const verifyResponse = await fetch(`/verify-challenge/${nodeId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ signature: toHex(signatureBuffer) }),
  });

  const verifyData = await verifyResponse.json();

  if (!verifyResponse.ok || verifyData.status !== "success") {
    throw new Error("Verification failed. Ensure you're using the correct private key.");
  }

  if (!verifyData.token) {
    throw new Error("Coordinator verified the node but issued no session token.");
  }

  localStorage.setItem(TOKEN_KEY, verifyData.token);
  localStorage.setItem(NODE_ID_KEY, nodeId);

  // 4. Hand the token to the node process so it can authenticate its heartbeats.
  //    A failure here is not fatal for the browser session, but without it the
  //    coordinator will mark this node disconnected after ~5 minutes.
  try {
    const sessionResponse = await fetch("/node-session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: nodeId, token: verifyData.token }),
    });

    if (!sessionResponse.ok) {
      console.warn("⚠️ Node process did not accept the session token; heartbeats will not run.");
    }
  } catch (err) {
    console.warn("⚠️ Could not reach the node process to hand over the session token:", err);
  }

  return verifyData.token;
}
