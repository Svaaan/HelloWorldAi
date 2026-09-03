// Taking a node back, with the key this browser already has.
//
// A node agent keeps no identity across a restart. It is handed one by the
// browser: /connect-node tells it which public key it is answering for,
// /finalize-connection records it as connected, and the session token issued by
// the challenge-response is what lets it heartbeat. Stop the agent -- reboot,
// `docker compose down`, close the laptop -- and all of that is gone. The
// coordinator stops hearing from it and marks it disconnected, which is
// correct.
//
// What was wrong was the way back. The node page said "reconnect it from the
// front page with your key file", and the front page asked for a file. But the
// keypair is in this browser's localStorage -- it has been since registration,
// which is the whole reason the page can offer to download it. Asking for the
// file to do something the browser can already do left somebody who could not
// find that file with one apparent option: register again, and strand the node
// they already had.
//
// So: the file is for another machine. On the machine that registered the node,
// this is enough.

import { nodeCallError } from "./nodeErrors.js";
import { establishNodeSession } from "./nodeSession.js";

const PRIVATE_KEY = "nodePrivateKey";
const PUBLIC_KEY = "nodePublicKeyBase64";

/** The keypair this browser holds for its node, or null. */
export function storedNodeKeys() {
  try {
    const privateKeyRaw = localStorage.getItem(PRIVATE_KEY);
    const publicKeyBase64 = localStorage.getItem(PUBLIC_KEY);
    if (!privateKeyRaw || !publicKeyBase64) return null;
    return { privateKeyJwk: JSON.parse(privateKeyRaw), publicKeyBase64 };
  } catch {
    return null;                     // unreadable or refused: same as absent
  }
}

/** Whether taking the node back needs nothing but this browser. */
export function canReconnectHere() {
  return storedNodeKeys() !== null;
}

async function importPrivateKey(privateKeyJwk) {
  // Keys exported before key_ops was set come back unusable for signing.
  if (!privateKeyJwk.key_ops) privateKeyJwk.key_ops = ["sign"];

  return window.crypto.subtle.importKey(
    "jwk",
    privateKeyJwk,
    { name: "ECDSA", namedCurve: "P-256" },
    true,
    ["sign"],
  );
}

/**
 * Hand a running agent its identity back and prove ownership of it.
 *
 * The same four steps the key-file dialog runs, in the same order, because they
 * have to happen in that order: the agent has to know which key it answers for
 * before the coordinator can be told it is connected, and the session token has
 * to exist before it can heartbeat.
 *
 * @param {object} keys  from storedNodeKeys(), or parsed out of a key file
 * @returns {Promise<string>} the node id now connected
 */
export async function reconnectNode({ privateKeyJwk, publicKeyBase64 }) {
  const privateKey = await importPrivateKey(privateKeyJwk);

  // 1. Tell the agent beside this dashboard which node it is.
  const connect = await fetch("/connect-node", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_name: "Reconnected node",
                           public_key: publicKeyBase64 }),
  });
  const connectData = await connect.json().catch(() => ({}));

  // nodeCallError, not a hand-built Error: it carries the proxy's
  // `no_local_node` marker through as `offerSetupGuide`, which is what puts a
  // link to the setup guide on the one failure that is really an instruction.
  // Building the error here by hand dropped that, and the link with it.
  if (!connect.ok) {
    throw nodeCallError(connect, connectData,
                        "No node agent is running on this machine.");
  }
  if (connectData.status === "rejected") {
    throw new Error(connectData.reason || "The node agent refused to start.");
  }

  // 2. Which node this key belongs to, according to the coordinator.
  const found = await fetch("/find-node-id", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ public_key: publicKeyBase64 }),
  });
  const foundData = await found.json().catch(() => ({}));

  if (!found.ok || !foundData.node_id) {
    throw new Error(
      "The coordinator has no node registered to this key. If you registered "
      + "it against a different server, this is the wrong one.");
  }

  // 3. Prove the key, and get the session the agent heartbeats with.
  await establishNodeSession(foundData.node_id, privateKey);

  // 4. Refresh what the machine can do, and mark it connected.
  const finalize = await fetch("/finalize-connection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ public_key: publicKeyBase64 }),
  });
  const finalizeData = await finalize.json().catch(() => ({}));

  if (!finalize.ok || !finalizeData.node_id) {
    throw new Error(finalizeData.detail || finalizeData.error
                    || "The coordinator would not mark this node connected.");
  }

  return finalizeData.node_id;
}

/** Reconnect using whatever this browser already holds. */
export async function reconnectFromThisBrowser() {
  const keys = storedNodeKeys();
  if (!keys) throw new Error("No node key in this browser.");
  return reconnectNode(keys);
}
