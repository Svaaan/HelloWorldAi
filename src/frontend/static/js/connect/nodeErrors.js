// What to say when registering or reconnecting a node does not work, and where
// to send someone next.
//
// The front door used to answer this question before it was asked. It called
// /local-node on load and, where no agent was running, took the two buttons
// away and put a "Set up your machine" link in their place -- next to the Setup
// guide link that was already in the corner, on a card whose whole subject is
// setting up a machine.
//
// The check was right about the facts and wrong about the moment. Somebody
// reading that card has not decided anything yet; somebody who has just pressed
// "Create key file" has. So the page keeps its two doors, and the missing agent
// is explained where it actually gets in the way, with the guide offered as the
// next step rather than as a replacement for the thing they came to do.
//
// The proxy marks that case with `no_local_node` on its 503. Matching the
// message text instead would mean nobody could reword it.

/** Did this fail because no node agent is running beside this dashboard? */
export function isMissingLocalNode(response, body) {
  return response?.status === 503 && Boolean(body?.no_local_node);
}

/**
 * Build the error to throw for a failed node call.
 *
 * Carries `offerSetupGuide` so the catch that finally displays it knows whether
 * this is a dead end or a signpost.
 */
export function nodeCallError(response, body, fallback) {
  const error = new Error(
    body?.detail || body?.error || fallback
    || `Request failed: ${response?.status}`);

  if (isMissingLocalNode(response, body)) error.offerSetupGuide = true;
  return error;
}

/**
 * Write a message into `target`, with a link to the guide when one helps.
 *
 * textContent throughout: these messages carry the server's `detail`, and some
 * of those are built from a node_id taken out of the URL.
 */
export function showNodeMessage(target, message, kind = "success", options = {}) {
  if (!target) return;

  target.replaceChildren();
  target.className = kind === "success" ? "success-message" : "error-message";

  const line = document.createElement("span");
  line.textContent = message;
  target.appendChild(line);

  if (!options.offerSetupGuide) return;

  // A link, not a redirect. They pressed a button on this card a moment ago;
  // moving the page out from under them is not an answer to "that did not
  // work".
  const guide = document.createElement("a");
  guide.href = "/setup";
  guide.className = "message-action";
  guide.textContent = "Read the setup guide";
  target.append(document.createTextNode(" "), guide);
}
