// src/frontend/static/js/distribution/modalHandler.js
//
// Node detail modal: shows a node's specs and lets you send it work.
//
// Built with createElement + textContent rather than innerHTML — node names and
// specs come from other people's machines, so treating them as markup would let
// a malicious node run script in everyone else's dashboard.

import { submitterHeaders } from "./submitter.js";
import { buildJobForm } from "./jobForm.js";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function specRow(label, value) {
  const row = el("p");
  row.appendChild(el("strong", null, label));
  row.appendChild(el("span", null, value));
  return row;
}

function describeGpus(node) {
  const gpus = node.capabilities?.gpu;
  if (Array.isArray(gpus) && gpus.length) {
    return gpus.map((g) => g.name || "Unknown GPU").join(", ");
  }
  return gpus?.name || "None";
}

// `node` is null when the submitter did not choose a machine: the coordinator
// picks one at submit time. Everything below that touches a specific node is
// guarded on it.
export function showNodeModal(node) {
  const auto = !node;
  const modal = document.getElementById("nodeModal");
  const content =
    document.getElementById("nodeModalDetails") ||
    document.getElementById("modalNodeDetails");

  content.replaceChildren();

  if (auto) {
    content.appendChild(el("h3", null, "Send to the best available node"));
    content.appendChild(el("p", "field-hint",
      "The coordinator picks whichever machine can start soonest, preferring "
      + "idle GPUs over fast but busy ones. Pick a node yourself from the list "
      + "if you would rather choose."));
  } else {
    content.appendChild(el("h3", null, node.node_id));

    const cpu = node.capabilities?.cpu || {};
    content.appendChild(specRow("CPU", `${cpu.brand || "Unknown"} (${cpu.cores ?? "-"} cores)`));
    content.appendChild(specRow("GPU", describeGpus(node)));

    const tflops = node.total_gpu_tflops;
    if (tflops) content.appendChild(specRow("Pooled compute", `${Number(tflops).toFixed(2)} TFLOPS`));

    content.appendChild(specRow("Status", node.isAvailable ? "Available" : "Unavailable"));
  }

  // --- dataset ---------------------------------------------------------

  const datasetLabel = el("label", "field-label", "Training data (CSV, optional)");
  datasetLabel.htmlFor = "datasetFile";
  content.appendChild(datasetLabel);

  const hint = el(
    "p",
    "field-hint",
    "Every column is a feature except the last, which is the label. " +
      "Without a file the node trains on synthetic data, which only proves it works."
  );
  content.appendChild(hint);

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.id = "datasetFile";
  fileInput.accept = ".csv,text/csv";
  content.appendChild(fileInput);

  const datasetStatus = el("div", "field-status");
  content.appendChild(datasetStatus);

  const privacy = el("details", "data-privacy");
  privacy.appendChild(el("summary", null, "Who can see this data"));

  const privacyBody = el("div", "data-privacy-body");
  privacyBody.appendChild(el("p", "data-privacy-warning",
    "The contributor running your job can read the numbers in this file. "
    + "Training needs the data in the clear on their GPU, so there is no way "
    + "around it. Do not send anything you would not hand to a stranger."));

  const facts = el("ul");
  [
    "Column names are not sent. The node receives unlabelled numbers, not "
      + "\u201csalary\u201d or \u201cdiagnosis\u201d.",
    "It is stored encrypted here, so a database dump does not expose it.",
    "The node keeps it in memory only and never writes it to disk.",
    "Both copies are deleted once your job has finished and been checked.",
    "Part of it is held back from the node and used to verify the result.",
  ].forEach((text) => facts.appendChild(el("li", null, text)));
  privacyBody.appendChild(facts);

  privacy.appendChild(privacyBody);
  content.appendChild(privacy);

  // --- job spec --------------------------------------------------------

  const jobLabel = el("label", "field-label", "Job");
  content.appendChild(jobLabel);

  const formHost = el("div", "job-form");
  content.appendChild(formHost);

  // Generated from the coordinator's own schema, so the fields offered here
  // are exactly the fields it will accept.
  let jobForm = null;
  buildJobForm(formHost)
    .then((form) => { jobForm = form; })
    .catch((error) => {
      console.error("Could not load the job form:", error);
      formHost.replaceChildren(
        el("p", "error-message",
           `Could not load the job options. ${error.message}`)
      );
    });

  const sendButton = el("button", null, "Send job");
  sendButton.id = "sendTaskButton";
  sendButton.type = "button";
  content.appendChild(sendButton);

  const responseMessage = el("div");
  responseMessage.id = "taskResponseMessage";
  content.appendChild(responseMessage);

  modal.classList.remove("hidden");

  // --- behaviour -------------------------------------------------------

  let datasetId = null;

  function setStatus(target, message, kind) {
    target.replaceChildren();
    if (!message) return;
    target.appendChild(el("span", kind ? `${kind}-message` : null, message));
  }

  fileInput.addEventListener("change", async () => {
    datasetId = null;
    const file = fileInput.files?.[0];
    if (!file) {
      setStatus(datasetStatus, "");
      return;
    }

    setStatus(datasetStatus, `Uploading ${file.name}…`);

    try {
      const res = await fetch("/artifacts?kind=dataset&format=csv", {
        method: "POST",
        headers: { "Content-Type": "text/csv" },
        body: await file.text(),
      });

      const data = await res.json();

      if (!res.ok || !data.artifact_id) {
        const detail = data?.detail?.detail || data?.detail || data?.message;
        throw new Error(detail || `Upload failed (${res.status})`);
      }

      datasetId = data.artifact_id;

      const parts = [`${data.rows?.toLocaleString?.() ?? data.rows} rows`];
      if (data.features) parts.push(`${data.features} features`);
      if (data.classes) parts.push(`${data.classes} classes`);
      if (data.class_names?.length) parts.push(`(${data.class_names.join(", ")})`);

      setStatus(datasetStatus, `Ready: ${parts.join(", ")}`, "success");
    } catch (error) {
      console.error("Dataset upload failed:", error);
      setStatus(datasetStatus, error.message, "error");
      fileInput.value = "";
    }
  });

  sendButton.addEventListener("click", async () => {
    if (!jobForm) {
      setStatus(responseMessage, "The job options are still loading.", "error");
      return;
    }

    let payload;
    try {
      payload = jobForm.read();
    } catch (error) {
      setStatus(responseMessage, error.message, "error");
      return;
    }

    if (datasetId) payload.dataset_id = datasetId;

    sendButton.disabled = true;
    setStatus(responseMessage, "Sending…");

    try {
      const url = auto
        ? "/submit-task"
        : `/submit-task/${encodeURIComponent(node.node_id)}`;

      const res = await fetch(url, {
        method: "POST",
        // The key identifies this browser as the job's owner, which is what
        // later lets it collect the trained model.
        headers: submitterHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(payload),
      });

      const result = await res.json();

      if (!res.ok || result.status !== "success") {
        const detail = result?.detail?.detail || result?.detail || result?.message;
        throw new Error(detail || "The coordinator refused the job.");
      }

      const note = result.verifiable
        ? " Results will be checked against data the node never sees."
        : " No dataset attached, so the result cannot be verified.";

      // When the coordinator chose the machine, say which one and why --
      // otherwise the job vanishes into the network with no account of where.
      const placement = result.chosen?.summary
        ? `${result.chosen.summary} `
        : "";

      setStatus(
        responseMessage,
        `${placement}Queued as ${result.task_id}. `
        + `The node picks it up on its next poll.${note}`,
        "success"
      );

      // The job is now out of sight; say where it reappears, and where the
      // finished model will be waiting.
      const link = el("a", "modal-followup", "Track it in your workspace →");
      link.href = "/workspace";
      responseMessage.appendChild(link);
    } catch (error) {
      console.error("Error sending job:", error);
      setStatus(responseMessage, error.message, "error");
    } finally {
      sendButton.disabled = false;
    }
  });
}

export function initModalCloseHandler() {
  const modal = document.getElementById("nodeModal");
  const closeBtn = document.getElementById("modalClose");

  const close = () => modal?.classList.add("hidden");

  if (closeBtn) closeBtn.addEventListener("click", close);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });

  // Clicking the backdrop, but not the panel itself, closes the modal.
  modal?.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
}
