// src/frontend/static/js/distribution/modalHandler.js
//
// Node detail modal: shows a node's specs and lets you send it work.
//
// Built with createElement + textContent rather than innerHTML — node names and
// specs come from other people's machines, so treating them as markup would let
// a malicious node run script in everyone else's dashboard.

import { dataKindsList } from "../component/dataKinds.js";
import { submitterHeaders } from "./submitter.js";
import { buildJobForm, jobSchema } from "./jobForm.js";

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

  const datasetLabel = el("label", "field-label", "Your training data");
  datasetLabel.htmlFor = "datasetFile";
  content.appendChild(datasetLabel);

  // What each kind of file gives you, rather than a paragraph describing both
  // at once. The choice of file *is* the choice of model, so the two are shown
  // together.
  //
  // From component/dataKinds.js, which the node page's self-test also uses.
  // The rule about the last column being the label was written out in both
  // places, in different words, and only one of them would have been updated.
  content.appendChild(dataKindsList(["csv", "text"]));

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.id = "datasetFile";
  fileInput.accept = ".csv,.txt,.md,text/csv,text/plain";
  // More than one, and more than once. How much data you bring is the
  // strongest thing you control here, and bringing more used to mean joining
  // the files by hand before uploading.
  fileInput.multiple = true;
  content.appendChild(fileInput);

  const datasetStatus = el("div", "field-status");
  content.appendChild(datasetStatus);

  // Said before the job is sent, not after it comes back. A corpus too small
  // to learn from still trains and still passes verification, so the only
  // moment this helps anyone is now.
  const datasetAdvice = el("p", "field-advice");
  content.appendChild(datasetAdvice);

  const addMore = el("p", "field-hint add-more",
    "Found more? Choose another file and it is added to this one.");
  addMore.hidden = true;
  content.appendChild(addMore);

  const privacy = el("details", "data-privacy");
  privacy.appendChild(el("summary", null, "Who can see this data"));

  const privacyBody = el("div", "data-privacy-body");
  privacyBody.appendChild(el("p", "data-privacy-warning",
    "The contributor running your job can read the numbers in this file. "
    + "Training needs the data in the clear on their GPU, so there is no way "
    + "around it. Do not send anything you would not hand to a stranger."));

  const facts = el("ul");
  let storageFact = null;
  [
    "Column names are not sent. The node receives unlabelled numbers, not "
      + "\u201csalary\u201d or \u201cdiagnosis\u201d.",
    // Replaced below with whichever is true. It used to assert encryption
    // flatly, which is a promise this page cannot make on its own: it holds
    // only if the operator set an encryption key, and on a deployment
    // without one the sentence was simply false. Overstating protection is
    // the one mistake this panel exists to avoid.
    "It is stored on the coordinator; whether that copy is encrypted "
      + "depends on how this service is run.",
    "The node keeps it in memory only and never writes it to disk.",
    // Named because it is the one thing that does travel with a name on it.
    // The point above says column names are not sent, which is true and which
    // makes it easy to assume nothing legible goes; the job's name does, and a
    // job called “hospital-readmission-risk” says plenty.
    "The name you give the job is shown to the contributor running it. "
      + "The data is not labelled, but that name is.",
    "Both copies are deleted once your job has finished and been checked.",
    "Part of it is held back from the node and used to verify the result.",
  ].forEach((text, index) => {
    const item = el("li", null, text);
    if (index === 1) storageFact = item;       // the storage line
    facts.appendChild(item);
  });
  privacyBody.appendChild(facts);

  // Ask the service what it actually does, and say that. Until the answer
  // arrives the cautious wording above stands.
  jobSchema()
    .then((schema) => {
      if (!storageFact) return;
      storageFact.textContent = schema.artifacts_encrypted
        ? "It is stored encrypted on the coordinator, so a database dump "
          + "does not expose it."
        : "It is stored on the coordinator without encryption, so anyone "
          + "who can read that database can read it.";
    })
    .catch(() => { /* leave the cautious wording in place */ });

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
  // Nothing to send until there is data to train on. A job used to be
  // sendable with no file, which ran the model on made-up numbers and proved
  // only that the plumbing worked -- and a contributor now answers that
  // question for themselves, on their own machine, from their node page.
  sendButton.disabled = true;
  content.appendChild(sendButton);

  const sendHint = el("p", "field-hint send-hint",
    "Choose a file above to send this job.");
  content.appendChild(sendHint);

  const responseMessage = el("div");
  responseMessage.id = "taskResponseMessage";
  content.appendChild(responseMessage);

  modal.classList.remove("hidden");

  // --- behaviour -------------------------------------------------------

  let datasetId = null;

  function setReady(ready) {
    sendButton.disabled = !ready;
    sendHint.textContent = ready ? "" : "Choose a file above to send this job.";
  }

  function setStatus(target, message, kind) {
    target.replaceChildren();
    if (!message) return;
    target.appendChild(el("span", kind ? `${kind}-message` : null, message));
  }

  // Which converter the coordinator should run. A .csv is rows of numbers; a
  // .txt is a stream of characters. They become completely different datasets,
  // so this is decided by the file rather than left to a setting somebody has
  // to remember to change.
  function formatFor(name) {
    return /\.(txt|md|text)$/i.test(name || "") ? "text" : "csv";
  }

  // What a dataset looks like once the coordinator has read it.
  function describe(data, format) {
    const number = (value) => value?.toLocaleString?.() ?? value;

    if (format === "text") {
      return [
        `${number(data.tokens)} characters`,
        `${number(data.rows)} sequences of ${data.seq_len}`,
      ];
    }

    const parts = [`${number(data.rows)} rows`];
    if (data.features) parts.push(`${data.features} features`);
    if (data.classes) parts.push(`${data.classes} classes`);
    if (data.class_names?.length) parts.push(`(${data.class_names.join(", ")})`);
    return parts;
  }

  // The first file becomes the dataset; every one after is added to it.
  async function send(file) {
    const format = formatFor(file.name);
    const url = datasetId
      ? `/artifacts/${encodeURIComponent(datasetId)}/append?format=${format}`
      : `/artifacts?kind=dataset&format=${format}`;

    // The key goes with the upload, not just with the job that follows it.
    // Storing a dataset is the expensive half, and it used to be the half that
    // asked for nothing at all.
    const res = await fetch(url, {
      method: "POST",
      headers: submitterHeaders({ "Content-Type": "text/plain" }),
      body: await file.text(),
    });

    const data = await res.json();
    if (!res.ok || !data.artifact_id) {
      const detail = data?.detail?.detail || data?.detail || data?.message;
      throw new Error(detail || `Upload failed (${res.status})`);
    }

    // Adding returns a *new* id: the dataset already uploaded may be attached
    // to a queued job, so it is never edited in place.
    datasetId = data.artifact_id;
    return data;
  }

  fileInput.addEventListener("change", async () => {
    const files = [...(fileInput.files || [])];
    if (!files.length) return;

    const format = formatFor(files[0].name);
    setReady(false);

    let data = null;
    try {
      for (const [index, file] of files.entries()) {
        setStatus(datasetStatus, files.length > 1
          ? `Reading ${file.name} (${index + 1} of ${files.length})…`
          : `Reading ${file.name}…`);
        data = await send(file);
      }

      const parts = describe(data, format);
      if (data.parts > 1) parts.push(`from ${data.parts} files`);

      setStatus(datasetStatus, `Ready: ${parts.join(", ")}`, "success");
      datasetAdvice.textContent = data.advice || "";
      // The row count lets the form size the run and show what it will do,
      // before the job is sent rather than in the reply after it.
      if (jobForm?.suggest) jobForm.suggest(format, { rows: data.rows });
      setReady(true);

      // Cleared so the control is ready for the next file rather than showing
      // the last one as though it were the whole dataset.
      fileInput.value = "";
      addMore.hidden = false;
    } catch (error) {
      console.error("Dataset upload failed:", error);
      setStatus(datasetStatus, error.message, "error");
      fileInput.value = "";
      // A refused file leaves whatever was already accepted intact.
      setReady(Boolean(datasetId));
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

      // Always true now that a job cannot be sent without data.
      const note = " Part of your data is held back to check the result.";

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

      // Anything the coordinator thought worth mentioning about the shape of
      // the run -- too many passes over too little data, most often.
      (result.notes || []).forEach((text) => {
        responseMessage.appendChild(el("p", "field-advice", text));
      });

      // The job is now out of sight; say where it reappears, and where the
      // finished model will be waiting.
      const link = el("a", "modal-followup", "Track it in your workspace →");
      link.href = "/workspace";
      responseMessage.appendChild(link);
    } catch (error) {
      console.error("Error sending job:", error);
      setStatus(responseMessage, error.message, "error");
    } finally {
      setReady(Boolean(datasetId));
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
