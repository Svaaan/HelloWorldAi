// The form for describing a job.
//
// This used to be a textarea holding raw JSON. To send anything you had to
// already know the field names, which of the two architectures existed, and
// what each one expected -- none of which appeared anywhere on the page.
//
// The fields are generated from /job-schema, the same definition the
// coordinator validates against, so the form cannot drift out of step with what
// the server will accept.
//
// Hand-written JSON is still available behind a toggle: the form covers the
// common case, it should not remove a capability people already had.

let schemaPromise = null;

// The coordinator holds this share of every dataset back to verify the result
// with, so it is not part of what the model reads.
const HOLDOUT_FRACTION = 0.2;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function loadSchema() {
  // Fetched once per page; the definition does not change under us.
  if (!schemaPromise) {
    schemaPromise = fetch("/job-schema")
      .then((res) => {
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        return res.json();
      })
      .catch((error) => {
        schemaPromise = null;       // let a later attempt retry
        throw error;
      });
  }
  return schemaPromise;
}

function field(spec, value) {
  const wrap = el("div", "job-field");

  const label = el("label", "job-field-label", spec.label || spec.name);
  label.htmlFor = `field-${spec.name}`;
  wrap.appendChild(label);

  const input = document.createElement("input");
  input.type = "number";
  input.id = `field-${spec.name}`;
  input.name = spec.name;
  input.value = value ?? spec.default;
  input.min = spec.min;
  input.max = spec.max;
  if (spec.type === "float") input.step = "any";
  wrap.appendChild(input);

  if (spec.hint) wrap.appendChild(el("p", "job-field-hint", spec.hint));

  // The hint says what the box means; the example says what to put in it.
  //
  // Somebody who came to find out what this is gets "Neurons per hidden
  // layer." and no way to tell whether 64 is small, ordinary or absurd -- so
  // they either leave every default alone or pick a number out of the air, and
  // then judge the whole service on what comes back.
  if (spec.example) {
    const example = el("p", "job-field-example");
    example.appendChild(el("span", "job-field-example-tag", "Example"));
    example.appendChild(document.createTextNode(" " + spec.example));
    wrap.appendChild(example);
  }

  return { wrap, input };
}

/** The service description, for callers that need a fact from it. */
export function jobSchema() {
  return loadSchema();
}

export async function buildJobForm(container, { modelName } = {}) {
  const schema = await loadSchema();
  const inputs = new Map();

  // --- name ---------------------------------------------------------------
  const nameWrap = el("div", "job-field");
  const nameLabel = el("label", "job-field-label", "Model name");
  nameLabel.htmlFor = "field-model_name";
  nameWrap.appendChild(nameLabel);

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.id = "field-model_name";
  nameInput.value = modelName || "my-first-model";
  nameInput.maxLength = 80;
  nameWrap.appendChild(nameInput);
  container.appendChild(nameWrap);

  // --- architecture -------------------------------------------------------
  const archWrap = el("div", "job-field");
  const archLabel = el("label", "job-field-label", "Model type");
  archLabel.htmlFor = "field-architecture";
  archWrap.appendChild(archLabel);

  const archSelect = document.createElement("select");
  archSelect.id = "field-architecture";
  Object.entries(schema.architectures).forEach(([key, definition]) => {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = definition.label;
    archSelect.appendChild(option);
  });
  archWrap.appendChild(archSelect);

  const archSummary = el("p", "job-field-hint");
  archWrap.appendChild(archSummary);
  container.appendChild(archWrap);

  // --- per-architecture fields -------------------------------------------
  const specGrid = el("div", "job-field-grid");
  container.appendChild(specGrid);

  const derivedNote = el("p", "job-derived-note");
  container.appendChild(derivedNote);

  // --- about the data ------------------------------------------------------
  //
  // Above the training settings, because it is a question about what was
  // uploaded rather than how to train on it -- and because the answer changes
  // how the result is judged, which is worth deciding before choosing a step
  // count.
  const dataQuestions = new Map();
  (schema.data_questions || []).forEach((spec) => {
    const wrap = el("div", "job-check");

    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = `field-${spec.name}`;
    input.checked = Boolean(spec.default);

    const label = document.createElement("label");
    label.htmlFor = input.id;
    label.className = "job-check-label";
    label.textContent = spec.label || spec.name;

    wrap.appendChild(input);
    wrap.appendChild(label);
    if (spec.hint) wrap.appendChild(el("p", "job-field-hint", spec.hint));
    if (spec.example) {
      const example = el("p", "job-field-example");
      example.appendChild(el("span", "job-field-example-tag", "Example"));
      example.appendChild(document.createTextNode(" " + spec.example));
      wrap.appendChild(example);
    }

    dataQuestions.set(spec.name, input);
    container.appendChild(wrap);
  });

  // --- training -----------------------------------------------------------
  container.appendChild(el("h4", "job-section-title", "Training"));

  const hyperGrid = el("div", "job-field-grid");
  const hyperInputs = new Map();
  container.appendChild(hyperGrid);

  // What this run will actually do to the uploaded data, updated as the
  // numbers are typed. The coordinator says the same thing in its reply, but
  // by then the job is already queued -- which makes it a post-mortem rather
  // than a decision.
  const runShape = el("p", "job-run-shape");
  container.appendChild(runShape);

  // How long this is likely to take, from what the network has actually
  // managed on jobs like it. Fetched once; the estimate does not change while
  // somebody is typing, only the arithmetic on top of it does.
  const runTime = el("p", "job-run-time");
  container.appendChild(runTime);

  // What the machines on the network will accept. A job larger than every one
  // of them is refused by the coordinator, so the form says so first rather
  // than letting somebody fill in a number that cannot be sent.
  const limits = el("p", "job-limits");
  container.appendChild(limits);

  let biggest = null;
  fetch("/nodes")
    .then((res) => (res.ok ? res.json() : []))
    .then((nodes) => { biggest = largestLimits(nodes); describeRun(); })
    .catch(() => { /* the coordinator still enforces it; this is a courtesy */ });

  let throughput = null;
  fetch("/throughput?architecture=mlp")
    .then((res) => (res.ok ? res.json() : null))
    .then((body) => { throughput = body; describeRun(); })
    .catch(() => { /* an estimate is a nicety; its absence is not an error */ });

  // Rows as uploaded; part is held back for verification and never reaches
  // the node, so the model reads fewer than the file contains.
  let datasetRows = 0;
  // The input width, once a file has been chosen. Before that the first
  // layer's size is unknown, so the parameter count below assumes it
  // matches the hidden width -- which over-counts and never under-counts.
  let datasetFeatures = 0;

  function trainingRows() {
    return Math.max(1, Math.round(datasetRows * (1 - HOLDOUT_FRACTION)));
  }

  function describeRun() {
    if (!datasetRows) {
      runShape.textContent = "";
      return;
    }

    const steps = readNumber(hyperInputs.get("steps").input, hyperInputs.get("steps").spec);
    const batch = readNumber(hyperInputs.get("batch_size").input, hyperInputs.get("batch_size").spec);
    const rows = trainingRows();
    const samples = steps * batch;

    // Batches are drawn with replacement, so n draws do not touch n distinct
    // rows. 1 - e^(-n/N) is the share actually reached.
    const coverage = 1 - Math.exp(-samples / rows);
    const passes = samples / rows;
    const thin = coverage < (schema.guidance?.min_coverage ?? 0.95);

    // Say which rows these are. The panel above reports the number of rows in
    // the file; this one counts only the training half, and the two numbers
    // sitting a few centimetres apart with no explanation read as a bug rather
    // than as a holdout.
    const held = Math.max(0, datasetRows - rows);
    const heldNote = held
      ? ` The other ${held.toLocaleString()} ${held === 1 ? "row is" : "rows are"} `
        + `held back to check the result.`
      : "";

    describeTime(samples);
    describeLimits();

    runShape.classList.toggle("is-thin", thin);
    runShape.textContent = (thin
      ? `Draws ${samples.toLocaleString()} samples from ${rows.toLocaleString()} `
        + `training rows — about ${Math.round(coverage * 100)}% of them. `
        + `More steps would use the rest.`
      : `Reads your ${rows.toLocaleString()} training rows about `
        + `${passes.toFixed(1)} times (${samples.toLocaleString()} samples).`)
      + heldNote;
  }

  /** Turn a sample count into minutes, or say why it cannot.
   *
   * There was no estimate at all, and "more steps means a longer job" does not
   * help anybody choose between 2,000 and 20,000. This uses the median
   * throughput of finished jobs of the same kind rather than a card's
   * theoretical TFLOPS, which on a two-layer network at batch 32 is wrong by
   * about a factor of ten -- the time goes on overhead, not arithmetic.
   */
  function describeTime(samples) {
    if (!throughput) { runTime.textContent = ""; return; }

    if (!throughput.samples_per_second) {
      runTime.textContent =
        "No time estimate yet — " + (throughput.why || "not enough finished jobs")
        + ".";
      return;
    }

    const seconds = samples / throughput.samples_per_second;
    const spread = (throughput.slowest && throughput.fastest
      && throughput.fastest > throughput.slowest * 2)
      ? " Machines on this network vary a lot, so treat it loosely."
      : "";

    const shown = seconds < 90
      ? `${Math.max(1, Math.round(seconds))} seconds`
      : seconds < 5400
        ? `${Math.round(seconds / 60)} minutes`
        : `${(seconds / 3600).toFixed(1)} hours`;

    runTime.textContent =
      `Roughly ${shown} of somebody's graphics card, based on `
      + `${throughput.based_on} finished job${throughput.based_on === 1 ? "" : "s"}.`
      + spread;
  }

  /** The most generous limits on the network, since a job goes to one machine.
   *
   * The largest rather than the smallest: a model too big for an 8GB card is
   * still sendable if a 24GB one is free, and the coordinator now picks from
   * the machines that can take it rather than assigning first and refusing
   * after.
   */
  function largestLimits(nodes) {
    const all = (Array.isArray(nodes) ? nodes : [])
      .filter((node) => node.isAvailable !== false)
      .map((node) => node.capabilities?.limits)
      .filter(Boolean);

    if (!all.length) return null;

    return {
      max_model_parameters: Math.max(...all.map((l) => l.max_model_parameters || 0)),
      max_batch_size: Math.max(...all.map((l) => l.max_batch_size || 0)),
      max_steps: Math.max(...all.map((l) => l.max_steps || 0)),
      machines: all.length,
    };
  }

  /** Roughly how many weights the current form describes.
   *
   * The same count the coordinator makes, so the two agree about whether a job
   * fits. Over-counts slightly before a file is chosen, because the input width
   * is read from the data.
   */
  function parameterCount() {
    const definition = schema.architectures[archSelect.value];
    const read = (name) => {
      const field = inputs.get(name);
      return field ? readNumber(field.input, field.spec) : 0;
    };

    if (archSelect.value === "transformer") {
      const width = read("d_model");
      const layers = Math.max(1, read("n_layer"));
      return layers * 12 * width * width + width * width * 4;
    }

    const hidden = read("hidden_dim");
    const depth = Math.max(1, read("depth"));
    const inputDim = datasetFeatures || hidden;
    const outputDim = 2;

    return inputDim * hidden + hidden
      + (depth - 1) * (hidden * hidden + hidden)
      + hidden * outputDim + outputDim;
  }

  function describeLimits() {
    if (!biggest) { limits.textContent = ""; return; }

    const wanted = parameterCount();
    const allowed = biggest.max_model_parameters;
    const fits = !allowed || wanted <= allowed;

    limits.classList.toggle("is-over", !fits);
    limits.textContent = fits
      ? `About ${wanted.toLocaleString()} parameters. The largest machine `
        + `offering right now takes ${allowed.toLocaleString()}.`
      : `About ${wanted.toLocaleString()} parameters, and the largest machine `
        + `on the network takes ${allowed.toLocaleString()}. This will be `
        + `refused — reduce the width or the number of layers.`;
  }

  function renderArchitecture() {
    const definition = schema.architectures[archSelect.value];
    archSummary.textContent = definition.summary || "";

    specGrid.replaceChildren();
    inputs.clear();

    definition.fields.forEach((spec) => {
      const { wrap, input } = field(spec);
      inputs.set(spec.name, { input, spec });
      specGrid.appendChild(wrap);
    });

    // Saying what is filled in automatically stops people hunting for an
    // input that deliberately is not there.
    derivedNote.textContent = definition.derived_note || "";

    // A learning rate that suits a small classifier will not train a
    // transformer at all, so the starting values follow the model rather than
    // being one set shared by both.
    const overrides = definition.hyperparameter_defaults || {};

    // And so do the examples, for the same reason: "0.01 for a feedforward
    // network" printed under a box reading 0.0005 is not guidance, it is a
    // contradiction the reader has to resolve.
    const exampleOverrides = definition.hyperparameter_examples || {};

    hyperGrid.replaceChildren();
    hyperInputs.clear();
    schema.hyperparameters.forEach((spec) => {
      const withDefault = (spec.name in overrides || spec.name in exampleOverrides)
        ? {
            ...spec,
            default: spec.name in overrides ? overrides[spec.name] : spec.default,
            example: spec.name in exampleOverrides
              ? exampleOverrides[spec.name] : spec.example,
          }
        : spec;
      const { wrap, input } = field(withDefault);
      input.addEventListener("input", describeRun);
      hyperInputs.set(spec.name, { input, spec: withDefault });
      hyperGrid.appendChild(wrap);
    });

    describeRun();
  }

  archSelect.addEventListener("change", renderArchitecture);
  renderArchitecture();

  // --- raw JSON escape hatch ---------------------------------------------
  const advanced = el("details", "job-advanced");
  advanced.appendChild(el("summary", null, "Edit as JSON instead"));

  const advancedBody = el("div", "job-advanced-body");
  advancedBody.appendChild(el("p", "job-field-hint",
    "While this is open, what is written here is what gets sent."));

  const textarea = document.createElement("textarea");
  textarea.id = "taskDataInput";
  textarea.spellcheck = false;
  advancedBody.appendChild(textarea);
  advanced.appendChild(advancedBody);
  container.appendChild(advanced);

  // A number input hands back an empty string when it is blank, or when the
  // browser rejected what was typed -- which happens on locales that display a
  // decimal comma. Number("") is 0 and parseInt("") is NaN, so the server would
  // answer a blank box with "must be between 1e-08 and 10", which explains
  // nothing. Falling back to the value already shown as the default is both
  // truer to what the person saw and easier to act on.
  function readNumber(input, definition) {
    const raw = String(input.value).trim();
    const parsed = definition.type === "float"
      ? Number(raw) : parseInt(raw, 10);

    return raw === "" || !Number.isFinite(parsed) ? definition.default : parsed;
  }

  function readForm() {
    const spec = { architecture: archSelect.value };
    inputs.forEach(({ input, spec: definition }, name) => {
      spec[name] = readNumber(input, definition);
    });

    const hyperparameters = {};
    hyperInputs.forEach(({ input, spec: definition }, name) => {
      hyperparameters[name] = readNumber(input, definition);
    });

    const answers = {};
    dataQuestions.forEach((input, name) => { answers[name] = input.checked; });

    return {
      task_type: schema.default_task_type,
      model_name: nameInput.value.trim() || "model",
      model_spec: spec,
      hyperparameters,
      ...answers,
    };
  }

  // Opening the JSON view shows exactly what the form would have sent, so it
  // is a starting point rather than a blank page.
  advanced.addEventListener("toggle", () => {
    if (advanced.open) textarea.value = JSON.stringify(readForm(), null, 2);
  });

  return {
    /** Point the form at the model that matches the file just uploaded.
     *
     * The two architectures take incompatible data -- rows of numbers against
     * a stream of characters -- so uploading a .txt with the classifier still
     * selected can only end in a refusal. Switching here means the person
     * chooses by picking a file, which is the choice they were already making.
     */
    suggest(format, info) {
      datasetRows = Number(info?.rows) || 0;
      datasetFeatures = Number(info?.features) || 0;

      const match = Object.entries(schema.architectures)
        .find(([, definition]) => definition.accepts === format);

      if (match && archSelect.value !== match[0]) {
        archSelect.value = match[0];
        renderArchitecture();          // rebuilds the fields, then describes
      }

      // Enough steps to at least read the data that was uploaded. Only ever
      // upward: training less over a small corpus was measured on this
      // service and produced a worse model, not a less overfitted one.
      const steps = hyperInputs.get("steps");
      const batch = hyperInputs.get("batch_size");
      if (datasetRows && steps && batch) {
        const perBatch = readNumber(batch.input, batch.spec);
        const wanted = Math.ceil(
          ((schema.guidance?.target_passes ?? 3) * trainingRows()) / perBatch
        );
        const capped = Math.min(
          Math.max(steps.spec.default, wanted),
          schema.guidance?.max_suggested_steps ?? 20000,
          steps.spec.max,
        );
        steps.input.value = capped;
      }

      describeRun();
    },

    /** The job to submit. Throws if the JSON view holds something unparseable. */
    read() {
      if (!advanced.open) return readForm();

      try {
        return JSON.parse(textarea.value);
      } catch {
        throw new Error("The JSON below is not valid. Fix it, or close that section to use the form.");
      }
    },
  };
}
