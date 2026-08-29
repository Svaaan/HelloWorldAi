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

  return { wrap, input };
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

  // --- training -----------------------------------------------------------
  container.appendChild(el("h4", "job-section-title", "Training"));

  const hyperGrid = el("div", "job-field-grid");
  const hyperInputs = new Map();
  container.appendChild(hyperGrid);

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

    hyperGrid.replaceChildren();
    hyperInputs.clear();
    schema.hyperparameters.forEach((spec) => {
      const withDefault = spec.name in overrides
        ? { ...spec, default: overrides[spec.name] }
        : spec;
      const { wrap, input } = field(withDefault);
      hyperInputs.set(spec.name, { input, spec: withDefault });
      hyperGrid.appendChild(wrap);
    });
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

    return {
      task_type: schema.default_task_type,
      model_name: nameInput.value.trim() || "model",
      model_spec: spec,
      hyperparameters,
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
    suggest(format) {
      const match = Object.entries(schema.architectures)
        .find(([, definition]) => definition.accepts === format);

      if (!match || archSelect.value === match[0]) return;

      archSelect.value = match[0];
      renderArchitecture();
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
