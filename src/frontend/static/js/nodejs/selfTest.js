// Prove this machine works, before a stranger depends on it.
//
// A contributor could register a node and have no way of knowing whether it
// actually trains anything until somebody else's job either ran or did not.
// This runs a real job locally and reports what happened.
//
// It also gives a better throughput number than the startup benchmark. That
// times a burst of matrix multiplications, which flatters the card; a real
// workload sustains considerably less. Since what a submitter cares about is
// how quickly their job finishes, the sustained figure is the honest one --
// so both are shown, with the difference explained rather than hidden.

const POLL_MS = 2000;

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function stat(label, value, hint) {
  const box = el("div", "selftest-stat");
  box.appendChild(el("span", "selftest-stat-label", label));
  box.appendChild(el("span", "selftest-stat-value", value));
  if (hint) box.appendChild(el("span", "selftest-stat-hint", hint));
  return box;
}

function notice(text, kind) {
  return el("p", kind ? `selftest-note ${kind}` : "selftest-note", text);
}

function render(result) {
  const target = document.getElementById("selfTestResult");
  if (!target) return;

  if (!result) {
    target.replaceChildren(notice("Not tested yet."));
    return;
  }

  if (result.running) {
    target.replaceChildren(notice("Training… this takes a few seconds.", "is-running"));
    return;
  }

  if (result.status !== "completed") {
    target.replaceChildren(
      notice(result.result || "The test job did not finish.", "is-error")
    );
    return;
  }

  const grid = el("div", "selftest-stats");

  // A small model on somebody's own rows finishes almost instantly, so its
  // throughput measures start-up cost rather than the card. Reporting it as
  // "sustained" beside the benchmark would invite exactly the wrong
  // comparison, so on a CSV run the figure is left out.
  const sustained = Number(result.sustained_tflops);
  if (!result.used_dataset) {
    grid.appendChild(stat(
      "Sustained",
      sustained ? `${sustained.toFixed(2)} TFLOPS` : "—",
      "on a real workload",
    ));
  }

  const peak = Number(result.peak_tflops);
  if (peak) {
    grid.appendChild(stat("Burst benchmark", `${peak.toFixed(1)} TFLOPS`,
      "short matmul, flatters the card"));
  }

  if (result.seconds) {
    grid.appendChild(stat("Took", `${Number(result.seconds).toFixed(1)}s`,
      result.steps ? `${result.steps} steps` : null));
  }

  if (result.initial_loss != null && result.final_loss != null) {
    grid.appendChild(stat("Loss",
      `${result.initial_loss} → ${result.final_loss}`,
      result.learned ? "it learned" : "no improvement"));
  }

  target.replaceChildren(grid);

  // Whether the loss actually fell is the part that says the card computed
  // something real, rather than merely running without erroring.
  target.appendChild(
    result.learned
      ? notice(
          result.used_dataset
            ? "Your data trains on this machine. It is ready for work."
            : "This machine trains correctly and is ready for work.",
          "is-ok",
        )
      : notice(
          "The job ran but the loss did not fall. That can happen on a very "
          + "short test; run it again, and if it repeats something is wrong.",
          "is-warn",
        )
  );

  if (Array.isArray(result.devices) && result.devices.length) {
    target.appendChild(notice(`Ran on: ${result.devices.join(", ")}`));
  }
}

async function loadExisting() {
  try {
    const res = await fetch("/self-test");
    if (!res.ok) return;
    const data = await res.json();
    if (data.result) render(data.result);
  } catch {
    // No previous result is not worth reporting.
  }
}

function setCsvStatus(message, kind) {
  const target = document.getElementById("selfTestCsvStatus");
  if (!target) return;
  target.replaceChildren();
  if (!message) return;

  const line = document.createElement("span");
  if (kind) line.className = `${kind}-message`;
  line.textContent = message;
  target.appendChild(line);
}

export function initSelfTest() {
  const button = document.getElementById("runSelfTest");
  if (!button) return;

  loadExisting();

  const csvInput = document.getElementById("selfTestCsv");
  if (csvInput) {
    csvInput.addEventListener("change", () => {
      const file = csvInput.files?.[0];
      setCsvStatus(
        file ? `Will train on ${file.name} (${Math.round(file.size / 1024)} kB).` : "",
      );
    });
  }

  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Testing…";
    render({ running: true });

    // The node presents this as its current task, so the live view picks it up
    // on its next poll -- the same overlay a real job opens.
    const file = csvInput?.files?.[0];

    // The request holds open for the whole job, so nothing else is needed to
    // know when it finished.
    try {
      const res = await fetch("/self-test", {
        method: "POST",
        headers: file ? { "Content-Type": "text/csv" } : {},
        body: file ? await file.text() : undefined,
      });
      const data = await res.json().catch(() => null);

      if (!res.ok) {
        const detail = data?.detail?.detail || data?.detail;
        throw new Error(detail || `Server returned ${res.status}`);
      }

      render(data);
    } catch (error) {
      console.error("Self test failed:", error);
      render({ status: "failed", result: error.message });
    } finally {
      button.disabled = false;
      button.textContent = "Run a test job";
    }
  });
}
