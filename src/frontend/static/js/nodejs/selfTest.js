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

import { dataKindsList } from "../component/dataKinds.js";

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

function formatDuration(seconds) {
  const whole = Math.round(seconds);
  if (whole < 60) return `${whole}s`;
  return `${Math.floor(whole / 60)}m ${String(whole % 60).padStart(2, "0")}s`;
}

function render(result) {
  const target = document.getElementById("selfTestResult");
  if (!target) return;

  if (!result) {
    target.replaceChildren(notice("Not tested yet."));
    return;
  }

  if (result.running) {
    target.replaceChildren(notice(
      result.mode === "stress"
        ? "Working the card… watch the temperature in the live view."
        : "Training… this takes a few seconds.",
      "is-running",
    ));
    return;
  }

  if (result.status !== "completed") {
    target.replaceChildren(
      notice(result.result || "The test job did not finish.", "is-error")
    );
    return;
  }

  const grid = el("div", "selftest-stats");

  // A stress run is about heat, not throughput. Lead with what it was for.
  if (result.mode === "stress") {
    if (result.peak_temperature != null) {
      grid.appendChild(stat("Peak temperature", `${result.peak_temperature}°C`,
        result.ran_hot ? "reached its warning point" : "stayed within limits"));
    }
    if (result.peak_power_w != null) {
      grid.appendChild(stat("Peak power", `${result.peak_power_w} W`));
    }
    if (result.peak_utilisation != null) {
      grid.appendChild(stat("Peak load", `${result.peak_utilisation}%`));
    }
    if (result.seconds_run != null) {
      grid.appendChild(stat("Ran for", formatDuration(result.seconds_run),
        result.ended_because === "stopped" ? "you stopped it" : null));
    }

    target.replaceChildren(grid);
    target.appendChild(
      result.ran_hot
        ? notice(
            "The card reached its warning temperature. It is protected — a real "
            + "job would pause before any damage — but better airflow would let "
            + "it work for longer.",
            "is-warn",
          )
        : notice(
            "The card held up under sustained load without getting close to its "
            + "limit. This machine can take long jobs.",
            "is-ok",
          )
    );
    if (Array.isArray(result.devices) && result.devices.length) {
      target.appendChild(notice(`Ran on: ${result.devices.join(", ")}`));
    }
    return;
  }

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

  const stressButton = document.getElementById("runStressTest");
  const stopButton = document.getElementById("stopSelfTest");
  const csvInput = document.getElementById("selfTestCsv");

  // The same description of a CSV the send-work form gives, from the same
  // place. Only the csv entry: this measures a graphics card, and a text file
  // trains a different kind of model that would tell a contributor nothing
  // about theirs.
  const kinds = document.getElementById("selfTestDataKinds");
  if (kinds) kinds.replaceChildren(dataKindsList(["csv"]));

  // The way down from the node details panel, where somebody has just found
  // out their card is connected and wants to know whether it works.
  const jumpHere = document.getElementById("testMachineButton");
  const panel = document.getElementById("selfTestPanel");
  if (jumpHere && panel) {
    jumpHere.addEventListener("click", () => {
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
      // Focus lands on the first test rather than the panel, so a keyboard is
      // one press from starting it and a screen reader is told what arrived.
      button.focus({ preventScroll: true });
    });
  }

  loadExisting();

  if (csvInput) {
    csvInput.addEventListener("change", () => {
      const file = csvInput.files?.[0];
      setCsvStatus(
        file ? `Will train on ${file.name} (${Math.round(file.size / 1024)} kB).` : "",
      );
    });
  }

  function setRunning(running, mode) {
    button.disabled = running;
    if (stressButton) stressButton.disabled = running;
    if (stopButton) stopButton.hidden = !running;

    button.textContent = running && mode === "quick" ? "Testing…" : "Quick test";
    if (stressButton) {
      stressButton.textContent = running && mode === "stress"
        ? "Working the card…" : "Stress test · 5 min";
    }
  }

  if (stopButton) {
    stopButton.addEventListener("click", async () => {
      stopButton.disabled = true;
      stopButton.textContent = "Stopping…";
      try {
        await fetch("/self-test/stop", { method: "POST" });
      } catch (error) {
        console.error("Could not stop the test:", error);
      } finally {
        // The run itself reports back; this button only asks.
        stopButton.textContent = "Stop";
        stopButton.disabled = false;
      }
    });
  }

  async function run(mode) {
    setRunning(true, mode);
    render({ running: true, mode });

    const file = csvInput?.files?.[0];

    try {
      const res = await fetch(`/self-test?mode=${encodeURIComponent(mode)}`, {
        method: "POST",
        headers: file && mode === "quick" ? { "Content-Type": "text/csv" } : {},
        body: file && mode === "quick" ? await file.text() : undefined,
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
      setRunning(false, mode);
    }
  }

  // The node presents either run as its current task, so the live view picks
  // it up on its next poll -- the same overlay a real job opens.
  button.addEventListener("click", () => run("quick"));
  if (stressButton) stressButton.addEventListener("click", () => run("stress"));
}
