// Setup guide behaviour: the install command, and copy buttons.
//
// The page is a list of shell commands, several of them multi-line. Selecting
// those by hand in a browser is fiddly and easy to get wrong, so every command
// box gets a copy button. The old page had a copyToClipboard() helper that was
// never wired to anything, and announced itself with alert().

const COPY_RESET_MS = 1600;

export function initSetup() {
  buildInstallCommand();
  addCopyButtons();
}

// The one-liner in the fast path has to carry the address of the coordinator
// this page is served from, so it is built here rather than hard-coded.
function buildInstallCommand() {
  const input = document.getElementById("coordinatorUrl");
  const output = document.getElementById("setupCommand");
  if (!input || !output) return;

  const origin = window.location.origin;
  input.value = origin;

  const render = () => {
    const coordinator = input.value.trim() || origin;
    // curl needs an absolute URL, so the script is always fetched from the
    // origin serving this page even when the node reports elsewhere.
    output.textContent =
      `bash <(curl -fsSL ${origin}/static/scripts/setup-node.sh)` +
      ` --coordinator ${coordinator}`;
  };

  input.addEventListener("input", render);
  render();
}

function addCopyButtons() {
  document.querySelectorAll(".command-box").forEach((box) => {
    const code = box.querySelector("code");
    if (!code) return;

    // The button lives on a wrapper, not on the box itself: .command-box
    // scrolls horizontally, and anything positioned inside it drifts across
    // the code as the user scrolls a long command.
    const wrap = document.createElement("div");
    wrap.className = "command-wrap";
    box.parentNode.insertBefore(wrap, box);
    wrap.appendChild(box);

    const button = document.createElement("button");
    button.type = "button";
    button.className = "copy-btn";
    button.textContent = "Copy";
    button.setAttribute("aria-label", "Copy command to clipboard");

    button.addEventListener("click", async () => {
      const ok = await writeClipboard(code.textContent);

      button.textContent = ok ? "Copied" : "Press Ctrl+C";
      button.classList.toggle("is-copied", ok);

      if (!ok) selectText(code);

      window.setTimeout(() => {
        button.textContent = "Copy";
        button.classList.remove("is-copied");
      }, COPY_RESET_MS);
    });

    wrap.appendChild(button);
  });
}

async function writeClipboard(text) {
  // navigator.clipboard needs a secure context. That covers localhost and
  // https, but a node reached over plain http on a LAN address gets neither,
  // so fall back rather than throwing an unhandled rejection.
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (error) {
    console.warn("Clipboard write refused:", error);
  }

  try {
    const scratch = document.createElement("textarea");
    scratch.value = text;
    scratch.setAttribute("readonly", "");
    scratch.style.position = "fixed";
    scratch.style.opacity = "0";
    document.body.appendChild(scratch);
    scratch.select();
    const ok = document.execCommand("copy");
    scratch.remove();
    return ok;
  } catch (error) {
    console.warn("Clipboard fallback failed:", error);
    return false;
  }
}

// Last resort: at least select it so Ctrl+C works.
function selectText(node) {
  const range = document.createRange();
  range.selectNodeContents(node);
  const selection = window.getSelection();
  selection.removeAllRanges();
  selection.addRange(range);
}
