// Detect the GPU this browser is running on, for the connect screen.
//
// Chrome reports the renderer through ANGLE, wrapped in a structure the old
// code did not account for:
//
//   ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 (0x00002484) Direct3D11 vs_5_0 ps_5_0, D3D11)
//
// Stripping keywords alone left "NVIDIA, NVIDIA GeForce RTX 3070 ," -- the
// vendor duplicated and a dangling comma. The structure has to be unwrapped
// first, then the noise removed.

const NOISE = /\b(Direct3D\d*|D3D\d*|vs_\d+_\d+|ps_\d+_\d+|OpenGL(\s+ES)?[\d.]*|Vulkan|Metal)\b/gi;

/**
 * Reduce a raw WebGL renderer string to a human-readable card name.
 * Exported separately from the WebGL call so it can be tested on its own.
 */
export function cleanRendererName(raw) {
  if (!raw || typeof raw !== "string") return "";

  let name = raw.trim();

  // ANGLE (vendor, renderer, backend) -- the middle field is the card.
  const angle = name.match(/^ANGLE\s*\((.*)\)\s*$/i);
  if (angle) {
    const parts = angle[1].split(",").map(p => p.trim()).filter(Boolean);
    name = parts.length >= 2 ? parts[1] : (parts[0] || name);
  }

  return name
    .replace(/\(0x[0-9a-f]+\)/gi, "")   // PCI device id
    .replace(NOISE, "")
    .replace(/[(),]+/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
}

/**
 * @returns {{supported: boolean, name: string, raw: string|null}}
 */
/**
 * Draw the result of a GPU check into `box`.
 *
 * Lived inline on the connect page. Now that registering happens on the front
 * door, it runs when somebody actually asks to register -- a visitor with data
 * and a MacBook must never be shown a hardware check for a thing they are not
 * doing.
 */
export function renderGpuStatus(box) {
  if (!box) return { supported: false, name: "" };

  const result = detectGpu();
  const { supported, name } = result;

  box.classList.remove("is-checking");
  box.classList.toggle("is-ok", supported);
  box.classList.toggle("is-bad", !supported);

  const title = supported ? "GPU ready" : "No supported GPU found";
  const detail = supported
    ? name
    : (name
        ? `${name} is not an NVIDIA card. The node only accepts NVIDIA GPUs.`
        : "Your browser did not report a GPU. Check your drivers.");

  // Built as nodes rather than markup: the renderer string comes from the
  // driver, and it is never worth trusting a driver string to innerHTML.
  box.replaceChildren();

  const dot = document.createElement("span");
  dot.className = "gpu-status-dot";
  dot.setAttribute("aria-hidden", "true");

  const text = document.createElement("span");
  text.className = "gpu-status-text";

  const strong = document.createElement("strong");
  strong.className = "gpu-status-title";
  strong.textContent = title;

  const small = document.createElement("span");
  small.className = "gpu-status-detail";
  small.textContent = detail;

  text.append(strong, small);
  box.append(dot, text);

  return result;
}


export function detectGpu() {
  let raw = null;

  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
    const debugInfo = gl && gl.getExtension("WEBGL_debug_renderer_info");
    if (debugInfo) {
      raw = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
    }
  } catch (err) {
    console.warn("GPU detection failed:", err);
  }

  const name = cleanRendererName(raw);

  return {
    // The node only accepts NVIDIA hardware, so that is what we look for.
    supported: /nvidia|geforce|quadro|tesla|rtx|gtx/i.test(name),
    name,
    raw,
  };
}
