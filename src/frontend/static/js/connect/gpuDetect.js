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
