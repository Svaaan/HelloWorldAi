// Which side of the network a page belongs to, and what each visitor sees.
//
// Run with: node tests/test_role_pages.js
//
// The rules live in the browser, so this drives the modules directly with a
// stand-in for localStorage rather than through a page. What it is checking is
// the thing that was wrong: the navigation used to describe where you were
// standing instead of who you are, so opening the other side's page grew you
// half of that side's links -- and a first-time visitor who followed a link to
// an interior page was shown both sides at once.

import assert from "node:assert/strict";

// --- a browser, more or less ---------------------------------------------

const store = new Map();
globalThis.localStorage = {
  getItem: (k) => (store.has(k) ? store.get(k) : null),
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
  clear: () => store.clear(),
};

function signedInAs(...roles) {
  store.clear();
  if (roles.includes("contributor")) localStorage.setItem("currentNodeId", "node_x");
  if (roles.includes("builder")) localStorage.setItem("submitterKey", "k".repeat(64));
}

const role = await import("../src/frontend/static/js/component/role.js");

let passed = 0;
function check(name, fn) {
  try {
    fn();
    passed += 1;
  } catch (error) {
    console.error(`  FAILED: ${name}\n    ${error.message}`);
    process.exitCode = 1;
  }
}

// --- who the browser is --------------------------------------------------

check("a browser with a node key is a contributor", () => {
  signedInAs("contributor");
  assert.equal(role.isContributor(), true);
  assert.equal(role.isBuilder(), false);
  assert.equal(role.currentRole(), "contributor");
});

check("a browser with a submitter key is a builder", () => {
  signedInAs("builder");
  assert.equal(role.isBuilder(), true);
  assert.equal(role.currentRole(), "builder");
});

check("holding both keys is allowed and belongs to neither side alone", () => {
  // Somebody can genuinely lend a card and train their own models. The point
  // of currentRole() is "which side to send you back to", and for this person
  // there is no such single side.
  signedInAs("contributor", "builder");
  assert.equal(role.isContributor(), true);
  assert.equal(role.isBuilder(), true);
  assert.equal(role.currentRole(), null);
  assert.equal(role.isNewHere(), false);
});

check("a browser with no keys is new here", () => {
  signedInAs();
  assert.equal(role.isNewHere(), true);
  assert.equal(role.currentRole(), null);
});

// --- what the navigation offers ------------------------------------------

check("each side is offered only its own links", () => {
  signedInAs("contributor");
  assert.equal(role.showsFor("contributor"), true);
  assert.equal(role.showsFor("builder"), false);

  signedInAs("builder");
  assert.equal(role.showsFor("contributor"), false);
  assert.equal(role.showsFor("builder"), true);
});

check("a newcomer is offered neither side rather than both", () => {
  // This was the bug: showsFor used to return true for both roles when
  // nothing was set up, so following a link to /node showed a first-time
  // visitor five destinations, none of which were theirs.
  signedInAs();
  assert.equal(role.showsFor("contributor"), false);
  assert.equal(role.showsFor("builder"), false);
});

check("somebody who is both is offered both", () => {
  signedInAs("contributor", "builder");
  assert.equal(role.showsFor("contributor"), true);
  assert.equal(role.showsFor("builder"), true);
});

check("untagged links are always shown", () => {
  signedInAs();
  assert.equal(role.showsFor(undefined), true);
  assert.equal(role.showsFor("any"), true);
});

// --- which page is whose -------------------------------------------------

check("every page is attributed to the side that owns it", () => {
  assert.equal(role.pageRole("/node").role, "contributor");
  assert.equal(role.pageRole("/setup").role, "contributor");
  assert.equal(role.pageRole("/workspace").role, "builder");
  assert.equal(role.pageRole("/distribution").role, "builder");
});

check("the front door belongs to neither side", () => {
  // It is the page that asks which side you are on, so a notice explaining
  // that you are on the wrong one would be nonsense there.
  assert.equal(role.pageRole("/"), null);
  assert.equal(role.pageRole("/nonsense"), null);
});

check("a trailing slash is the same page", () => {
  assert.equal(role.pageRole("/node/").role, "contributor");
  assert.equal(role.pageRole("/workspace/").role, "builder");
});

check("entry pages are marked apart from interior ones", () => {
  // Arriving at an entry page without the role is the point of the page.
  // Arriving at an interior one means an empty room.
  assert.equal(role.pageRole("/setup").entry, true);
  assert.equal(role.pageRole("/distribution").entry, true);
  assert.equal(role.pageRole("/node").entry, undefined);
  assert.equal(role.pageRole("/workspace").entry, undefined);
});

check("every side has somewhere to send people back to", () => {
  for (const side of ["contributor", "builder"]) {
    assert.ok(role.ROLE_HOME[side], `no home for ${side}`);
    assert.ok(role.ROLE_ENTRY[side], `no entry for ${side}`);
    assert.ok(role.ROLE_LABEL[side], `no label for ${side}`);
    // Home is always that side's own page.
    assert.equal(role.pageRole(role.ROLE_HOME[side]).role, side);

    // The way in is either that side's own page, or the front door -- which
    // belongs to neither side and is where both are taken up.
    const entry = role.pageRole(role.ROLE_ENTRY[side]);
    assert.ok(entry === null || entry.role === side, `bad entry for ${side}`);
  }
});

// --- and the question the notice asks ------------------------------------

check("landing on your own page explains nothing", () => {
  signedInAs("contributor");
  assert.equal(role.hasRole(role.pageRole("/node").role), true);

  signedInAs("builder");
  assert.equal(role.hasRole(role.pageRole("/workspace").role), true);
});

check("landing on the other side's page needs explaining", () => {
  signedInAs("contributor");
  assert.equal(role.hasRole(role.pageRole("/workspace").role), false);

  signedInAs("builder");
  assert.equal(role.hasRole(role.pageRole("/node").role), false);
});

check("somebody who is both is never told they are in the wrong place", () => {
  signedInAs("contributor", "builder");
  for (const path of ["/node", "/workspace", "/distribution", "/setup"]) {
    assert.equal(role.hasRole(role.pageRole(path).role), true, path);
  }
  // And the front door belongs to nobody, so there is nothing to be wrong.
  assert.equal(role.pageRole("/"), null);
});

check("storage being refused reads as no role, not as an error", () => {
  // Private windows can refuse localStorage outright. That means nothing has
  // been set up here, which is exactly what a newcomer is.
  const real = globalThis.localStorage;
  const warn = console.warn;
  globalThis.localStorage = {
    getItem() { throw new Error("denied"); },
  };
  // The module warns, correctly. Here the throw is the point, so the warning
  // is noise that would bury a genuine failure.
  console.warn = () => {};
  try {
    assert.equal(role.isContributor(), false);
    assert.equal(role.isBuilder(), false);
    assert.equal(role.isNewHere(), true);
  } finally {
    globalThis.localStorage = real;
    console.warn = warn;
  }
});

// --- when the notice is worth showing -----------------------------------
//
// The same rule showRoleNotice applies, kept here so the matrix is written
// down: an entry page introduces itself, so a first-time visitor does not
// need a banner above "Contribute your GPU" saying this page is for lending
// a GPU. Somebody signed in to the other side does need one.

function noticeShown(path) {
  const page = role.pageRole(path);
  if (!page) return false;
  if (role.hasRole(page.role)) return false;
  if (page.entry && role.isNewHere()) return false;
  return true;
}

check("an entry page introduces itself to a newcomer", () => {
  signedInAs();
  assert.equal(noticeShown("/setup"), false);
  assert.equal(noticeShown("/distribution"), false);
});

check("an interior page cannot introduce itself, so it is explained", () => {
  // /node and /workspace are empty without the key. "No node connected yet"
  // reads as a fault rather than as somebody else's half of the product.
  signedInAs();
  assert.equal(noticeShown("/node"), true);
  assert.equal(noticeShown("/workspace"), true);
});

check("the other side is always told where it has landed", () => {
  signedInAs("builder");
  for (const path of ["/setup", "/node"]) {
    assert.equal(noticeShown(path), true, path);
  }
  for (const path of ["/distribution", "/workspace"]) {
    assert.equal(noticeShown(path), false, path);
  }

  signedInAs("contributor");
  for (const path of ["/distribution", "/workspace"]) {
    assert.equal(noticeShown(path), true, path);
  }
  for (const path of ["/setup", "/node"]) {
    assert.equal(noticeShown(path), false, path);
  }
});

check("somebody who is both is never interrupted anywhere", () => {
  signedInAs("contributor", "builder");
  for (const path of ["/setup", "/node", "/distribution", "/workspace"]) {
    assert.equal(noticeShown(path), false, path);
  }
});

check("the front door never carries a notice", () => {
  for (const who of [[], ["contributor"], ["builder"], ["contributor", "builder"]]) {
    signedInAs(...who);
    assert.equal(noticeShown("/"), false);
  }
});

// --- and which links belong on the page you are standing on -------------
//
// A link is yours if you hold the key for its side, and it belongs here if it
// is for the same side as the page. Standing on /connect -- which asks you to
// register a graphics card -- while the header offers "Send work" and "Your
// workspace" puts both profiles in front of somebody at the moment they are
// being asked about one of them.

const NAV = [
  ["/node", "contributor", "Your node"],
  ["/setup", "contributor", "Setup"],
  ["/distribution", "builder", "Send work"],
  ["/workspace", "builder", "Your workspace"],
];

/** Exactly what applyRoles() decides, for one page. */
function navOn(path) {
  if (role.isNewHere()) return [];

  // The front door carries no navigation for anybody. Its two cards say
  // "Open your workspace" and "Connect to node" in full sentences; a row of
  // links above them saying the same thing smaller is the same choice offered
  // twice, on the one page whose whole job is asking it once.
  if ((path.replace(/\/+$/, "") || "/") === "/") return [];

  const here = role.pageRole(path);
  return NAV.filter(([, linkRole]) => {
    if (!role.showsFor(linkRole)) return false;
    if (here && linkRole !== here.role) return false;
    return true;
  }).map(([, , label]) => label);
}

check("a data person on the GPU side is offered nothing else", () => {
  // The reported problem: being shown the other profile's navigation while
  // standing on a page that belongs to this one.
  signedInAs("builder");
  assert.deepEqual(navOn("/setup"), []);
  assert.deepEqual(navOn("/node"), []);
});

check("a GPU owner is not offered the data side while on it", () => {
  signedInAs("contributor");
  assert.deepEqual(navOn("/distribution"), []);
  assert.deepEqual(navOn("/workspace"), []);
});

check("each side keeps its own links on its own pages", () => {
  signedInAs("contributor");
  assert.deepEqual(navOn("/node"), ["Your node", "Setup"]);

  signedInAs("builder");
  assert.deepEqual(navOn("/workspace"), ["Send work", "Your workspace"]);
});

check("the front door carries no navigation for anybody", () => {
  // Changing sides happens through the two cards, which are the page. The
  // header there is the logo and the name.
  for (const who of [[], ["contributor"], ["builder"], ["contributor", "builder"]]) {
    signedInAs(...who);
    assert.deepEqual(navOn("/"), [], who.join("+") || "nobody");
  }
});

check("somebody who is both still sees one side at a time on a page", () => {
  signedInAs("contributor", "builder");
  assert.deepEqual(navOn("/node"), ["Your node", "Setup"]);
  assert.deepEqual(navOn("/workspace"), ["Send work", "Your workspace"]);
});

check("registering happens where there is nothing else to click", () => {
  // Both halves of the front door's quietness, together: no links, and no
  // count of machines online, which means nothing to somebody who has not
  // chosen a side.
  signedInAs("builder");
  assert.deepEqual(navOn("/"), []);
  assert.equal(role.ROLE_ENTRY.contributor, "/");
});

check("a newcomer is offered nothing anywhere", () => {
  signedInAs();
  for (const path of ["/", "/setup", "/node", "/distribution", "/workspace"]) {
    assert.deepEqual(navOn(path), [], path);
  }
});

check("registering happens on the front door, not on a page of its own", () => {
  // /connect was a second screen doing the job the front door already did for
  // the other side. It redirects now, so it belongs to no side and appears in
  // no navigation.
  assert.equal(role.pageRole("/connect"), null);
  assert.ok(!NAV.some(([, , label]) => label === "Connect"));
  assert.equal(role.ROLE_ENTRY.contributor, "/");
});

console.log(`  ${passed} checks passed`);
