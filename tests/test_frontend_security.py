"""The browser-facing defences, checked as source rather than as a running site.

Three separate mistakes are guarded here, each of which was real.

1. Error text went to innerHTML. `showMessage(err.message)` in the connect
   modals wrote straight into innerHTML, and those errors are built from the
   server's `detail` field -- which in several places is an f-string holding a
   node_id taken from the URL. Markup from outside, parsed in a page that keeps
   a private key in localStorage.

2. The page bootstraps were inline. A Content-Security-Policy cannot say
   `script-src 'self'` while a page carries an inline <script>; it would need
   'unsafe-inline', which permits injected scripts too. They live in
   /static/js/page/ now so the policy can be stated without an exception.

3. The node agent answered anybody. It runs on a contributor's own machine and
   returned `Access-Control-Allow-Origin: *` while /approve-task, /decline-task
   and /approval-mode asked for no credentials at all, so any website the
   contributor visited could read what their GPU was doing and then approve
   work on it.
"""

import os
import re

import pytest

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")
JS = os.path.join(SRC, "frontend", "static", "js")
TEMPLATES = os.path.join(SRC, "frontend", "template")


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def js_files():
    for folder, _dirs, files in os.walk(JS):
        for name in sorted(files):
            if name.endswith(".js"):
                yield os.path.join(folder, name)


# --- 1. nothing untrusted reaches innerHTML ------------------------------

# header.js loads our own template from a fixed same-origin path, which is the
# one place parsing markup is the point rather than a mistake.
INNERHTML_ALLOWED = {"header.js"}


def test_no_module_assigns_to_innerhtml():
    offenders = []
    for path in js_files():
        if os.path.basename(path) in INNERHTML_ALLOWED:
            continue
        for i, line in enumerate(read(path).split("\n"), 1):
            code = line.split("//")[0]
            if re.search(r"\.(inner|outer)HTML\s*=", code):
                offenders.append(f"{os.path.relpath(path, JS)}:{i}")

    assert not offenders, (
        "these write to innerHTML; use textContent or createElement: "
        + ", ".join(offenders)
    )


def test_the_connect_modals_show_errors_as_text():
    """The pair that carried the server's `detail` into the DOM.

    Some of those strings are formatted with a node_id taken straight out of the
    URL, so this is markup arriving from outside and being parsed -- the one
    thing these files must not do.

    Both used to set textContent themselves. They now hand the message to
    connect/nodeErrors.js, which also appends a link to the setup guide where
    the failure was that no node agent is running. The check follows the render
    to where it lives rather than pinning it to two files that no longer do it.
    """
    renderer = read(os.path.join(JS, "connect/nodeErrors.js"))
    assert "textContent = message" in renderer, (
        "nodeErrors.js is where the message reaches the page; it has to set text"
    )

    # The link it adds is built, not interpolated, and points at one fixed page.
    assert 'href = "/setup"' in renderer, (
        "the guide link should be a literal path, never built from a response"
    )

    for name in ("connect/connectExistingNode.js", "connect/registerNodeModal.js"):
        source = read(os.path.join(JS, name))
        assert "showNodeMessage" in source, (
            f"{name} should render through nodeErrors.js, which sets text"
        )
        # innerHTML is caught for every file by the check above; this says the
        # two that handle `detail` did not start assembling markup instead.
        assert "insertAdjacentHTML" not in source, (
            f"{name} builds markup from a response"
        )


# --- 2. the CSP can stay strict ------------------------------------------

def test_no_page_carries_an_inline_script():
    """An inline block would force 'unsafe-inline' back into the policy."""
    offenders = []
    for name in sorted(os.listdir(TEMPLATES)):
        if not name.endswith(".html"):
            continue
        html = read(os.path.join(TEMPLATES, name))
        for match in re.finditer(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>",
                                 html, re.S):
            if match.group(1).strip():
                offenders.append(name)

    assert not offenders, (
        "inline <script> in " + ", ".join(offenders)
        + " -- move it to static/js/page/ so script-src can stay 'self'"
    )


@pytest.mark.parametrize("directive", [
    "default-src 'self'",
    "script-src 'self'",
    "frame-ancestors 'none'",
    "object-src 'none'",
    "base-uri 'self'",
])
def test_the_policy_states(directive):
    source = read(os.path.join(SRC, "app.py"))
    assert directive in source


def test_the_policy_never_allows_inline_or_eval():
    """Read the policy itself, not the file.

    The file mentions 'unsafe-inline' in the comment explaining why it is not
    used, so searching the whole source finds the word and proves nothing.
    """
    source = read(os.path.join(SRC, "app.py"))
    body = source[source.index("CSP = "):]
    policy = body[:body.index("])") + 2]

    assert "unsafe-inline" not in policy
    assert "unsafe-eval" not in policy
    assert "script-src 'self'" in policy


@pytest.mark.parametrize("header", [
    "Content-Security-Policy",
    "X-Frame-Options",
    "X-Content-Type-Options",
    "Referrer-Policy",
])
def test_the_dashboard_sets(header):
    assert header in read(os.path.join(SRC, "app.py"))


# --- 3. the node does not take orders from other websites ----------------

def test_the_node_does_not_allow_every_origin():
    source = read(os.path.join(SRC, "backend", "node.py"))

    assert 'allow_origins=["*"]' not in source, (
        "the node agent listens on a contributor's machine; a wildcard there "
        "lets any site they visit read what their GPU is doing"
    )
    assert "ALLOWED_ORIGINS" in source


def test_the_node_refuses_requests_from_an_unknown_origin():
    """CORS stops the reply being read, not the request being sent.

    /approve-task is the whole attack whether or not the answer comes back, so
    an unknown Origin has to be turned away before it reaches a route.
    """
    source = read(os.path.join(SRC, "backend", "node.py"))

    assert "refuse_unknown_origins" in source
    assert "origin not in ALLOWED_ORIGINS" in source
