"""Everything a node agent calls has to be reachable from one address.

A contributor installs the node with the command the setup page generates, and
that command points at the dashboard -- because one public address is one thing
to expose and one certificate to keep, and the production compose binds the
coordinator to localhost so its own port does not exist from outside.

The dashboard forwarded some of what the agent calls and not the rest. The
heartbeat was forwarded, so a node installed that way registered, reported in,
and showed as Connected with the right graphics card. `next-task` was not, so
it never received a single job. The one endpoint that decides whether a machine
looks healthy was the one that worked, which is the worst shape a failure can
take.

This reads both sides and compares them, so the next endpoint added to the
agent cannot quietly go unproxied.

What this file no longer checks: whether a route forwards the caller's
credentials, the query string or the body. That was done by searching the
source for `auth_headers(request)` near each handler, which only worked while
every route was written out by hand. The proxy is now one forwarder fed by a
table, and those properties are checked in test_proxy_forwarding.py by driving
the real router against a fake upstream -- behaviour, rather than the shape of
the source.
"""

import os
import re
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
os.environ.setdefault("ENV", "test")

NODE = os.path.join(HERE, "..", "src", "backend", "node.py")
COORDINATOR = os.path.join(HERE, "..", "src", "backend", "coordinator.py")

import backend.proxypage as proxypage        # noqa: E402


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def endpoints_the_agent_calls():
    """Every coordinator path built from COORDINATOR_URL in the node agent."""
    source = read(NODE)
    # f"{COORDINATOR_URL}/next-task/{node_id}" -> "next-task"
    return {
        match.group(1)
        for match in re.finditer(r"COORDINATOR_URL\}/([a-z][a-z0-9-]*)", source)
    }


def endpoints_the_dashboard_forwards():
    """Every path the dashboard proxy exposes, taken from the live router.

    Asking the router rather than reading the file means this keeps working
    however the routes come to be registered.
    """
    found = set()
    for route in proxypage.router.routes:
        first = getattr(route, "path", "").strip("/").split("/")[0]
        if first and not first.startswith("{"):
            found.add(first)
    return found


def test_the_agent_calls_something():
    # A guard on the reader itself: a regex that silently matches nothing
    # would make every assertion below pass for the wrong reason.
    called = endpoints_the_agent_calls()

    assert len(called) >= 4, f"only found {called}; the reader is probably broken"
    assert "next-task" in called
    assert "node-heartbeat" in called


def test_the_dashboard_forwards_something():
    forwarded = endpoints_the_dashboard_forwards()

    assert len(forwarded) >= 10, f"only found {forwarded}; the reader is broken"
    assert "node-heartbeat" in forwarded


def test_every_endpoint_the_agent_calls_is_forwarded():
    """The one that was false.

    next-task, task-result and task-cancelled were missing, so a node
    installed from the setup page's command could register and never work.
    """
    called = endpoints_the_agent_calls()
    forwarded = endpoints_the_dashboard_forwards()

    missing = sorted(called - forwarded)

    assert not missing, (
        "the node agent calls these, and the dashboard does not forward them: "
        f"{', '.join(missing)}. A node installed against the dashboard origin "
        f"would register and then silently never work."
    )


@pytest.mark.parametrize("endpoint", [
    "next-task",        # claiming a job
    "task-result",      # returning the trained weights
    "task-cancelled",   # noticing the submitter stopped it
    "node-heartbeat",   # staying online
    "artifacts",        # fetching the dataset
])
def test_each_one_by_name(endpoint):
    # Named individually so a failure says which capability is gone rather
    # than only that a set difference is non-empty.
    assert endpoint in endpoints_the_dashboard_forwards()


def test_nothing_is_forwarded_to_a_route_the_node_does_not_serve():
    """/execute-task was proxied to a path node.py has never had.

    A forward to a route that does not exist is a guaranteed 404 which, from
    the page, is indistinguishable from the node being offline.
    """
    served = {
        m.group(2).split("{")[0].rstrip("/")
        for m in re.finditer(r'@app\.(get|post|patch|put|delete)\("([^"]+)"',
                             read(NODE))
    }

    proxied = {path.split("{")[0].rstrip("/")
               for _methods, path, _timeout in proxypage.NODE_ROUTES}

    missing = sorted(proxied - served)

    assert not missing, (
        "the proxy forwards these to the node, which does not serve them: "
        f"{', '.join(missing)}"
    )


# --- storing a blob needs a caller ---------------------------------------

def test_uploading_an_artifact_requires_a_caller():
    """Anyone who could reach the coordinator could fill its database.

    POST /artifacts took no credentials at all. MAX_ARTIFACT_BYTES is 512 MB
    and nothing limited how many times it could be sent, so a stranger could
    write to GridFS until the disk gave out, with no caller recorded to
    attribute, count or clean up against.

    Both real callers already prove who they are: the node sends the bearer
    token from /verify-challenge, the browser sends the submitter key it makes
    on first use.
    """
    # The routes moved into backend/routes/ when coordinator.py was split; the
    # decorator is what is being read, so this has to follow them.
    source = read(os.path.join(HERE, "..", "src", "backend", "routes", "artifacts.py"))

    for route in ('@router.post("/artifacts")',
                  '@router.post("/artifacts/{artifact_id}/append")'):
        start = source.index(route)
        window = source[start:start + 400]
        assert "require_uploader" in window, (
            f"{route} does not ask who is uploading"
        )
