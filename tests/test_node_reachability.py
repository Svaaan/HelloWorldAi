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
"""

import os
import re
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

NODE = os.path.join(HERE, "..", "src", "backend", "node.py")
PROXY = os.path.join(HERE, "..", "src", "backend", "proxypage.py")


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
    """Every path the dashboard proxy exposes."""
    source = read(PROXY)
    return {
        match.group(1)
        for match in re.finditer(r'@router\.\w+\("/([a-z][a-z0-9-]*)', source)
    }


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


def test_the_forwarded_routes_carry_the_agents_credentials():
    """A node proves itself with a bearer token on every call.

    Forwarding the path but not the Authorization header would turn a working
    node into one the coordinator refuses, which is a different silent failure
    with the same shape.
    """
    source = read(PROXY)

    for endpoint in ("next-task", "task-result", "task-cancelled"):
        start = source.index(f"/{endpoint}/")
        window = source[start:start + 900]
        assert "auth_headers(request)" in window, (
            f"the {endpoint} proxy does not forward the node's token"
        )


def test_the_result_route_forwards_the_body():
    """Weights, metrics and logs travel in the body.

    Forwarding headers but not the body is a mistake this proxy has made
    before -- on /self-test, on /available-nodes and on /retry-task -- and it
    always looks like success while doing nothing.
    """
    source = read(PROXY)
    start = source.index("/task-result/")
    window = source[start:start + 900]

    assert "await request.body()" in window
    assert "content=body" in window


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
    source = read(os.path.join(HERE, "..", "src", "backend", "coordinator.py"))

    for route in ('@app.post("/artifacts")', '@app.post("/artifacts/{artifact_id}/append")'):
        start = source.index(route)
        window = source[start:start + 400]
        assert "require_uploader" in window, (
            f"{route} does not ask who is uploading"
        )


def test_the_upload_proxy_forwards_the_callers_credentials():
    """Gating the coordinator is no good if the proxy strips the proof.

    The dashboard forwarded the body and set its own Content-Type, replacing
    the header dict rather than adding to it, so the submitter key never
    arrived. It surfaces as a 401 from the coordinator and reads, to the person
    uploading, as their own key being rejected.
    """
    source = read(PROXY)

    for route in ('@router.post("/artifacts")',
                  '@router.post("/artifacts/{artifact_id}/append")'):
        start = source.index(route)
        window = source[start:start + 800]
        assert "auth_headers(request" in window, (
            f"the {route} proxy sends no credentials, so every upload 401s"
        )
