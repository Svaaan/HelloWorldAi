"""What the dashboard proxy actually sends upstream.

Every request the browser makes goes through this layer, and it has been the
most reliable source of bugs in the project. The pattern is always the same: a
route forwards some of what it was given and silently drops the rest.

    next-task, task-result, task-cancelled   not forwarded at all -- a node
                                             registered, showed Connected, and
                                             never received a single job
    artifacts                                replaced the header dict, so the
                                             submitter key never arrived
    execute-task                             forwarded to a route the node does
                                             not serve
    self-test, available-nodes, retry-task   headers forwarded, body dropped

None of those fail loudly. They look like a working page that does nothing, or
like the person's own key being rejected.

The older guard reads the source as text, which catches a route that is missing
but not one that forwards the wrong things. These drive the real router against
a fake upstream and record what came out of it.
"""

import json
import os
import re
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("ENV", "test")

from fastapi import FastAPI                                   # noqa: E402
from fastapi.testclient import TestClient                     # noqa: E402

import backend.proxypage as proxypage                          # noqa: E402
from backend.utils.config import COORDINATOR_URL, NODE_URL     # noqa: E402

COORD = "COORDINATOR"
NODE = "NODE"


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self.content = payload
        self.text = payload.decode()
        self.headers = {"content-type": "application/json"}

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx
            raise httpx.HTTPStatusError("upstream said no", request=None,
                                        response=self)


class Recorder:
    """Stands in for httpx.AsyncClient and remembers what was sent."""

    def __init__(self, sink, status=200, payload=b'{"ok": true}'):
        self.sink = sink
        self.status = status
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    def _record(self, method, url, **kw):
        headers = {k.lower(): v for k, v in (kw.get("headers") or {}).items()}
        body = kw.get("content")
        if body is None and kw.get("json") is not None:
            body = json.dumps(kw["json"]).encode()
        if isinstance(body, str):
            body = body.encode()
        self.sink.append({
            "method": method.upper(),
            "url": str(url),
            "params": dict(kw.get("params") or {}),
            "headers": headers,
            "body": body,
        })
        return FakeResponse(self.status, self.payload)

    async def request(self, method, url, **kw):
        return self._record(method, url, **kw)

    async def get(self, url, **kw):
        return self._record("GET", url, **kw)

    async def post(self, url, **kw):
        return self._record("POST", url, **kw)

    async def patch(self, url, **kw):
        return self._record("PATCH", url, **kw)

    async def put(self, url, **kw):
        return self._record("PUT", url, **kw)

    async def delete(self, url, **kw):
        return self._record("DELETE", url, **kw)


def call(method, path, params=None, body=None, headers=None, status=200):
    """Drive one proxy route; return (what it sent upstream, its response)."""
    sink = []
    app = FastAPI()
    app.include_router(proxypage.router)

    def fake_client(*_a, **_k):
        return Recorder(sink, status=status)

    with patch.object(proxypage.httpx, "AsyncClient", fake_client):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.request(method, path, params=params, content=body,
                                  headers=headers or {})

    return sink, response


def upstream_of(url):
    if url.startswith(COORDINATOR_URL):
        return COORD
    if url.startswith(NODE_URL):
        return NODE
    return "unknown(%s)" % url


# (method, path as the browser calls it, which service should receive it)
ROUTES = [
    ("GET", "/nodes", COORD),
    ("GET", "/available-nodes", COORD),
    ("GET", "/get-connected-nodes-count", COORD),
    ("GET", "/tasks", COORD),
    ("GET", "/my-tasks", COORD),
    ("GET", "/job-schema", COORD),
    ("GET", "/get-task-results", COORD),
    ("POST", "/receive-task-result", COORD),
    ("PATCH", "/toggle-availability/node_1", COORD),
    ("GET", "/next-task/node_1", COORD),
    ("POST", "/task-result/task_1", COORD),
    ("GET", "/task-cancelled/task_1", COORD),
    ("POST", "/node-heartbeat/node_1", COORD),
    ("DELETE", "/node/node_1", COORD),
    ("POST", "/verify-task/task_1", COORD),
    ("POST", "/submit-task", COORD),
    ("POST", "/submit-task/node_1", COORD),
    ("POST", "/cancel-task/task_1", COORD),
    ("POST", "/retry-task/task_1", COORD),
    ("POST", "/artifacts", COORD),
    ("POST", "/artifacts/art_1/append", COORD),
    ("GET", "/artifacts/art_1", COORD),
    ("GET", "/my-tasks/task_1/bundle", COORD),
    ("POST", "/my-tasks/task_1/predict", COORD),
    ("POST", "/my-tasks/task_1/sample", COORD),
    ("GET", "/generate-challenge/node_1", COORD),
    ("POST", "/verify-challenge/node_1", COORD),
    ("POST", "/find-node-id", COORD),
    ("POST", "/verify-node/node_1/cpu", COORD),
    ("POST", "/verify-node/node_1/gpu", COORD),
    ("POST", "/connect-node", NODE),
    ("POST", "/finalize-connection", NODE),
    ("POST", "/node-session", NODE),
    ("GET", "/current-task", NODE),
    ("GET", "/usage", NODE),
    ("POST", "/approve-task/task_1", NODE),
    ("POST", "/decline-task/task_1", NODE),
    ("POST", "/approval-mode", NODE),
    ("GET", "/self-test", NODE),
    ("POST", "/self-test", NODE),
    ("POST", "/self-test/stop", NODE),
]

IDS = ["%s %s" % (m, p) for m, p, _ in ROUTES]

# These answer for themselves rather than forwarding: /distribution renders a
# page, and /local-node reports whether a node agent runs beside this dashboard
# at all -- false on the central deployment, where there is no node and the
# front door should not offer to register one.
NOT_A_PROXY = {("GET", "/distribution"), ("GET", "/local-node")}


def shape(path):
    """Reduce a concrete path to its template form for comparison."""
    return re.sub(r"/(node_1|task_1|art_1)", "/{}", path)


def test_the_table_covers_every_route():
    """A guard on the table itself.

    A route added to the proxy without a line here would go unchecked, which is
    exactly how three of them came to be missing.
    """
    served = set()
    for route in proxypage.router.routes:
        for method in getattr(route, "methods", []):
            if method in ("HEAD", "OPTIONS"):
                continue
            served.add((method, re.sub(r"\{[^}]+\}", "{}", route.path)))

    listed = {(m, shape(p)) for m, p, _ in ROUTES}
    uncovered = served - listed - {(m, p) for m, p in NOT_A_PROXY}

    assert not uncovered, (
        "these proxy routes are not covered by this test: %s" % sorted(uncovered)
    )


@pytest.mark.parametrize("method,path,expected", ROUTES, ids=IDS)
def test_each_route_reaches_the_right_service(method, path, expected):
    body = b"{}" if method in ("POST", "PATCH", "PUT") else None
    sink, _response = call(method, path, body=body)

    assert len(sink) == 1, "%s %s made %d upstream calls" % (method, path, len(sink))

    sent = sink[0]
    assert upstream_of(sent["url"]) == expected
    assert sent["url"].endswith(path), (
        "%s %s went to %s" % (method, path, sent["url"])
    )


# --- the contract: everything travels ------------------------------------
#
# Measured on the forty-two hand-written routes this replaced: 21 dropped the
# caller's credentials, 34 dropped the query string, 8 dropped the body. The
# point of one forwarder is that these three checks can be stated once and hold
# for every route, including whichever is added next.

CREDENTIALS = {"X-Submitter-Key": "key-123", "Authorization": "Bearer tok-456"}


def with_everything(method, path):
    body = b'{"probe": 1}' if method in ("POST", "PATCH", "PUT") else None
    sink, _res = call(method, path, params={"probe": "yes"}, body=body,
                      headers=CREDENTIALS)
    assert sink, "%s %s made no upstream call" % (method, path)
    return sink[0], body


@pytest.mark.parametrize("method,path,_expected", ROUTES, ids=IDS)
def test_every_route_forwards_the_callers_credentials(method, path, _expected):
    """The bug that broke uploads the moment /artifacts required a key.

    An endpoint that does not check credentials today is not a reason to drop
    them, because the day it starts checking is the day the feature breaks with
    a 401 that looks like the person's own key being refused.
    """
    sent, _body = with_everything(method, path)

    assert sent["headers"].get("x-submitter-key") == "key-123"
    assert sent["headers"].get("authorization") == "Bearer tok-456"


@pytest.mark.parametrize("method,path,_expected", ROUTES, ids=IDS)
def test_every_route_forwards_the_query_string(method, path, _expected):
    sent, _body = with_everything(method, path)

    assert sent["params"].get("probe") == "yes"


@pytest.mark.parametrize(
    "method,path,_expected",
    [r for r in ROUTES if r[0] in ("POST", "PATCH", "PUT")],
    ids=[f"{m} {p}" for m, p, _ in ROUTES if m in ("POST", "PATCH", "PUT")])
def test_every_writing_route_forwards_the_body(method, path, _expected):
    """Weights, metrics, logs and job definitions all travel in the body.

    Forwarding the headers but not the body is a mistake this layer made on
    /self-test, /available-nodes and /retry-task, and it always looks like
    success while doing nothing.
    """
    sent, body = with_everything(method, path)

    assert sent["body"] == body


# --- the answer comes back intact ----------------------------------------

def test_an_upstream_failure_keeps_its_status():
    """A 404 has to arrive as a 404.

    The pages check `response.ok`. Turning every upstream failure into a 200
    carrying the word "error" is why some of them showed an empty list instead
    of saying something had gone wrong.
    """
    _sink, res = call("GET", "/my-tasks", status=404)

    assert res.status_code == 404


def test_a_download_keeps_its_content_type():
    """The model bundle is a zip and the prediction is a CSV."""
    sink = []
    app = FastAPI()
    app.include_router(proxypage.router)

    class Zip(Recorder):
        def _record(self, method, url, **kw):
            super()._record(method, url, **kw)
            res = FakeResponse(200, b"PK\x03\x04zip-bytes")
            res.headers = {"content-type": "application/zip",
                           "content-disposition": 'attachment; filename="m.zip"'}
            return res

    with patch.object(proxypage.httpx, "AsyncClient",
                      lambda *a, **k: Zip(sink)):
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/my-tasks/task_1/bundle")

    assert res.status_code == 200
    assert res.headers["content-type"] == "application/zip"
    assert "m.zip" in res.headers.get("content-disposition", "")
    assert res.content == b"PK\x03\x04zip-bytes"


def test_an_unreachable_upstream_says_so_rather_than_pretending():
    import httpx as _httpx

    app = FastAPI()
    app.include_router(proxypage.router)

    class Dead(Recorder):
        def _record(self, method, url, **kw):
            raise _httpx.ConnectError("no route to host")

    with patch.object(proxypage.httpx, "AsyncClient",
                      lambda *a, **k: Dead([])):
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/my-tasks")

    assert res.status_code == 502
    assert "error" in res.json()


# --- a dashboard with no node beside it ----------------------------------
#
# The same image serves two jobs. Next to a contributor's agent it is how they
# register and manage it; on the central server there is no node and there
# cannot be one, because a graphics card is offered from the machine it is in.
#
# The central deployment offered "Lend your graphics card" anyway, and the
# button called /connect-node, which the proxy forwarded to http://node:9100 --
# a name that does not resolve there. The person saw:
#
#   Could not reach http://node:9100: [Errno -3] Temporary failure in name
#   resolution
#
# which invites an hour of debugging DNS for a container that was never meant
# to exist.

def test_an_unreachable_node_is_explained_not_leaked():
    import httpx as _httpx

    app = FastAPI()
    app.include_router(proxypage.router)

    class NoNode(Recorder):
        def _record(self, method, url, **kw):
            raise _httpx.ConnectError(
                "[Errno -3] Temporary failure in name resolution")

    with patch.object(proxypage.httpx, "AsyncClient",
                      lambda *a, **k: NoNode([])):
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/current-task")

    # 503, not 502: the service is unavailable here by design, not broken.
    assert res.status_code == 503
    body = res.json()

    # The flag is the contract. The pages branch on this to decide whether to
    # offer the setup guide, and an earlier version of this test asserted on a
    # phrase in the sentence instead -- which made the wording load-bearing and
    # broke the moment anyone improved it.
    assert body.get("no_local_node") is True, (
        "the pages need a marker they can branch on without matching prose"
    )

    for key in ("error", "detail"):
        assert "name resolution" not in body[key], (
            "the DNS error should not reach the person reading this"
        )
        assert "node agent" in body[key], (
            "it should still say, in words, what is missing"
        )


def test_a_coordinator_failure_still_reads_as_a_failure():
    """The friendlier message must not swallow a real outage.

    And it must not carry `no_local_node` either: the pages would offer the
    setup guide for a coordinator that is simply down, telling somebody to
    install software they already have.
    """
    import httpx as _httpx

    app = FastAPI()
    app.include_router(proxypage.router)

    class Dead(Recorder):
        def _record(self, method, url, **kw):
            raise _httpx.ConnectError("connection refused")

    with patch.object(proxypage.httpx, "AsyncClient",
                      lambda *a, **k: Dead([])):
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/my-tasks")

    assert res.status_code == 502
    body = res.json()
    assert "no node agent" not in body["error"]
    assert "no_local_node" not in body, (
        "a coordinator that is down would send the reader off to install a node "
        "agent they already have"
    )


def test_the_dashboard_can_say_whether_it_has_a_node():
    import httpx as _httpx

    app = FastAPI()
    app.include_router(proxypage.router)

    class NoNode(Recorder):
        def _record(self, method, url, **kw):
            raise _httpx.ConnectError("no such host")

    proxypage._local_node_seen.update({"answer": None, "at": 0.0})
    with patch.object(proxypage.httpx, "AsyncClient",
                      lambda *a, **k: NoNode([])):
        client = TestClient(app, raise_server_exceptions=False)
        res = client.get("/local-node")

    assert res.status_code == 200
    assert res.json() == {"present": False}
    proxypage._local_node_seen.update({"answer": None, "at": 0.0})
