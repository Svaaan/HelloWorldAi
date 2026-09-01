"""One real job, from an empty database to a downloadable model.

Runs inside docker/docker-compose.smoke.yml against the images that ship. See
that file for why this exists; the short version is that every production
failure this project has had was an integration failure, and none of them was
visible to a unit test.

This plays the node's half of the protocol rather than running the node image,
which is built on nvidia/cuda and wants a graphics card CI does not have. It
registers, proves it holds the key, claims the task, trains it by calling the
same executor the real agent calls, uploads the weights and reports back.
Everything on the other side of those calls is the real thing.

It talks to the dashboard, not to the coordinator, because the proxy is where a
path can quietly reach the wrong service -- /connect-node meant two different
things for months and no contributor's node could register through the public
address at all.
"""

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "/app/src")

from cryptography.hazmat.primitives import hashes, serialization      # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec              # noqa: E402

DASHBOARD = os.environ.get("DASHBOARD_URL", "http://dashboard:3000")
CONTRIBUTOR = os.environ.get("CONTRIBUTOR_DASHBOARD_URL",
                             "http://dashboard_contributor:3000")

SUBMITTER_KEY = "smoke-submitter-key-0123456789abcdef"
TIMEOUT = 180


class Failed(Exception):
    pass


def say(message):
    print("  " + message, flush=True)


def call(method, path, body=None, headers=None, base=None, raw=False,
         timeout=60, expect=None):
    """One HTTP call, with the failure printed rather than a stack trace."""
    url = (base or DASHBOARD) + path
    data = None
    head = dict(headers or {})

    if body is not None:
        if raw:
            data = body
            head.setdefault("Content-Type", "application/octet-stream")
        else:
            data = json.dumps(body).encode()
            head.setdefault("Content-Type", "application/json")

    request = urllib.request.Request(url, data=data, headers=head, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        status = exc.code
    except Exception as exc:
        raise Failed(f"{method} {path} could not be reached: {exc}")

    if expect is not None and status != expect:
        raise Failed(f"{method} {path} returned {status}, expected {expect}\n"
                     f"      {payload[:400].decode('utf-8', 'replace')}")

    if payload[:1] in (b"{", b"["):
        try:
            return status, json.loads(payload)
        except ValueError:
            pass
    return status, payload


# --- being a node ----------------------------------------------------------

def make_keypair():
    """The same shape the browser makes: P-256, public key as base64 SPKI."""
    private = ec.generate_private_key(ec.SECP256R1())
    spki = private.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo)
    return private, base64.b64encode(spki).decode()


def register_node():
    private, public_key = make_keypair()

    _, body = call("POST", "/register-node", expect=200, body={
        "capabilities": {
            "cpu": {"brand": "x86_64", "cores": 2, "threads": 4},
            "gpu": [{
                "name": "Smoke Test GPU", "total_memory": 8192,
                "free_memory": 8192, "load_percentage": 0, "temperature": 40,
                "theoretical_tflops": 10.0,
            }],
            "total_gpu_tflops": 10.0,
        },
        "isConnected": True,
        "isAvailable": True,
        "total_gpu_tflops": 10.0,
        "public_key": public_key,
    })

    if body.get("status") != "success" or not body.get("node_id"):
        raise Failed(f"registration did not return a node_id: {body}")

    node_id = body["node_id"]
    say(f"registered {node_id}")

    # Prove the key, the way the browser does, to get a session token.
    _, challenge = call("GET", f"/generate-challenge/{node_id}", expect=200)
    signature = private.sign(challenge["challenge"].encode(),
                             ec.ECDSA(hashes.SHA256()))

    _, verified = call("POST", f"/verify-challenge/{node_id}", expect=200,
                       body={"signature": signature.hex()})
    if not verified.get("token"):
        raise Failed(f"challenge verified but no token came back: {verified}")

    say("proved the key and took a session token")
    return node_id, {"Authorization": "Bearer " + verified["token"]}


# --- being a submitter -----------------------------------------------------

def make_dataset(rows=600):
    """Two clearly separable classes, so a small model genuinely learns them.

    Not random: verification rejects weights that score no better than an
    untrained model, and it is right to. A dataset with nothing to learn would
    fail this run for a reason that has nothing to do with the plumbing.

    Packed with the project's own pack_dataset. A dataset does not travel as
    CSV -- the browser turns the file into arrays before uploading, and the
    coordinator refuses anything it cannot load without unpickling. Sending raw
    CSV here got as far as submit-task and came back:

        Refused to load artifact: This file contains pickled (object) data.

    which is the artifact loader doing its job on a test that was wrong.
    """
    import numpy as np
    from backend.service.artifacts import pack_dataset

    x = np.zeros((rows, 2), dtype=np.float32)
    y = np.zeros((rows,), dtype=np.int64)
    for i in range(rows):
        base = 1.0 if i % 2 == 0 else 9.0
        x[i] = (base + (i % 7) * 0.01, base + (i % 5) * 0.01)
        y[i] = i % 2

    return pack_dataset(x, y)


def upload_dataset():
    submitter = {"X-Submitter-Key": SUBMITTER_KEY}
    _, body = call("POST", "/artifacts", body=make_dataset(), raw=True,
                   headers=submitter, expect=200, timeout=120)
    artifact_id = body.get("artifact_id")
    if not artifact_id:
        raise Failed(f"upload returned no artifact_id: {body}")
    say(f"uploaded a dataset as {artifact_id}")
    return artifact_id


def submit_job(dataset_id):
    submitter = {"X-Submitter-Key": SUBMITTER_KEY}
    _, body = call("POST", "/submit-task", expect=200, headers=submitter, body={
        "model_name": "smoke-model",
        "architecture": "mlp",
        "dataset_id": dataset_id,
        "hyperparameters": {"steps": 300, "batch_size": 32,
                            "learning_rate": 0.05},
        "hidden_dim": 16,
        "depth": 2,
    })
    task_id = body.get("task_id") or (body.get("task") or {}).get("task_id")
    if not task_id:
        raise Failed(f"submit-task returned no task_id: {body}")
    say(f"submitted {task_id}")
    return task_id


# --- doing the work --------------------------------------------------------

def claim_task(node_id, auth):
    """Poll the way the agent does, until the coordinator hands the job over."""
    deadline = time.time() + 60
    last = None
    while time.time() < deadline:
        status, body = call("GET", f"/next-task/{node_id}", headers=auth)
        last = (status, body)
        # The task arrives nested: {"task": {...}}, or {"task": None} when
        # there is nothing waiting for this node.
        task = body.get("task") if isinstance(body, dict) else None
        if status == 200 and task and task.get("task_id"):
            say(f"claimed {task['task_id']}")
            return task
        time.sleep(2)
    raise Failed(f"the coordinator never handed the task over. Last: {last}")


def train(task, auth):
    """Train with the executor the real agent calls, on the real dataset."""
    from backend.service.artifacts import unpack_dataset, pack_state_dict
    from backend.service.taskExecutor import execute_task

    dataset = None
    dataset_id = task.get("dataset_id")
    if dataset_id:
        _, blob = call("GET", f"/artifacts/{dataset_id}", headers=auth,
                       timeout=120)
        if isinstance(blob, (dict, list)):
            raise Failed(f"the dataset came back as JSON, not bytes: {blob}")
        dataset = unpack_dataset(blob)
        say(f"pulled the dataset back down ({len(blob):,} bytes)")

    logs = []
    outcome = execute_task(task.get("task_data", {}), logs.append, dataset, None)

    state_dict = outcome.get("state_dict")
    if not state_dict:
        raise Failed(f"training produced no weights: {outcome.get('result')}")

    say(f"trained: {outcome.get('result')}")
    return pack_state_dict(state_dict, outcome.get("manifest")), outcome, logs


def report(task_id, blob, outcome, logs, auth):
    _, body = call("POST", "/artifacts?kind=weights", body=blob, raw=True,
                   headers=auth, expect=200, timeout=120)
    weights_id = body.get("artifact_id")
    if not weights_id:
        raise Failed(f"weights upload returned no id: {body}")

    call("POST", f"/task-result/{task_id}", expect=200, headers=auth, body={
        "status": "completed",
        "result": outcome.get("result"),
        "metrics": outcome.get("metrics", {}),
        "weights_id": weights_id,
        "logs": logs,
    })
    say(f"reported the result, weights {weights_id}")
    return weights_id


# --- what the submitter sees -----------------------------------------------

def wait_for_completion(task_id):
    submitter = {"X-Submitter-Key": SUBMITTER_KEY}
    deadline = time.time() + 120
    last = None

    while time.time() < deadline:
        _, jobs = call("GET", "/my-tasks", headers=submitter)
        for job in jobs if isinstance(jobs, list) else []:
            if job.get("task_id") != task_id:
                continue
            last = job
            # Verification runs in the background after the result lands, and
            # its findings are their own object -- `metrics` is what the node
            # reported about itself, `verification` is what the coordinator
            # measured. Waiting on the wrong one waits forever.
            if job.get("status") == "completed" and job.get("verification"):
                return job
            if job.get("status") in ("failed", "rejected", "cancelled"):
                raise Failed(f"the job ended as {job['status']}: "
                             f"{job.get('result') or job.get('error')}")
        time.sleep(2)

    raise Failed(f"the job never finished being verified. Last status: "
                 f"{(last or {}).get('status')}")


def check_verification(job):
    """What the coordinator measured, not what the node claimed.

    This is the part of the product that makes the rest of it worth anything:
    the weights come back from a stranger's machine, and the coordinator
    rebuilds the model from them and scores it on rows that machine never saw.
    A node cannot talk its way to a good result.
    """
    verification = job.get("verification") or {}
    measured = verification.get("measured") or {}
    verdict = verification.get("verdict")

    accuracy = measured.get("holdout_accuracy")
    untrained = measured.get("untrained_accuracy")

    if verdict != "accepted":
        failed = [c for c in verification.get("checks", []) if not c.get("passed")]
        raise Failed(f"verification returned {verdict!r}: {failed or verification}")

    if accuracy is None:
        raise Failed(f"accepted without a holdout score: {verification}")

    say(f"verified: {verdict}, holdout {accuracy} vs untrained {untrained} "
        f"({verification.get('strength')})")

    # If the coordinator ever silently starts trusting a self-reported number,
    # this is where it shows: a model that learned nothing would still pass.
    if untrained is not None and accuracy <= untrained:
        raise Failed(
            f"holdout accuracy {accuracy} is no better than an untrained "
            f"model at {untrained} -- the run completed but nothing was learned")


def check_download(task_id):
    submitter = {"X-Submitter-Key": SUBMITTER_KEY}
    status, body = call("GET", f"/my-tasks/{task_id}/bundle", headers=submitter,
                        timeout=120)
    if status != 200:
        raise Failed(f"the model bundle came back {status}")
    if len(body) < 200:
        raise Failed(f"the bundle is {len(body)} bytes, which is not a model")
    say(f"downloaded the model bundle ({len(body):,} bytes)")


def check_contributor_dashboard():
    """The image built from the five-package requirements file.

    This is the one that crash-looped on `No module named 'bson'`. Compose has
    already waited for its healthcheck, so reaching it at all is most of the
    check; asking for the front door confirms it serves rather than merely
    listens.
    """
    status, _ = call("GET", "/", base=CONTRIBUTOR)
    if status != 200:
        raise Failed(f"the contributor dashboard answered {status}")

    # And the call that needs a node agent must fail as designed rather than
    # hang or leak a DNS error, since there is no agent beside it here.
    status, body = call("POST", "/connect-node", base=CONTRIBUTOR,
                        body={"node_name": "x", "public_key": "y"})
    if status != 503 or not isinstance(body, dict) or not body.get("no_local_node"):
        raise Failed(
            "a dashboard with no node agent should answer 503 with "
            f"no_local_node; got {status} {body}")
    say("the contributor dashboard starts and explains its missing agent")


def main():
    print("\nsmoke: one job, end to end\n", flush=True)
    steps = []

    try:
        check_contributor_dashboard()
        node_id, auth = register_node()
        dataset_id = upload_dataset()
        task_id = submit_job(dataset_id)
        task = claim_task(node_id, auth)
        blob, outcome, logs = train(task, auth)
        report(task_id, blob, outcome, logs, auth)
        job = wait_for_completion(task_id)
        check_verification(job)
        check_download(task_id)
    except Failed as failure:
        print(f"\nFAILED: {failure}\n", flush=True)
        return 1
    except Exception as unexpected:                       # noqa: BLE001
        import traceback
        print("\nFAILED, unexpectedly:\n", flush=True)
        traceback.print_exc()
        return 1

    print("\nsmoke: passed\n", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
